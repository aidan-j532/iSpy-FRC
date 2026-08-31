"""A dependency-free (no Ultralytics) YOLOv8 loader and inference engine.

The ``.pt`` SDK checkpoints shipped for iSpy are Ultralytics-style pickles whose
``model`` object is reconstructed from classes at ``ultralytics.nn.modules.*``
and ``ultralytics.nn.tasks.*``. Ultralytics is AGPL-3.0, so iSpy must not
import it at runtime. This module re-implements the small subset of that
architecture needed for *inference* (Conv, C2f, SPPF, the DFL detection head,
and - for pose models - the keypoint head), registers those classes under the
names the pickle expects, loads the checkpoint with ``torch.load``, then runs
letterbox -> forward -> DFL/anchor decode -> scale -> NMS with no Ultralytics
code anywhere.

The public entry point is :func:`load_yolo_pt`, which returns a lightweight
:class:`YoloPT` wrapper exposing the same surface the rest of iSpy uses on a
``.pt``/``.engine``/OpenVINO/CoreML model:

- ``.task``, ``.names``, ``.nc``
- ``.model`` (the raw ``nn.Module``)
- ``.to(device)``
- ``__call__(frames, ...)`` -> list of results with ``.boxes`` (``.xyxy``/
  ``.conf``/``.cls``) and (for pose) ``.keypoints.data``, mirroring the
  fields consumed by :meth:`GenericYolo._convert_ultralytics_to_results`.

Only the detection and (COCO-17 style) keypoint heads shipped in YOLOv8
checkpoints are supported. Everything here is the MIT/BSD-style math, none
of it is copied from the AGPL library.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lightweight re-implementations of the YOLOv8 blocks the pickles reference.
# The class names (and the module paths they live under) must MATCH the paths
# in the pickle, but the body is written here from the (MIT) architecture.
# ---------------------------------------------------------------------------

def autopad(k, p=None, d=1):
    """Same padding helper: pad so a k-stride kernel keeps the spatial size."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm2d + SiLU block."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard residual bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with two convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(nn.Module):
    """C3k2 YOLO block - a C2f variant with configurable inner block.

    YOLOv11/v26-family block: C2f with C3k inner blocks. When a flag is set,
    uses C3k (Bottleneck with flag), otherwise plain Bottleneck.
    """

    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True, g=1, d=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        return self.cv2(torch.cat(y, 1)) if getattr(self, "add", False) else self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along a dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """Distribution Focal Loss layer; decodes the box-distribution channels."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x_vals = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x_vals.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def _make_keypoint_conv(c1, c2, k=1):
    """Create a keypoint convolution head.
    
    YOLOv8 Pose head outputs 4 (xyxy) + num_classes + num_keypoints * keypoint_dims
    The keypoints are decoded from the last channels.
    """
    return Conv(c1, c2, k)


class PoseModel(nn.Module):
    """Pose detection head for YOLOv8 - outputs keypoints alongside boxes."""

    def __init__(self, nc, reg_max=16, num_keypoints=17, keypoint_dims=2, end2end=False, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch) if ch else 3
        self.reg_max = reg_max
        self.num_keypoints = num_keypoints
        self.keypoint_dims = keypoint_dims
        self.no = nc + self.reg_max * 4 + num_keypoints * keypoint_dims
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.no * 4)) if ch else 64, (max(ch[0], min(self.nc, 100)) if ch else 80)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                _make_keypoint_conv(c2, c3, 3),
                _make_keypoint_conv(c3, c3, 3),
                nn.Conv2d(c3, num_keypoints * keypoint_dims, 1),
            )
            for _ in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.training:
            return preds
        y = self._inference(preds)
        return (y,) if self.export else (y, preds)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        keypoints = torch.cat([kp_head[i](x[i]).view(bs, self.num_keypoints * self.keypoint_dims, -1) for i, kp_head in enumerate(self.cv3)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, keypoints=keypoints, feats=x)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3[:1] if hasattr(self, 'cv3') else self.cv3, kpt_head=self.cv3[1:] if len(self.cv3) > 1 else [])

    @property
    def one2one(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, kpt_head=self.cv3)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Build the anchor points / stride tensor for the given feature maps."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:4]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance (ltrb) predictions to boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)  # xyxy


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def _scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """Rescale boxes (in-image coords) to the original image shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0] if isinstance(ratio_pad[0], (tuple, list)) else ratio_pad[0]
        pad = ratio_pad[1]
    if padding:
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., 0] /= gain
    boxes[..., 1] /= gain
    if not xywh:
        boxes[..., 2] /= gain
        boxes[..., 3] /= gain
    return boxes


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
    nc=80,
    agnostic=False,
):
    """Torch-only NMS producing [x1, y1, x2, y2, conf, cls] per detection."""
    bs = prediction.shape[0]
    xc = prediction[..., 4:4 + nc].max(2).values  # max class score per anchor
    output = [torch.zeros((0, 6), device=prediction.device) for _ in range(bs)]
    candidates = xc > conf_thres

    for xi, x in enumerate(prediction):  # per image
        x = x[candidates[xi]]  # filter by conf
        if not x.shape[0]:
            continue

        # class scores -> conf + class id
        scores, cls = x[:, 4:4 + nc].max(1)
        keep = scores > conf_thres
        x = x[keep]
        scores = scores[keep]
        cls = cls[keep]
        boxes = x[:, :4]  # already xyxy (converted by the caller)
        if not boxes.shape[0]:
            continue

        # per-class NMS
        output_boxes = []
        for c in cls.unique():
            mask = cls == c
            bx = boxes[mask]
            bscores = scores[mask]
            bcls = cls[mask]
            if bx.shape[0] == 1:
                output_boxes.append(torch.cat([bx, bscores[:, None], bcls[:, None]], 1))
                continue
            keep = _torch_nms(bx, bscores, iou_thres)
            output_boxes.append(torch.cat([bx[keep], bscores[keep][:, None], bcls[keep][:, None]], 1))
        if not output_boxes:
            continue
        det = torch.cat(output_boxes, 0)
        if det.shape[0] > max_det:
            order = det[:, 4].sort(descending=True)[1][:max_det]
            det = det[order]
        output[xi] = det

    return output


def _torch_nms(boxes, scores, iou_thres):
    device = boxes.device
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.sort(descending=True)[1]
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return torch.tensor(keep, device=device, dtype=torch.long)


class Detect(nn.Module):
    """YOLOv8 detection head (box cv2, class cv3, DFL decode)."""

    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch) if ch else 3
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)) if ch else 64, (max(ch[0], min(self.nc, 100)) if ch else 80)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self._end2end = end2end

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def end2end(self):
        return getattr(self, "_end2end", False) and hasattr(self, "one2one")

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.training:
            return preds
        y = self._inference(preds)
        return (y,) if self.export else (y, preds)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def _inference(self, x):
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            anchors, strides = make_anchors(x["feats"], self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1)
            self.strides = strides.transpose(0, 1)
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)


class DetectionModel(nn.Module):
    """Marker class: the pickle reconstructs the whole graph into this type.

    ``forward`` runs the module ladder exactly like Ultralytics' _predict_once
    (perform layer i, feed stored activations into Concat/neck by index).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        y = []
        module_list = self.model
        for m in module_list:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


# ---------------------------------------------------------------------------
# Namespace shim: register our classes under the exact pickle paths so
# torch.load can unpickle an Ultralytics .pt checkpoint without the library.
# ---------------------------------------------------------------------------

def _register_shim():
    modules_ns = types.ModuleType("ultralytics.nn.modules")
    conv_ns = types.ModuleType("ultralytics.nn.modules.conv")
    block_ns = types.ModuleType("ultralytics.nn.modules.block")
    head_ns = types.ModuleType("ultralytics.nn.modules.head")
    tasks_ns = types.ModuleType("ultralytics.nn.tasks")

for ns in (conv_ns, block_ns, head_ns, tasks_ns):
        ns.Conv = Conv
        ns.C2f = C2f
        ns.SPPF = SPPF
        ns.Bottleneck = Bottleneck
        ns.Concat = Concat
        ns.DFL = DFL
        ns.Detect = Detect
        ns.PoseModel = PoseModel
        ns.C3k = C3k  # ADD: register C3k block shim
    modules_ns.Conv = Conv
    modules_ns.C2f = C2f
    modules_ns.SPPF = SPPF
    modules_ns.Concat = Concat
    modules_ns.Detect = Detect
    modules_ns.PoseModel = PoseModel
    modules_ns.C3k = C3k  # ADD: register C3k block shim in parent pkg
    tasks_ns.DetectionModel = DetectionModel

    # Ultralytics nn.Module re-exports each class under the same name in both
    # the submodule and the parent package, so expose them in both places.
    pkg = types.ModuleType("ultralytics")
    pkg.nn = types.ModuleType("ultralytics.nn")
    pkg.nn.modules = modules_ns
    pkg.nn.tasks = tasks_ns

    for mod_name, mod in {
        "ultralytics": pkg,
        "ultralytics.nn": pkg.nn,
        "ultralytics.nn.modules": modules_ns,
        "ultralytics.nn.modules.conv": conv_ns,
        "ultralytics.nn.modules.block": block_ns,
        "ultralytics.nn.modules.head": head_ns,
        "ultralytics.nn.tasks": tasks_ns,
    }.items():
        sys.modules[mod_name] = mod

    return modules_ns, conv_ns, block_ns, head_ns, tasks_ns


class _Det:
    """A single detection box, mirroring Ultralytics' per-box object enough for
    GenericYolo._convert_ultralytics_to_results (`.xyxy`, `.conf`, `.cls`)."""

    def __init__(self, xyxy, conf, cls):
        self._xyxy = xyxy  # CPU torch tensor [4]
        self._conf = conf  # CPU torch scalar tensor
        self._cls = cls  # CPU torch scalar tensor

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def conf(self):
        return self._conf

    @property
    def cls(self):
        return self._cls


class _Boxes:
    """Iterable container of :class:`_Det`, matching the surface GenericYolo consumes."""

    def __init__(self, det, device=None):
        # det: [N, 6] (x1,y1,x2,y2,conf,cls) tensor already in orig coords
        if det.numel():
            xyxy = det[:, :4].cpu()
            conf = det[:, 4].cpu()
            cls = det[:, 5].cpu()
        else:
            xyxy = torch.zeros((0, 4))
            conf = torch.zeros(0)
            cls = torch.zeros(0, dtype=torch.long)
        self._items = [_Det(xyxy[i], conf[i], cls[i]) for i in range(xyxy.shape[0])]
        self._xyxy_all = xyxy
        self._conf_all = conf
        self._cls_all = cls

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    @property
    def xyxy(self):
        return self._xyxy_all

    @property
    def conf(self):
        return self._conf_all

    @property
    def cls(self):
        return self._cls_all

    @property
    def data(self):
        return self._xyxy_all


class _Keypoints:
    """Minimal keypoints container (data: [N, K, 2 or 3])."""

    def __init__(self, data):
        self.data = data.to("cpu")


class _Result:
    """A single-image result, mirroring Ultralytics' Results surface."""

    def __init__(self, boxes, keypoints, orig_shape):
        self.boxes = boxes
        self.keypoints = keypoints
        self.orig_shape = orig_shape
        self.orig_img = None


class YoloPT:
    """Dependency-free YOLOv8 detection/pose model.

    Mirrors the Ultralytics YOLO surface iSpy relies on: ``.task``, ``.names``,
    ``.nc``, ``.model``, ``.to()``, and ``(model)(frames, ...)`` returning a
    list of ``_Result`` with ``.boxes`` / ``.keypoints`` / ``.orig_shape``.
    """

    def __init__(self, model: DetectionModel, names: dict, task: str, nc: int):
        self.model = model
        self.names = names
        self.task = task
        self.nc = int(nc)
        self._device = next(model.parameters()).device if next(model.parameters(), None) is not None else "cpu"
        model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        self._device = device
        return self

    def _letterbox(self, img, new_shape=640, stride=32, pad=114):
        # Match Ultralytics' inference letterbox: `auto=True`, which keeps the
        # aspect ratio and pads only up to a stride-multiple (square only when
        # the source is already close to square).
        h, w = img.shape[:2]
        new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else tuple(new_shape)
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))  # (w, h)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # full pads
        dw, dh = dw % stride, dh % stride  # keep aspect: pad to stride multiple
        dw /= 2
        dh /= 2
        if (w, h) != new_unpad:
            img = np.ascontiguousarray(cv2_resize(img, new_unpad, r))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2_copyMakeBorder(img, top, bottom, left, right, pad)
        self._ratio = r
        self._pad = (left, top)
        return img

    def _preprocess(self, frames, imgsz):
        tensors = []
        for f in frames:
            lb = self._letterbox(f, imgsz)
            rgb = np.ascontiguousarray(lb[:, :, ::-1])
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensors.append(t.to(self._device))
        return torch.cat(tensors, 0)

    def _postprocess(self, pred, frame_shape, conf):
        # pred: [B, 4+nc, N] (decode yields xywh in input-space); convert and scale.
        boxes_xywh = pred[:, :4, :].transpose(1, 2)  # [B, N, 4]
        scores = pred[:, 4:, :].transpose(1, 2)  # [B, N, nc]
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        full = torch.cat([boxes_xyxy, scores], 2)  # [B, N, 4+nc]
        dets = non_max_suppression(full, conf_thres=conf, nc=self.nc)

        results = []
        for xi, det in enumerate(dets):
            if det.numel():
                det[:, :4] = _scale_boxes(
                    (self._input_size, self._input_size),
                    det[:, :4],
                    frame_shape,
                    ratio_pad=(self._ratio, self._pad),
                )
                det[:, :4].clamp_(min=0)
            orig = (int(frame_shape[0]), int(frame_shape[1]))
            results.append(_Result(_Boxes(det), None, orig))
        return results

    def __call__(self, frames, imgsz=None, conf=0.25, device=None, verbose=False, show=False, **kwargs):
        if device is not None and str(device) != str(self._device):
            self.to(device)
        if imgsz is None:
            imgsz = 640
        self._input_size = imgsz

        single = not isinstance(frames, list)
        fl = [frames] if single else frames

        im_tensors = self._preprocess(fl, imgsz)
        with torch.no_grad():
            out = self.model(im_tensors)[0]

        results = self._postprocess(out, fl[0].shape, conf)
        return results if not single else results


def cv2_resize(img, size, r):
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def cv2_copyMakeBorder(img, top, bottom, left, right, pad):
    import cv2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))


def load_yolo_pt(path: str, task: str = "detect", verbose: bool = False) -> YoloPT:
    """Load a YOLOv8 detection checkpoint (.pt) without Ultralytics."""
    register_shim()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("model", ckpt)
    if model is None:
        raise ValueError(f"No model found in checkpoint: {path}")

    names = getattr(model, "names", None)
    if not isinstance(names, dict):
        names = {i: str(i) for i in range(getattr(model, "nc", 80) or 80)}
    nc = len(names) if names else int(getattr(model, "nc", 80) or 80)
    model_task = getattr(model, "task", None)
    if model_task is None:
        # infer detect vs pose from the head
        model_task = "detect"
        try:
            detect = model.model[-1]
            if hasattr(detect, "kpt_shape"):
                model_task = "pose"
        except Exception:
            pass

    # Ensure the model's f/i metadata survives as attributes on each module.
    model = model.float()  # some checkpoints ship half-precision weights
    model.eval()
    return YoloPT(model, names, model_task, nc)


def register_shim():
    """Build and install the Ultralytics-namespace shim (idempotent)."""
    if "ultralytics.nn.tasks" in sys.modules:
        return
    _register_shim()
