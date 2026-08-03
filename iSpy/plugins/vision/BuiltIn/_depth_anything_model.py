"""Depth Anything V2 (small) inference graph.

Self-contained implementation of the Depth-Anything-V2 DPT model (DINOv2
small encoder + DPT fusion head) so the plugin can run straight from the
bundled `_depth_anything.pt` checkpoint without needing the upstream package.
The checkpoint layout matches the official release at
https://github.com/DepthAnything/Depth-Anything-V2 (keys prefixed with
`pretrained.` and `depth_head.`).

This module requires torch and is imported lazily by the plugin so the rest
of the codebase stays importable without it.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, init_values=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads=num_heads)
        self.ls1 = _LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, patch_size=14):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class DINOv2Small(nn.Module):
    """DINOv2 small backbone (embed dim 384, patch 14) matching the keys in
    the `_depth_anything.pt` checkpoint (prefix `pretrained.`)."""

    PATCH_SIZE = 14
    EMBED_DIM = 384
    DEPTH = 12
    INTERMEDIATE = (2, 5, 8, 11)

    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed(3, self.EMBED_DIM, self.PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, self.EMBED_DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, self.EMBED_DIM))
        self.blocks = nn.ModuleList([_Block(self.EMBED_DIM, num_heads=6) for _ in range(self.DEPTH)])
        self.norm = nn.LayerNorm(self.EMBED_DIM, eps=1e-6)

    def _interpolate_pos_embed(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos = self.pos_embed.float()
        cls_pos = pos[:, 0]
        patch_pos = pos[:, 1:]
        w0 = w // self.PATCH_SIZE + 0.1
        h0 = h // self.PATCH_SIZE + 0.1
        sqrt_n = math.sqrt(N)
        patch_pos = F.interpolate(
            patch_pos.reshape(1, int(sqrt_n), int(sqrt_n), -1).permute(0, 3, 1, 2),
            scale_factor=(float(w0) / sqrt_n, float(h0) / sqrt_n),
            mode="bicubic",
            antialias=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.EMBED_DIM)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1).to(x.dtype)

    def forward_intermediate(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self._interpolate_pos_embed(x, h, w)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.INTERMEDIATE:
                outs.append(self.norm(x))
        return outs


class _ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = _ResidualConvUnit(features)
        self.resConfUnit2 = _ResidualConvUnit(features)

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        modifier = {"size": size} if size is not None else {"scale_factor": 2}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class DPTHead(nn.Module):
    """Depth Anything V2 fusion head (keys prefix `depth_head.`)."""

    def __init__(self, features=64, out_channels=(48, 96, 192, 384)):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(384, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=1, padding=1),
        ])
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.refinenet1 = _FeatureFusionBlock(features)
        self.scratch.refinenet2 = _FeatureFusionBlock(features)
        self.scratch.refinenet3 = _FeatureFusionBlock(features)
        self.scratch.refinenet4 = _FeatureFusionBlock(features)
        self.scratch.output_conv1 = nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


class DepthAnythingV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = DINOv2Small()
        self.depth_head = DPTHead()

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not any(k.startswith("pretrained.") for k in ckpt):
            raise ValueError(f"Not a Depth Anything V2 checkpoint: {path}")
        pretrained = {k[len("pretrained."):]: v for k, v in ckpt.items() if k.startswith("pretrained.")}
        head = {k[len("depth_head."):]: v for k, v in ckpt.items() if k.startswith("depth_head.")}
        missing_p, _ = self.pretrained.load_state_dict(pretrained, strict=False)
        missing_h, _ = self.depth_head.load_state_dict(head, strict=False)
        missing = [m for m in missing_p + missing_h if "num_batches_tracked" not in m]
        if missing:
            raise ValueError(f"Depth checkpoint missing keys: {missing[:10]}")

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.forward_intermediate(x)
        features = [(f[:, 1:], f[:, :1]) for f in features]
        depth = self.depth_head(features, patch_h, patch_w)
        return F.relu(depth).squeeze(1)


def infer_depth(model: DepthAnythingV2, frame_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
    """Run a BGR frame through the model and return a depth map (float32,
    original frame size).  Larger values mean closer surfaces."""
    h, w = frame_bgr.shape[:2]
    scale = max(input_size / h, input_size / w)
    sw = max(int(round(scale * w / 14)) * 14, 14)
    sh = max(int(round(scale * h / 14)) * 14, 14)

    rgb = cv2_resize_rgb(frame_bgr, (sw, sh))
    img = rgb.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = torch.from_numpy(np.ascontiguousarray((img - mean) / std, dtype=np.float32)).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.no_grad():
        depth = model(tensor.to(device))[0].float().cpu().numpy()
    return cv2_resize(depth, (w, h))


def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def cv2_resize_rgb(frame_bgr, size):
    import cv2
    return cv2.cvtColor(cv2.resize(frame_bgr, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) / 255.0
