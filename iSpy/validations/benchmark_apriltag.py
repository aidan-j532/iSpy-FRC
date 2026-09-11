"""Benchmark AprilTag detection backends (OpenCV vs native AprilTag 3).

Renders synthetic frames with ground-truth tag poses from the same 36h11
dictionary and runs each backend over identical input, reporting detection
time, FPS, success rate, corner error and pose error. Run it on the target
hardware (Orange Pi 5/RK3588) to decide which backend to enable:

    python -m iSpy.validations.benchmark_apriltag

The native backend needs `pip install apriltag-python` (ships prebuilt wheels
for win/linux x86 and linux aarch64). Missing backend is skipped, not fatal.
Its `blur` and `decimate` options are exposed as flags; on noisy frames
`--blur 1.0` (as used in most deployments) cuts the adaptive-threshold noise
cost dramatically.
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (_PROJECT_ROOT / "iSpy").is_dir():
    _PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_OCV_PARAMS = cv2.aruco.DetectorParameters()


def _native_detector(threads: int = 1, blur: float = 0.0, decimate: float = 2.0):
    from apriltag import apriltag as _apriltag

    return _apriltag("tag36h11", threads=threads, blur=blur, decimate=decimate)


def _native_available() -> bool:
    try:
        import apriltag  # noqa: F401

        return True
    except Exception:
        return False


def make_intrinsics(width: int, height: int, f: float | None = None):
    f = f or max(width, height) * 0.9
    return np.array(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64
    )


def _tag_texture(tag_id: int, px: int = 240):
    return cv2.aruco.generateImageMarker(DICT, tag_id, px)


def _object_points(half_m: float):
    # Tag corners in the standard OpenCV aruco order (tl, tr, br, bl) with the
    # pattern's +y pointing up in the object frame. With these points a tag that
    # is visible (pattern facing the camera) needs a pose whose rotation also
    # contains the x-flip that maps the pattern's front to the camera - the
    # renderer applies that base rotation in `_tag_specs`.
    return np.array(
        [
            [-half_m, half_m, 0.0],
            [half_m, half_m, 0.0],
            [half_m, -half_m, 0.0],
            [-half_m, -half_m, 0.0],
        ],
        dtype=np.float32,
    )


def _project_corners(rvec, tvec, half_m: float, cam_mat):
    pts, _ = cv2.projectPoints(
        _object_points(half_m), rvec, tvec, cam_mat, np.zeros(5)
    )
    return pts.reshape(-1, 2)


class Renderer:
    def __init__(self, width=640, height=480, tag_size_m=0.165):
        self.width = width
        self.height = height
        self.tag_size_m = tag_size_m
        self.cam_mat = make_intrinsics(width, height)
        self._textures: dict[int, np.ndarray] = {}

    def background(self, rng):
        # A scene-like backdrop: brightness gradient, a few soft contrast
        # patches (carpet/floor marks) and mild sensor noise. A pure gradient
        # or heavy per-pixel uniform noise is pathological for the detectors'
        # adaptive thresholds and biases the timing comparison.
        img = np.full((self.height, self.width), 170.0, np.float32)
        img += np.linspace(0, 40, self.height, dtype=np.float32)[:, None]
        h, w = img.shape
        for _ in range(8):
            cx = int(rng.uniform(0, w))
            cy = int(rng.uniform(0, h))
            sigma = float(rng.uniform(80, 200))
            amp = float(rng.uniform(-30, 30))
            yy, xx = np.mgrid[0:h, 0:w]
            img += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)))
        img += rng.normal(0.0, 2.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, tag_specs, rng):
        frame = self.background(rng)
        truths = []
        for tag_id, rvec, tvec in tag_specs:
            corners = _project_corners(rvec, tvec, self.tag_size_m / 2.0, self.cam_mat)
            if corners[:, 0].min() < 4 or corners[:, 0].max() > self.width - 4:
                continue
            if corners[:, 1].min() < 4 or corners[:, 1].max() > self.height - 4:
                continue
            tex = self._textures.setdefault(tag_id, _tag_texture(tag_id))
            s = tex.shape[0]
            dst = np.array(
                [[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32
            )
            H, _ = cv2.findHomography(dst, corners.astype(np.float32))
            warped = cv2.warpPerspective(
                tex,
                H,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            # paste the whole quad (black code cells included) - a threshold
            # on the warped texture alone would drop the dark cells and leave
            # background showing through the pattern.
            mask = cv2.warpPerspective(
                np.full((s, s), 255, np.uint8),
                H,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            where = mask > 127
            frame[where] = warped[where]
            truths.append(
                {
                    "id": tag_id,
                    "rvec": rvec,
                    "tvec": tvec,
                    "corners": corners,
                }
            )
        return frame, truths


# A tag whose rotation is the identity (in the y-up object frame of
# `_object_points`) has its patterned back facing the camera, so it cannot be
# detected. Real visible tags carry an extra 180 deg rotation about x that
# turns their front toward the camera; apply it to every generated pose so the
# benchmark only produces detectable tags.
_TAG_FRONT = cv2.Rodrigues(np.array([math.pi, 0.0, 0.0], dtype=np.float64))[0]


def _tag_specs(rng, n: int, cam_mat, z_m=(0.9, 3.0)):
    w, h = cam_mat[0, 2] * 2, cam_mat[1, 2] * 2
    specs = []
    ids = rng.choice(587, n, replace=False)
    for tag_id in ids:
        for _ in range(40):
            z = rng.uniform(*z_m)
            x = rng.uniform(-0.5, 0.5)
            y = rng.uniform(-0.4, 0.4)
            yaw = rng.uniform(0.0, 2 * math.pi)
            pitch = rng.uniform(-0.35, 0.35)
            roll = rng.uniform(-0.35, 0.35)
            r_base = cv2.Rodrigues(
                np.array([roll, pitch, yaw], dtype=np.float64).reshape(3)
            )[0]
            rvec = cv2.Rodrigues(r_base @ _TAG_FRONT)[0]
            tvec = np.array([x, y, z], dtype=np.float64).reshape(3, 1)
            frame_c = _project_corners(rvec, tvec, 0.165 / 2.0, cam_mat)
            if (
                frame_c[:, 0].min() > 8
                and frame_c[:, 0].max() < w - 8
                and frame_c[:, 1].min() > 8
                and frame_c[:, 1].max() < h - 8
            ):
                specs.append((int(tag_id), rvec, tvec))
                break
    return specs


def _angle_error(r_a, r_b) -> float:
    ra = cv2.Rodrigues(np.asarray(r_a).reshape(3))[0]
    rb = cv2.Rodrigues(np.asarray(r_b).reshape(3))[0]
    cos = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


# apriltag-python reports corners in (tr, tl, bl, br) for a tag at 0 yaw;
# OpenCV reports (tl, tr, br, bl). solvePnP below uses the OpenCV convention,
# so reorder before feeding it.
_APRILTAG_TO_OPENCV = (1, 0, 3, 2)


def _reorder_at(corners):
    return np.asarray(corners)[list(_APRILTAG_TO_OPENCV)]


class OpenCVBackend:
    name = "opencv"

    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(DICT, _OCV_PARAMS)

    def detect(self, gray):
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        return [(int(ids[i][0]), np.asarray(corners[i][0], dtype=np.float64)) for i in range(len(ids))] if ids is not None else []


class AprilTagBackend:
    def __init__(self, threads=1, blur=0.0, decimate=2.0):
        self.threads = threads
        self.blur = blur
        self.decimate = decimate
        self.name = f"apriltag_t{threads}"
        if blur:
            self.name += f"_b{blur}"
        if decimate and decimate != 2.0:
            self.name += f"_d{decimate}"
        self.detector = _native_detector(threads, blur, decimate)

    def detect(self, gray):
        out = []
        for d in self.detector.detect(gray):
            out.append((int(d["id"]), _reorder_at(d["lb-rb-rt-lt"]).astype(np.float64)))
        return out


def _augment_truth(truths, tag_size_m):
    out = []
    obj = _object_points(tag_size_m / 2.0)
    for t in truths:
        out.append({"id": t["id"], "rvec": t["rvec"], "tvec": t["tvec"], "corners": t["corners"], "obj": obj})
    return out


def _solve_pose(obj, corners, cam_mat, rvec_true, tvec_true):
    """solvePnP over both IPPE solutions, returning errors to the true pose.

    Planar pose recovery is ambiguous (front/back), so score the best of the
    two solutions IPPE returns rather than whichever one `solvePnP` happens to
    pick.
    """
    _ok, rvecs, tvecs, _reproj = cv2.solvePnPGeneric(
        obj, corners.astype(np.float32), cam_mat, np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    best = None
    for rv, tv in zip(rvecs, tvecs):
        r_err = _angle_error(rv, rvec_true)
        t_err = float(np.linalg.norm(tv.ravel() - np.asarray(tvec_true).ravel()))
        # prefer the solution that matches rotation and keep its translation;
        # score ties by translation.
        key = (r_err, t_err)
        if best is None or key < best[0]:
            best = (key, r_err, t_err)
    return best[1], best[2]


def _score(backend, frames, cam_mat, tag_size_m):
    results = []
    for frame, truths in frames:
        corner_errs = []
        t_errors = []
        r_errors = []
        t0 = time.perf_counter()
        dets = backend.detect(frame)
        dt = (time.perf_counter() - t0) * 1000
        truth_by_id = {t["id"]: t for t in truths}
        matched = 0
        for tag_id, corners in dets:
            truth = truth_by_id.get(tag_id)
            if truth is None:
                continue
            matched += 1
            errs = np.linalg.norm(corners - np.asarray(truth["corners"]), axis=1)
            corner_errs.append(float(errs.mean()))
            r_err, t_err = _solve_pose(
                truth["obj"], corners, cam_mat, truth["rvec"], truth["tvec"]
            )
            t_errors.append(t_err)
            r_errors.append(r_err)
        results.append(
            {
                "n_gt": len(truths),
                "n_detected": len(dets),
                "dt_ms": dt,
                "recall": matched / len(truths) if truths else None,
                "corner_err": float(np.mean(corner_errs)) if corner_errs else None,
                "t_err": float(np.mean(t_errors)) if t_errors else None,
                "r_err": float(np.mean(r_errors)) if r_errors else None,
            }
        )
    return results


def _fmt(n):
    return "  --" if n is None else f"{n:6.3f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark AprilTag detectors")
    parser.add_argument("--tags", default="6,3,1,0", help="comma list of tags per frame")
    parser.add_argument("--frames", type=int, default=24, help="frames per tag count")
    parser.add_argument("--runs", type=int, default=3, help="timing repeats over the frame set")
    parser.add_argument("--threads", type=int, default=1, help="native backend threads")
    parser.add_argument("--blur", type=float, default=0.0, help="native backend gaussian blur sigma (1.0 helps noisy frames)")
    parser.add_argument("--decimate", type=float, default=2.0, help="native backend detection downscale factor")
    parser.add_argument("--output", default=None, help="output json path")
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    renderer = Renderer(640, 480)
    sets = []
    for n in (int(x) for x in args.tags.split(",") if x.strip()):
        for _ in range(args.frames):
            if n > 0:
                specs = _tag_specs(rng, n, renderer.cam_mat)
                frame, truths = renderer.render(specs, rng)
            else:
                frame, truths = renderer.background(rng), []
            sets.append((frame, _augment_truth(truths, renderer.tag_size_m)))

    backends = [OpenCVBackend()]
    if _native_available():
        backends.append(AprilTagBackend(args.threads, args.blur, args.decimate))
    else:
        print("note: apriltag-python not installed - only benchmarking OpenCV")

    print(f"platform: {platform.platform()}  python: {sys.version.split()[0]}")
    print(f"frames: {args.frames} per tag count, {args.runs} repeats")
    print(
        f"{'backend':<14} {'tags':>4} {'av_ms':>7} {'p95_ms':>7} {'max_ms':>7}"
        f" {'fps':>6} {'recall':>7} {'corner':>8} {'t_err':>7} {'r_err':>7}"
    )

    cam_mat = renderer.cam_mat
    all_rows = []
    totals = {b.name: [] for b in backends}

    for n in sorted({len(s[1]) for s in sets}):
        subset = [s for s in sets if len(s[1]) == n]
        for backend in backends:
            times = []
            rows = None
            for _ in range(args.runs):
                rows = _score(backend, subset, cam_mat, renderer.tag_size_m)
                times.extend(r["dt_ms"] for r in rows)
            times = np.array(times)
            recalls = [r["recall"] for r in rows if r["recall"] is not None]
            recall = float(np.mean(recalls)) if recalls else float("nan")
            ce = np.nanmean([r["corner_err"] for r in rows if r["corner_err"] is not None])
            te = np.nanmean([r["t_err"] for r in rows if r["t_err"] is not None])
            re = np.nanmean([r["r_err"] for r in rows if r["r_err"] is not None])
            avg = float(np.mean(times))
            p95 = float(np.percentile(times, 95))
            mx = float(np.max(times))
            totals[backend.name].append(avg)
            print(
                f"{backend.name:<14} {n:>4} {avg:7.2f} {p95:7.2f} {mx:7.2f}"
                f" {1000.0 / avg:6.1f} {recall:7.2f} {_fmt(ce)} {_fmt(te)} {_fmt(re)}"
            )
            all_rows.append(
                {
                    "backend": backend.name,
                    "n_tags": n,
                    "avg_ms": round(avg, 2),
                    "p95_ms": round(p95, 2),
                    "max_ms": round(mx, 2),
                    "recall": round(recall, 3),
                    "corner_err_px": None if ce is None or math.isnan(ce) else round(ce, 3),
                    "t_err_m": None if te is None or math.isnan(te) else round(te, 3),
                    "r_err_deg": None if re is None or math.isnan(re) else round(re, 3),
                }
            )

    print()
    print("mean per-detector time (all tag counts):")
    for b in backends:
        m = float(np.mean(totals[b.name]))
        print(f"  {b.name:<14}{m:7.2f} ms  {1000.0 / m:6.1f} fps")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {"platform": platform.platform(), "rows": all_rows}, f, indent=2
            )
        print(f"\nresults saved to {out}")


if __name__ == "__main__":
    sys.exit(main())