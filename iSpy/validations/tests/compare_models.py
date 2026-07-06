"""
Compare a base .pt YOLO model against an optimized/converted model (RKNN,
ONNX, TFLite, TensorRT .engine, CoreML, etc.) to verify a hardware-accelerator
conversion didn't silently break detection quality or somehow make things
slower.

This intentionally reuses the exact same inference pipeline the rest of iSpy
runs in production - GenericYolo + fill_missing_config + metadata sidecars -
instead of re-implementing pre/post-processing here. That way the comparison
reflects real runtime behavior, not a synthetic stand-in for it.

Supersedes test_rknn.py, which was RKNN-only. This works for any backend
GenericYolo supports (RKNN, ONNX, TFLite, TensorRT, OpenVINO, CoreML, TPU).

Usage:
    # Reads vision_model.source_pt (base) + vision_model.file_path (optimized)
    # straight out of a normal iSpy config.json (this is exactly what boot.py
    # writes, so most of the time you don't need to pass anything else).
    python -m iSpy.validations.test_optimized_model --config Config/config.json

    # Or point at models explicitly:
    python -m iSpy.validations.test_optimized_model \
        --base YoloModels/pytorch/model.pt \
        --optimized YoloModels/rknn/model.rknn \
        --core-mask 7 --num-images 10

Test images are pulled at random from QuantizeDataset/valid/ (any
.jpg/.jpeg/.png/.bmp found recursively under that folder).
"""

# Made by Claude

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.resolve()
if not (_PROJECT_ROOT / "iSpy").is_dir():
    _PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("RKNN_LOG_LEVEL", "3")
os.environ.setdefault("YOLO_VERBOSE", "False")

# Same pipeline iSpy.py / game_loop.py / ObjectDetectionCamera.py run in
# production - not re-implemented here.
from iSpy.vision.genericYolo import GenericYolo, Box  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─── Console formatting ───────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


_USE_COLOR = sys.stdout.isatty()


def _c(text, color):
    return f"{color}{text}{C.RESET}" if _USE_COLOR else text


def _status(status):
    return {
        "PASS": _c("PASS", C.GREEN),
        "FAIL": _c("FAIL", C.RED),
        "WARN": _c("WARN", C.YELLOW),
        "SKIP": _c("SKIP", C.DIM),
    }.get(status, status)


def section(title: str):
    width = 74
    print()
    print(_c("=" * width, C.CYAN))
    print(_c(f"  {title}", C.BOLD + C.CYAN))
    print(_c("=" * width, C.CYAN))


def subline(label: str, value: str, status: str | None = None):
    tag = f"  [{_status(status)}]" if status else ""
    print(f"  {label:<38} {value}{tag}")


def warn(msg: str):
    print(_c(f"  ! {msg}", C.YELLOW))


@contextlib.contextmanager
def _quiet_native():
    """Suppress native C-library stdout/stderr spam (RKNN toolkit, etc. print
    directly to fd 1/2, bypassing Python logging entirely)."""
    devnull = "nul" if os.name == "nt" else "/dev/null"
    fd = os.open(devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)


# ─── Config / model path resolution ──────────────────────────────────────────

def _load_config_paths(
    config_path: str | None, base_override: str | None, optimized_override: str | None
) -> tuple[str, str]:
    base_path = base_override
    optimized_path = optimized_override

    if (not base_path or not optimized_path) and config_path:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with open(cfg_file) as f:
                data = json.load(f)
            # Works with either a full iSpy config.json (top-level
            # "vision_model" key) or a bare vision_model dict.
            vm = data.get("vision_model", data)
            optimized_path = optimized_path or vm.get("file_path")
            base_path = base_path or vm.get("source_pt")
        else:
            logger.warning("Config file not found: %s (falling back to --base/--optimized)", cfg_file)

    if not base_path or not optimized_path:
        logger.error(
            "Could not resolve both model paths. Provide --config pointing at "
            "a config.json with vision_model.source_pt (base) + "
            "vision_model.file_path (optimized) set, or pass --base and "
            "--optimized explicitly."
        )
        sys.exit(1)

    base_path = str(base_path)
    optimized_path = str(optimized_path)

    if not Path(base_path).is_absolute():
        base_path = str(_PROJECT_ROOT / base_path)
    if not Path(optimized_path).is_absolute():
        optimized_path = str(_PROJECT_ROOT / optimized_path)

    if not Path(base_path).exists():
        logger.error("Base model not found: %s", base_path)
        sys.exit(1)
    if not Path(optimized_path).exists():
        logger.error("Optimized model not found: %s", optimized_path)
        sys.exit(1)

    if Path(base_path).suffix.lower() == ".pt" and Path(optimized_path).suffix.lower() == ".pt":
        logger.error(
            "Both the base and optimized model are .pt files (%s vs %s) - "
            "there's nothing to compare here. Point --optimized (or "
            "vision_model.file_path in your config) at a converted model "
            "instead - .rknn, .onnx, .tflite, .engine, .mlpackage, etc.",
            base_path,
            optimized_path,
        )
        sys.exit(1)

    return base_path, optimized_path


def _find_test_images(images_dir: Path, num_images: int, seed: int | None) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    found: set[Path] = set()
    for ext in exts:
        found.update(images_dir.rglob(ext))
        found.update(images_dir.rglob(ext.upper()))
    found_list = sorted(found)

    if not found_list:
        logger.error("No test images found under %s", images_dir)
        sys.exit(1)

    if seed is not None:
        random.seed(seed)
    if len(found_list) <= num_images:
        return found_list
    return random.sample(found_list, num_images)


# ─── IoU / matching helpers ───────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(ref_boxes: list[Box], test_boxes: list[Box], iou_thresh: float):
    """Greedily match each ref (base .pt) box to its best same-class IoU
    partner in test_boxes (optimized model). Returns (ref_box, test_box, iou)
    triples for every match found."""
    used: set[int] = set()
    matches = []
    for rb in ref_boxes:
        best_iou, best_idx = 0.0, -1
        for i, tb in enumerate(test_boxes):
            if i in used or tb.cls_id != rb.cls_id:
                continue
            v = _iou(rb.xyxy, tb.xyxy)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_idx >= 0 and best_iou >= iou_thresh:
            used.add(best_idx)
            matches.append((rb, test_boxes[best_idx], best_iou))
    return matches


# ─── Speed test ───────────────────────────────────────────────────────────────

def _measure_speed(model: GenericYolo, frames: list[np.ndarray], duration: float) -> dict:
    for f in frames[: min(3, len(frames))]:
        model.predict(f, orig_shape=f.shape)  # warm up

    count = 0
    idx = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        frame = frames[idx % len(frames)]
        model.predict(frame, orig_shape=frame.shape)
        count += 1
        idx += 1
    elapsed = time.perf_counter() - start

    if count == 0:
        return {"fps": 0.0, "inference_ms": None, "frames": 0, "elapsed_s": elapsed}
    return {
        "fps": count / elapsed,
        "inference_ms": elapsed / count * 1000.0,
        "frames": count,
        "elapsed_s": elapsed,
    }


def _path_size(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


# ─── Results container ────────────────────────────────────────────────────────

@dataclass
class ComparisonResults:
    base_model: str = ""
    optimized_model: str = ""
    per_image: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    speed: dict = field(default_factory=dict)
    file_size: dict = field(default_factory=dict)
    overall_verdict: str = "UNKNOWN"
    verdict_reasons: list = field(default_factory=list)


# ─── Main comparison ──────────────────────────────────────────────────────────

def run_comparison(
    base_path: str,
    optimized_path: str,
    images: list[Path],
    core_mask: int | None,
    speed_duration: float,
    iou_thresh: float,
) -> ComparisonResults:
    results = ComparisonResults(base_model=base_path, optimized_model=optimized_path)

    section("LOADING MODELS")
    subline("Base (.pt)", base_path)
    subline("Optimized", optimized_path)

    try:
        logger.info("Loading base model (this may take a few seconds): %s", base_path)
        # with _quiet_native():
        base_model = GenericYolo({"file_path": base_path}, core_mask=None)
        logger.info("Loaded base model: %s", getattr(base_model, "model_type", "<unknown>"))
    except Exception as e:
        logger.error("Failed to load base model: %s", e)
        sys.exit(1)

    try:
        logger.info("Loading optimized model (this may take a while on some backends): %s", optimized_path)
        # with _quiet_native():
        optimized_model = GenericYolo({"file_path": optimized_path}, core_mask=core_mask)
        logger.info("Loaded optimized model: %s", getattr(optimized_model, "model_type", "<unknown>"))
    except Exception as e:
        logger.error("Failed to load optimized model: %s", e)
        try:
            base_model.release()
        except Exception:
            pass
        sys.exit(1)

    subline("Base type", base_model.model_type)
    subline("Optimized type", optimized_model.model_type, "PASS")

    # ── file size ──
    try:
        base_size = Path(base_path).stat().st_size
        opt_size = _path_size(Path(optimized_path))
        results.file_size = {
            "base_mb": round(base_size / (1024 * 1024), 2),
            "optimized_mb": round(opt_size / (1024 * 1024), 2),
            "reduction_pct": round((1 - opt_size / base_size) * 100, 1) if base_size else 0.0,
        }
    except Exception:
        pass

    # ── load test frames once, reuse for both detection + speed ──
    loaded: list[tuple[Path, np.ndarray]] = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            warn(f"Could not read {img_path}")
            continue
        loaded.append((img_path, frame))

    if not loaded:
        logger.error("No readable images among the selected test set.")
        sys.exit(1)

    # Warm up optimized model once so the first real inference delay is visible
    try:
        logger.info("Warming optimized model with one inference (may take several seconds)...")
        with _quiet_native():
            _ = optimized_model.predict(loaded[0][1], orig_shape=loaded[0][1].shape)
        logger.info("Optimized model warmup complete.")
    except Exception as e:
        logger.warning("Optimized model warmup failed: %s", e)

    # ── 1. detection agreement ──
    section("1. DETECTION AGREEMENT  -  optimized vs base (.pt) ground truth")

    per_image = []
    total_base_boxes = 0
    total_opt_boxes = 0
    total_matched = 0
    all_ious: list[float] = []
    all_conf_deltas: list[float] = []
    class_agree = 0
    class_total = 0

    for img_path, frame in loaded:
        base_res = base_model.predict(frame, orig_shape=frame.shape)
        opt_res = optimized_model.predict(frame, orig_shape=frame.shape)

        matches = _match_boxes(base_res.boxes, opt_res.boxes, iou_thresh)
        n_base, n_opt = len(base_res.boxes), len(opt_res.boxes)
        matched = len(matches)

        img_ious = [m[2] for m in matches]
        img_conf_deltas = [abs(m[0].conf - m[1].conf) for m in matches]
        img_class_agree = sum(1 for m in matches if m[0].cls_id == m[1].cls_id)

        per_image.append(
            {
                "image": img_path.name,
                "base_boxes": n_base,
                "optimized_boxes": n_opt,
                "matched": matched,
                "match_rate": (matched / n_base) if n_base else (1.0 if n_opt == 0 else 0.0),
                "mean_iou": float(np.mean(img_ious)) if img_ious else 0.0,
                "mean_conf_delta": float(np.mean(img_conf_deltas)) if img_conf_deltas else 0.0,
            }
        )

        total_base_boxes += n_base
        total_opt_boxes += n_opt
        total_matched += matched
        all_ious.extend(img_ious)
        all_conf_deltas.extend(img_conf_deltas)
        class_agree += img_class_agree
        class_total += matched

    for row in per_image:
        subline(
            row["image"],
            f"base={row['base_boxes']:<3} opt={row['optimized_boxes']:<3} "
            f"matched={row['matched']:<3} iou={row['mean_iou']:.2f}",
        )

    overlap_pct = (
        (total_matched / total_base_boxes * 100) if total_base_boxes else (100.0 if total_opt_boxes == 0 else 0.0)
    )
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_conf_delta = float(np.mean(all_conf_deltas)) if all_conf_deltas else 0.0
    class_agreement_pct = (class_agree / class_total * 100) if class_total else 100.0
    box_count_diff_pct = (
        abs(total_opt_boxes - total_base_boxes) / total_base_boxes * 100 if total_base_boxes else 0.0
    )

    print()
    subline("Images tested", str(len(per_image)))
    subline("Base total boxes", str(total_base_boxes))
    subline("Optimized total boxes", str(total_opt_boxes))
    subline(
        "Box count difference",
        f"{box_count_diff_pct:.1f}%",
        "PASS" if box_count_diff_pct <= 15 else "WARN",
    )
    subline(
        f"Bounding box overlap (IoU>{iou_thresh})",
        f"{overlap_pct:.1f}%",
        "PASS" if overlap_pct >= 85 else ("WARN" if overlap_pct >= 70 else "FAIL"),
    )
    subline("Mean IoU on matched boxes", f"{mean_iou:.3f}", "PASS" if mean_iou >= 0.75 else "WARN")
    subline("Mean confidence delta", f"{mean_conf_delta:.3f}", "PASS" if mean_conf_delta <= 0.1 else "WARN")
    subline(
        "Class agreement (matched boxes)",
        f"{class_agreement_pct:.1f}%",
        "PASS" if class_agreement_pct >= 95 else "WARN",
    )

    results.per_image = per_image
    results.aggregate = {
        "total_base_boxes": total_base_boxes,
        "total_optimized_boxes": total_opt_boxes,
        "total_matched": total_matched,
        "box_overlap_pct": overlap_pct,
        "mean_iou": mean_iou,
        "mean_conf_delta": mean_conf_delta,
        "class_agreement_pct": class_agreement_pct,
        "box_count_diff_pct": box_count_diff_pct,
        "iou_threshold": iou_thresh,
    }

    # ── 2. speed ──
    section("2. SPEED  -  base (.pt) vs optimized")

    frames = [f for _, f in loaded]
    with _quiet_native():
        base_speed = _measure_speed(base_model, frames, speed_duration)
    with _quiet_native():
        opt_speed = _measure_speed(optimized_model, frames, speed_duration)

    subline(
        "Base FPS",
        f"{base_speed['fps']:.1f}  ({base_speed['inference_ms']:.1f} ms/frame)" if base_speed["frames"] else "N/A",
    )
    subline(
        "Optimized FPS",
        f"{opt_speed['fps']:.1f}  ({opt_speed['inference_ms']:.1f} ms/frame)" if opt_speed["frames"] else "N/A",
    )

    speedup = (opt_speed["fps"] / base_speed["fps"]) if base_speed.get("fps") else None
    if speedup is not None:
        subline("Speedup vs base", f"{speedup:.2f}x", "PASS" if speedup >= 1.0 else "WARN")

    results.speed = {"base": base_speed, "optimized": opt_speed, "speedup": speedup}

    # ── 3. model size ──
    if results.file_size:
        section("3. MODEL SIZE")
        subline("Base size", f"{results.file_size['base_mb']} MB")
        subline("Optimized size", f"{results.file_size['optimized_mb']} MB")
        subline("Size reduction", f"{results.file_size['reduction_pct']:.1f}%")

    try:
        base_model.release()
    except Exception:
        pass
    try:
        optimized_model.release()
    except Exception:
        pass

    _verdict(results)
    return results


def _verdict(results: ComparisonResults) -> None:
    section("SUMMARY")

    agg = results.aggregate
    reasons = []
    verdict = "READY"

    overlap = agg.get("box_overlap_pct", 0)
    if overlap < 70:
        verdict = "NOT READY"
        reasons.append(f"Bounding box overlap too low ({overlap:.1f}%)")
    elif overlap < 85:
        verdict = "REVIEW RECOMMENDED"
        reasons.append(f"Bounding box overlap marginal ({overlap:.1f}%)")

    if agg.get("total_matched", 0) > 0 and agg.get("mean_iou", 0) < 0.6:
        verdict = "NOT READY"
        reasons.append(f"Mean IoU on matched boxes is low ({agg.get('mean_iou', 0):.2f})")

    if agg.get("box_count_diff_pct", 0) > 30:
        if verdict == "READY":
            verdict = "REVIEW RECOMMENDED"
        reasons.append(f"Box count differs significantly from base ({agg.get('box_count_diff_pct', 0):.1f}%)")

    speedup = results.speed.get("speedup")
    if speedup is not None and speedup < 1.0:
        reasons.append(
            f"Optimized model is SLOWER than the base .pt ({speedup:.2f}x) - "
            "the conversion may not be worth deploying."
        )

    if not reasons:
        reasons.append("Optimized model closely matches base .pt on all checked metrics.")

    color = {"READY": C.GREEN, "REVIEW RECOMMENDED": C.YELLOW, "NOT READY": C.RED}.get(verdict, C.DIM)
    print(_c(f"  VERDICT: {verdict}", C.BOLD + color))
    for r in reasons:
        print(f"    - {r}")
    print()

    results.overall_verdict = verdict
    results.verdict_reasons = reasons


def _clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _result_to_entry(results: ComparisonResults) -> dict:
    return _clean_for_json(
        {
            "base_model": results.base_model,
            "optimized_model": results.optimized_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "overall_verdict": results.overall_verdict,
            "verdict_reasons": results.verdict_reasons,
            "aggregate": results.aggregate,
            "per_image": results.per_image,
            "speed": results.speed,
            "file_size": results.file_size,
        }
    )


def upsert_json_report(results: ComparisonResults, out_path: Path) -> None:
    """Append/update this comparison in a single running ledger file at
    out_path. Entries are keyed by (base_model, optimized_model) - re-running
    the comparison for the same pair (e.g. re-converting the same .pt to the
    same format) updates that entry in place instead of piling up duplicates.
    """
    entry = _result_to_entry(results)

    ledger: dict = {"models": []}
    if out_path.exists():
        try:
            with open(out_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("models"), list):
                ledger = loaded
            elif isinstance(loaded, list):
                # Back-compat: file was previously a bare list of entries.
                ledger = {"models": loaded}
            else:
                logger.warning("Existing report at %s has an unexpected shape - starting a fresh ledger.", out_path)
        except Exception as e:
            logger.warning("Could not read existing report at %s (%s) - starting a fresh ledger.", out_path, e)

    found = False
    for i, existing in enumerate(ledger["models"]):
        if existing.get("base_model") == entry["base_model"] and existing.get("optimized_model") == entry["optimized_model"]:
            ledger["models"][i] = entry
            found = True
            break
    if not found:
        ledger["models"].append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ledger, f, indent=2)

    action = "Updated existing" if found else "Added new"
    print(f"  {action} entry in report ledger: {out_path}\n")


# Kept as an alias so any existing callers/scripts using the old name keep working.
save_json_report = upsert_json_report


# ─── Programmatic entry point (used by boot.py's convert_model) ─────────────

def compare_models(
    base_path: str,
    optimized_path: str,
    images_dir: str | Path,
    *,
    core_mask: int = 7,
    num_images: int = 5,
    seed: int | None = None,
    iou_thresh: float = 0.5,
    speed_duration: float = 5.0,
    output: str | Path | None = None,
    quiet: bool = False,
) -> ComparisonResults | None:
    """Run the base-vs-optimized comparison without going through argparse or
    exiting the process - meant to be called directly from other code (e.g.
    boot.py right after a conversion). Returns None (and logs why) instead of
    raising/exiting if inputs can't be resolved, so callers can treat a
    skipped comparison as non-fatal.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.warning("Comparison images dir not found (%s) - skipping optimized-model comparison.", images_dir)
        return None

    base_p = Path(base_path)
    opt_p = Path(optimized_path)
    if not base_p.exists() or not opt_p.exists():
        logger.warning("Base (%s) or optimized (%s) model missing - skipping comparison.", base_p, opt_p)
        return None
    if base_p.suffix.lower() == ".pt" and opt_p.suffix.lower() == ".pt":
        logger.info("Both models are .pt - nothing to compare, skipping.")
        return None

    try:
        images = _find_test_images(images_dir, num_images, seed)
    except SystemExit:
        logger.warning("No test images found under %s - skipping comparison.", images_dir)
        return None

    out_path = Path(output) if output else (_PROJECT_ROOT / "Outputs" / "optimized_model_report.json")

    try:
        cm = contextlib.nullcontext() if not quiet else _suppress_stdout()
        with cm:
            results = run_comparison(
                str(base_path),
                str(optimized_path),
                images,
                core_mask=core_mask,
                speed_duration=speed_duration,
                iou_thresh=iou_thresh,
            )
    except SystemExit as e:
        logger.warning(
            "Optimized-model comparison exited early for %s (code=%s) - skipping report generation.",
            opt_p.name,
            e.code,
        )
        return None
    except Exception as e:
        logger.warning(
            "Optimized-model comparison failed for %s: %s",
            opt_p.name,
            e,
        )
        return None

    upsert_json_report(results, out_path)
    return results


@contextlib.contextmanager
def _suppress_stdout():
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare a base .pt model against an optimized/converted model "
        "(RKNN, ONNX, TFLite, TensorRT engine, CoreML, TPU, etc.)"
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "Config" / "config.json"),
        help="Config file with vision_model.source_pt (base) and vision_model.file_path "
        "(optimized). Default: Config/config.json",
    )
    parser.add_argument("--base", default=None, help="Explicit path to base .pt model (overrides config)")
    parser.add_argument("--optimized", default=None, help="Explicit path to optimized model (overrides config)")
    parser.add_argument(
        "--images-dir",
        default=str(_PROJECT_ROOT / "QuantizeDataset" / "valid"),
        help="Directory to pull random test images from (default: QuantizeDataset/valid)",
    )
    parser.add_argument("--num-images", type=int, default=5, help="Number of random images to test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for image selection (for reproducibility)")
    parser.add_argument("--core-mask", type=int, default=7, help="RKNN NPU core mask (ignored for non-RKNN backends)")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold to count two boxes as a match")
    parser.add_argument(
        "--speed-duration", type=float, default=5.0, help="Seconds to run each model for the speed benchmark"
    )
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "Outputs" / "optimized_model_report.json"))
    args = parser.parse_args()

    base_path, optimized_path = _load_config_paths(args.config, args.base, args.optimized)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    images = _find_test_images(images_dir, args.num_images, args.seed)

    print(_c("\n  iSpy Optimized Model Validation", C.BOLD))
    print(_c(f"  {len(images)} random image(s) from {images_dir}\n", C.DIM))
    for img in images:
        try:
            shown = img.relative_to(_PROJECT_ROOT)
        except ValueError:
            shown = img
        print(f"    - {shown}")

    results = run_comparison(
        base_path,
        optimized_path,
        images,
        core_mask=args.core_mask,
        speed_duration=args.speed_duration,
        iou_thresh=args.iou_thresh,
    )

    upsert_json_report(results, Path(args.output))

    return 0 if results.overall_verdict in ("READY", "REVIEW RECOMMENDED") else 1


if __name__ == "__main__":
    sys.exit(main())