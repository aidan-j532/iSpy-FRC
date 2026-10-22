import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PROJECT_ROOT = Path.cwd()


def _bench_source_image() -> Path:
    import cv2
    import numpy as np

    out = _PROJECT_ROOT / "Outputs" / "bench_source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (35, 35, 35)
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 80, (480, 640, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.4, noise, 0.6, 0)
        cv2.circle(img, (320, 240), 110, (60, 120, 220), -1)
        cv2.rectangle(img, (80, 70), (250, 260), (60, 200, 90), -1)
        cv2.putText(
            img, "bench", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2
        )
        cv2.imwrite(str(out), img)
    return out


def _load_bench_config(config_path: Path, bench_image: Path):
    from iSpy.config.iSpyConfig import ensure_camera_entries_ready, iSpyConfig

    config = iSpyConfig(str(config_path), create=True)
    cameras = config.config.get("camera_configs") or {}
    if not cameras:
        raise SystemExit("No camera_configs in the config - run 'ispy setup' first.")

    # exactly one camera on the still: one pipeline, no live-scene flakiness,
    # and solo mode (what boards actually run)
    name, camera = next(iter(cameras.items()))
    camera = copy.deepcopy(camera if isinstance(camera, dict) else {})
    camera["source"] = str(bench_image)
    ensure_camera_entries_ready({name: camera})

    # use config.set() not direct dict writes, the camera wrappers are a cached
    # view and set() knows to rebuild them
    config.set("camera_configs", {name: camera})
    # keep it pure, no fps cap, no web server, bench recordings aside
    config.set("app_mode", False)
    config.set("max_fps", 0)
    rollback = dict(config.get("rollback") or {})
    rollback["data_dir"] = "Outputs/bench_recordings"
    config.set("rollback", rollback)
    return config


def _run_level(level: int, seconds: float, warmup_s: float, config_path: Path) -> dict:
    from iSpy.core import performance
    from iSpy.iSpy import iSpy
    from iSpy.web.Backend.WebApp import create_app

    profile = performance.PROFILES[level]
    performance.set_level_for_run(level)

    config = _load_bench_config(config_path, _bench_source_image())
    web = None
    if profile.web_enabled:
        # a real iSpyWebApp (all 14 modules, real scheduler) without the Flask
        # serving thread - update() is what costs per tick
        web = create_app(cameras=[], config=config)
    vision = iSpy(config, web_app=web)
    if web is not None and hasattr(web, "set_vision_instance"):
        web.set_vision_instance(vision)

    camera = vision.cameras[0]
    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        frame_data = vision._run_loop_body_solo(camera)
        vision._update_utilities(frame_data)
        vision._update_web(frame_data)

    runs = []
    parts: dict[str, list] = {}
    ticks = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        frame_data = vision._run_loop_body_solo(camera)
        vision._update_utilities(frame_data)
        t_web = time.perf_counter()
        vision._update_web(frame_data)
        frame_data["code_times"]["web"] = time.perf_counter() - t_web
        loop_s = time.perf_counter() - t0
        runs.append(loop_s)
        ticks += 1
        for key, val in (frame_data.get("code_times") or {}).items():
            parts.setdefault(key, []).append(val)

    try:
        vision._stop_all_plugins()
    except Exception:
        pass
    try:
        camera.destroy()
    except Exception:
        pass
    # no web.stop() on purpose, MetricsModule saves a file on stop and the bench
    # would fill Outputs with metrics_*.json snapshots
    shutil.rmtree(
        str(_PROJECT_ROOT / "Outputs" / "bench_recordings"), ignore_errors=True
    )

    runs.sort()
    mean = sum(runs) / len(runs) if runs else 0.0
    p95 = runs[int(len(runs) * 0.95)] if runs else 0.0
    part_means = {
        key: (sum(vals) / len(vals)) * 1000.0 for key, vals in parts.items() if vals
    }
    return {
        "level": level,
        "label": profile.label,
        "fps": (1.0 / mean) if mean else 0.0,
        "mean_ms": mean * 1000.0,
        "p95_ms": p95 * 1000.0,
        "ticks": ticks,
        # loop time the per-part timers miss. plot()/debug-frame drawing and the
        # raw-frame copies are the two big ones
        "other_ms": (mean * 1000.0) - sum(part_means.values()),
        "parts": part_means,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Bench the performance profiles with the real loop + web fan-out"
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="seconds per level")
    parser.add_argument(
        "--warmup", type=float, default=2.0, help="warm-up seconds before measuring"
    )
    parser.add_argument(
        "--levels",
        default="4,3,2,1,0",
        help="comma-separated levels to run (default: 4,3,2,1,0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "run the level list this many times, interleaved (4,3,2,1,0,4,3,...) "
            "so CPU thermal drift does not land on one level, and report medians"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "Config" / "config.json"),
        help="config to borrow the camera + pipeline from",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"No config at {config_path} - run 'ispy setup' first.")

    levels = []
    for chunk in args.levels.split(","):
        chunk = chunk.strip()
        if chunk:
            levels.append(int(chunk))

    print(f"bench source: {_bench_source_image()}")
    repeats = max(1, int(args.repeat))
    raw_rows = []
    for rep in range(repeats):
        # rotate the order every pass so each level sits in each position once -
        # cancels the "the machine only gets hotter" bias
        rot = rep % len(levels)
        order = levels[rot:] + levels[:rot]
        print(f"-- pass {rep + 1}/{repeats}: {order} --")
        for level in order:
            row = _run_level(level, args.seconds, args.warmup, config_path)
            raw_rows.append(row)
            print(
                f"   level {row['level']}: {row['fps']:.1f} fps "
                f"({row['mean_ms']:.1f} ms avg, {row['ticks']} ticks)"
            )

    def _median(values):
        values = sorted(values)
        return values[len(values) // 2] if values else 0.0

    results = []
    for level in levels:
        rows = [row for row in raw_rows if row["level"] == level]
        parts: dict[str, list] = {}
        for row in rows:
            for key, val in row["parts"].items():
                parts.setdefault(key, []).append(val)
        results.append(
            {
                "level": level,
                "label": rows[0]["label"],
                "fps": _median([r["fps"] for r in rows]),
                "mean_ms": _median([r["mean_ms"] for r in rows]),
                "p95_ms": _median([r["p95_ms"] for r in rows]),
                "other_ms": _median([r["other_ms"] for r in rows]),
                "ticks": sum(r["ticks"] for r in rows),
                "parts": {key: _median(vals) for key, vals in parts.items()},
            }
        )

    print(f"{'level':>5}  {'profile':<14} {'fps':>7}  {'loop ms':>8} {'p95 ms':>8}")
    for row in results:
        print(
            f"{row['level']:>5}  {row['label']:<14} {row['fps']:>7.1f}  "
            f"{row['mean_ms']:>8.2f} {row['p95_ms']:>8.2f}"
        )

    print()
    for row in results:
        ranked = sorted(row["parts"].items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{key} {val:.2f}ms" for key, val in ranked[:6])
        per_pass = [r for r in raw_rows if r["level"] == row["level"]]
        fps_list = ", ".join(f"{r['fps']:.1f}" for r in per_pass)
        print(
            f"level {row['level']} ({row['label']}, {row['ticks']} ticks): "
            f"{top}, ~other {row['other_ms']:.2f}ms"
        )
        print(f"   fps per pass: {fps_list}")

    if len(results) >= 2 and results[0]["level"] == 4 and results[-1]["level"] == 0:
        gain = results[-1]["fps"] - results[0]["fps"]
        print()
        print(
            f"Max FPS vs Full Features: {results[-1]['fps']:.1f} fps vs "
            f"{results[0]['fps']:.1f} fps (+{gain:.1f} fps)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
