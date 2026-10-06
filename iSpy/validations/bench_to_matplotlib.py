import logging
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# match the web dashboard so the chart feels like part of iSpy (this part of the file WAS AI generated)
_BG, _PANEL, _TEXT, _DIM = "#0d1117", "#161b22", "#e6edf3", "#9198a1"
_ACCENT, _OK, _BAD, _WARN = "#2f81f7", "#3fb950", "#f85149", "#d29922"
_STAGE_COLORS = {"preprocess": "#2f81f7", "device": "#3fb950", "postprocess": "#f4a261"}

_SUSPECT_MS = 0.5

def collect_system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        from iSpy.config.AutoOpt import has_tpu_hardware

        if has_tpu_hardware():
            import torch_xla

            info["torch_xla"] = getattr(torch_xla, "__version__", "present")
    except Exception:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        pass
    return info


def _is_error(r) -> bool:
    return bool(r.get("error")) or not r.get("fps") or r.get("ok") is False


def _is_suspect(r) -> bool:
    ms = r.get("inference_ms")
    return ms is not None and ms < _SUSPECT_MS


def _label(r, multi_model: bool) -> str:
    parts = [str(r.get("backend", "?"))]
    if r.get("batch_size") not in (None, 1):
        parts.append(f"b{r['batch_size']}")
    if r.get("mode"):
        parts.append(str(r["mode"]))
    if r.get("streams") not in (None, 1):
        parts.append(f"x{r['streams']} streams")
    label = " ".join(parts)
    return f"{r['model']}\n{label}" if multi_model else label


def _style(ax, title):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(colors=_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(axis="x", color="#30363d", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def _panel_throughput(ax, rows, multi_model):
    good = sorted((r for r in rows if not _is_error(r)), key=lambda r: r["fps"])
    bad = [r for r in rows if _is_error(r)]
    ordered = bad + good  # best ends up on top
    labels = [_label(r, multi_model) for r in ordered]
    vals = [0 if _is_error(r) else r["fps"] for r in ordered]

    real = [r for r in good if not _is_suspect(r)]
    best = max(real, key=lambda r: r["fps"]) if real else None
    # baseline for the speedup annotation: a CPU run if there is one, else the slowest valid run
    cpu = [r for r in real if "cpu" in str(r.get("backend", "")).lower()]
    base = min(cpu or real, key=lambda r: r["fps"]) if real else None

    colors = []
    for r in ordered:
        if _is_error(r):
            colors.append(_BAD)
        elif _is_suspect(r):
            colors.append("#6e7681")
        elif r is best:
            colors.append(_OK)
        else:
            colors.append(_ACCENT)

    bars = ax.barh(range(len(ordered)), vals, color=colors, height=0.62)
    for bar, r in zip(bars, ordered):
        if _is_suspect(r) and not _is_error(r):
            bar.set_hatch("//")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, color=_TEXT, fontsize=9)

    top = max(vals) if vals and max(vals) > 0 else 1
    for i, r in enumerate(ordered):
        if _is_error(r):
            msg = str(r.get("error") or "no result")[:70]
            ax.text(top * 0.01, i, f"ERROR: {msg}", va="center", color=_BAD, fontsize=8)
            continue
        txt = f"{r['fps']:.1f} FPS  |  {r.get('inference_ms', 0):.1f} ms"
        if _is_suspect(r):
            txt += "  SUSPECT (model likely not running)"
        elif base is not None and r is not base:
            txt += f"  |  {r['fps'] / base['fps']:.1f}x vs {base['backend']}"
        ax.text(r["fps"] + top * 0.01, i, txt, va="center", color=_TEXT, fontsize=8)
    ax.set_xlim(0, top * 1.55)
    ax.set_xlabel("frames per second (higher is better)", color=_DIM, fontsize=8)
    _style(ax, "Throughput")


def _panel_latency(ax, rows, multi_model):
    rows = [r for r in rows if not _is_error(r) and not _is_suspect(r)]
    with_pct = [r for r in rows if isinstance(r.get("latency_ms"), dict)]
    if with_pct:
        rows = sorted(with_pct, key=lambda r: r["latency_ms"].get("p50", 0), reverse=True)
        h = 0.26
        for j, (key, color) in enumerate((("p50", _OK), ("p95", _WARN), ("p99", _BAD))):
            ax.barh([i + (1 - j) * h for i in range(len(rows))],
                    [r["latency_ms"].get(key, 0) for r in rows],
                    height=h, color=color, label=key)
        ax.legend(facecolor=_PANEL, edgecolor="#30363d", labelcolor=_TEXT, fontsize=8)
        ax.set_xlabel("per-frame latency, ms (lower is better)", color=_DIM, fontsize=8)
    else:
        rows = sorted(rows, key=lambda r: r["inference_ms"], reverse=True)
        ax.barh(range(len(rows)), [r["inference_ms"] for r in rows], color=_ACCENT, height=0.6)
        ax.set_xlabel("mean ms per frame (lower is better)", color=_DIM, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r, multi_model) for r in rows], color=_TEXT, fontsize=8)
    _style(ax, "Latency")


def _panel_stages(ax, rows, multi_model):
    rows = [r for r in rows if isinstance(r.get("stage_ms"), dict) and not _is_error(r)]
    left = [0.0] * len(rows)
    for stage, color in _STAGE_COLORS.items():
        vals = [r["stage_ms"].get(stage, 0.0) for r in rows]
        ax.barh(range(len(rows)), vals, left=left, color=color, label=stage, height=0.6)
        left = [a + b for a, b in zip(left, vals)]
    for i, r in enumerate(rows):
        slowest = max(r["stage_ms"], key=r["stage_ms"].get)
        ax.text(left[i], i, f"  bottleneck: {slowest}", va="center", color=_TEXT, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r, multi_model) for r in rows], color=_TEXT, fontsize=8)
    ax.legend(facecolor=_PANEL, edgecolor="#30363d", labelcolor=_TEXT, fontsize=8)
    ax.set_xlabel("ms per frame, by stage", color=_DIM, fontsize=8)
    _style(ax, "Where the time goes")


def _panel_batch(ax, rows):
    groups = defaultdict(list)
    for r in rows:
        if not _is_error(r) and not _is_suspect(r) and r.get("batch_size"):
            groups[r["backend"]].append((r["batch_size"], r["fps"]))
    for backend, pts in groups.items():
        pts.sort()
        ax.plot(*zip(*pts), marker="o", label=backend)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size", color=_DIM, fontsize=8)
    ax.set_ylabel("FPS", color=_DIM, fontsize=8)
    ax.legend(facecolor=_PANEL, edgecolor="#30363d", labelcolor=_TEXT, fontsize=8)
    _style(ax, "Batch scaling")
    ax.grid(axis="y", color="#30363d", linewidth=0.5, alpha=0.6)


def render_report(payload: dict, out_path, title: str = "iSpy benchmark"):
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless boards and Colab have no display
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart (pip install matplotlib)")
        return None

    rows = payload.get("all") or []
    if not rows:
        return None
    multi_model = len({r["model"] for r in rows}) > 1

    lower = ["latency"]
    if any(isinstance(r.get("stage_ms"), dict) for r in rows):
        lower.append("stages")
    if len({r.get("batch_size") for r in rows if r.get("batch_size")}) > 1:
        lower.append("batch")

    height = 2.2 + 0.42 * len(rows) + 3.6
    fig = plt.figure(figsize=(15, max(height, 8)), facecolor=_BG)
    gs = GridSpec(3, len(lower), figure=fig, height_ratios=[0.5, 0.42 * len(rows) + 1.5, 3.4],
                  hspace=0.45, wspace=0.35, left=0.12, right=0.97, top=0.96, bottom=0.06)

    # header: what was tested, on what
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    sysinfo = payload.get("system") or {}
    best = (payload.get("best") or {})
    ax_head.text(0, 0.85, title, color=_TEXT, fontsize=18, fontweight="bold", va="top")
    line = " | ".join(
        str(x) for x in (
            sysinfo.get("gpu") or ("TPU (torch_xla " + sysinfo["torch_xla"] + ")" if "torch_xla" in sysinfo else None),
            f"{sysinfo.get('cpu_count')} CPU threads" if sysinfo.get("cpu_count") else None,
            f"torch {sysinfo['torch']}" if "torch" in sysinfo else None,
            f"python {sysinfo.get('python')}",
            payload.get("timestamp"),
        ) if x
    )
    ax_head.text(0, 0.25, line, color=_DIM, fontsize=9, va="top")

    _panel_throughput(fig.add_subplot(gs[1, :]), rows, multi_model)
    for col, kind in enumerate(lower):
        ax = fig.add_subplot(gs[2, col])
        {"latency": lambda: _panel_latency(ax, rows, multi_model),
         "stages": lambda: _panel_stages(ax, rows, multi_model),
         "batch": lambda: _panel_batch(ax, rows)}[kind]()

    fig.text(0.97, 0.01,
             "hatched gray = suspect (<0.5 ms, model probably not running) | red = failed to run",
             color=_DIM, fontsize=7, ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return out_path