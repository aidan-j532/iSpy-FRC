import logging
import math
import os
import platform
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

# Match the web dashboard so the chart feels like part of iSpy.
_BG, _PANEL, _TEXT, _DIM = "#0d1117", "#161b22", "#e6edf3", "#9198a1"
_ACCENT, _OK, _BAD, _WARN = "#2f81f7", "#3fb950", "#f85149", "#d29922"

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


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_ms(value) -> str:
    number = _number(value)
    return f"{number:.1f}" if number is not None else "-"


def _label(r, multi_model: bool) -> str:
    parts = []
    if multi_model:
        parts.append(str(r.get("model", "?")))
    backend = str(r.get("backend", "?"))
    model_format = r.get("format")
    device = r.get("device")
    label = backend
    if model_format and str(model_format).lower() not in backend.lower():
        label += f" ({model_format})"
    if device not in (None, ""):
        label += f" · {device}"
    parts.append(label)
    config = []
    if r.get("batch_size") not in (None, 1):
        config.append(f"batch {r['batch_size']}")
    if r.get("mode"):
        config.append(str(r["mode"]))
    if r.get("streams") not in (None, 1):
        config.append(f"{r['streams']} streams")
    if config:
        parts.append(" · ".join(config))
    return textwrap.fill("\n".join(parts), width=32, break_long_words=True)


def _latency_text(r) -> str:
    values = [r.get("inference_ms")]
    percentiles = r.get("latency_ms")
    values.extend(
        percentiles.get(key) if isinstance(percentiles, dict) else None
        for key in ("p50", "p95", "p99")
    )
    return " / ".join(_display_ms(value) for value in values)


def _stage_text(r) -> str:
    stages = r.get("stage_ms")
    return " / ".join(
        _display_ms(stages.get(stage)) if isinstance(stages, dict) else "-"
        for stage in ("preprocess", "device", "postprocess")
    )


def _detail_text(r) -> str:
    if _is_error(r):
        return "ERROR: " + str(r.get("error") or "no valid result")
    details = [str(r.get("provider") or r.get("device") or "provider unknown")]
    if r.get("detections") is not None:
        details.append(f"{r['detections']} detections")
    if r.get("bottleneck"):
        details.append(f"bottleneck: {r['bottleneck']}")
    if r.get("model_forward_fps"):
        details.append(
            f"vision-only: {r['model_forward_fps']:.1f} FPS "
            f"({r['model_forward_ms']:.2f} ms/frame)"
        )
    elif r.get("model_forward_error"):
        details.append(f"vision-only error: {r['model_forward_error']}")
    if _is_suspect(r):
        details.append("suspect latency")
    return " · ".join(details)


def _style(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_DIM, labelsize=8)
    ax.grid(axis="x", color="#30363d", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def render_report(payload: dict, out_path, title: str = "iSpy benchmark"):
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless boards and Colab have no display
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed - skipping chart (pip install matplotlib)"
        )
        return None

    rows = payload.get("all") or []
    if not rows:
        return None
    multi_model = len({r["model"] for r in rows}) > 1
    sysinfo = payload.get("system") or {}
    labels = [_label(r, multi_model) for r in rows]
    details = [_detail_text(r) for r in rows]
    wrapped_details = [
        textwrap.fill(text, width=38, break_long_words=True) for text in details
    ]
    row_heights = [
        max(1.0, max(label.count("\n") + 1, detail.count("\n") + 1) * 0.62)
        for label, detail in zip(labels, wrapped_details)
    ]
    y_positions = []
    cursor = 0.0
    for row_height in row_heights:
        y_positions.append(cursor + row_height / 2)
        cursor += row_height

    max_fps = max((_number(r.get("fps")) or 0.0 for r in rows), default=0.0)
    x_limit = max(max_fps / 0.39, 1.0)
    figure_height = max(4.8, 2.75 + cursor * 0.48)
    fig, ax = plt.subplots(figsize=(19, figure_height), facecolor=_BG)
    fig.subplots_adjust(left=0.255, right=0.99, top=0.81, bottom=0.12)
    ax.set_facecolor(_PANEL)
    ax.set_xlim(0, x_limit)
    ax.set_ylim(cursor + 0.45, -0.55)
    _style(ax)

    valid_fps = [
        _number(r.get("fps"))
        for r in rows
        if not _is_error(r) and not _is_suspect(r) and _number(r.get("fps")) is not None
    ]
    best_fps = max(valid_fps, default=None)
    colors = []
    for r in rows:
        fps = _number(r.get("fps"))
        if _is_error(r) or fps is None:
            colors.append(_BAD)
        elif _is_suspect(r):
            colors.append("#6e7681")
        elif fps == best_fps:
            colors.append(_OK)
        else:
            colors.append(_ACCENT)

    fps_values = [max(0.0, _number(r.get("fps")) or 0.0) for r in rows]
    bars = ax.barh(y_positions, fps_values, color=colors, height=0.58, zorder=3)
    for bar, r, y, color in zip(bars, rows, y_positions, colors):
        if _is_suspect(r) and not _is_error(r):
            bar.set_hatch("//")
        fps = _number(r.get("fps"))
        if not _is_error(r) and fps is not None:
            ax.text(
                fps + max_fps * 0.012,
                y,
                f"{fps:.1f}",
                va="center",
                ha="left",
                color=_TEXT if color != _OK else _OK,
                fontsize=8,
                fontweight="bold",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=_TEXT, fontsize=8.5, ha="right")
    ax.tick_params(axis="y", length=0, pad=10)
    tick_max = max(max_fps, 1.0)
    ticks = [tick_max * fraction for fraction in (0, 0.25, 0.5, 0.75, 1.0)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:.0f}" for value in ticks], color=_DIM, fontsize=8)
    ax.set_xlabel(
        "iSpy predict throughput (FPS; longer bars are faster)",
        color=_DIM,
        fontsize=9,
    )

    latency_x, stage_x, details_x = 0.43, 0.655, 0.81
    headers = (
        (latency_x, "PREDICT MEAN / P50 / P95 / P99 MS"),
        (stage_x, "PRE / DEVICE / POST MS"),
        (details_x, "PROVIDER / DETECTIONS / RESULT"),
    )
    for x, header in headers:
        ax.text(
            x,
            1.015,
            header,
            transform=ax.transAxes,
            color=_DIM,
            fontsize=7.5,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )

    for r, y, color, detail in zip(rows, y_positions, colors, wrapped_details):
        ax.text(
            latency_x,
            y,
            _latency_text(r),
            transform=ax.get_yaxis_transform(),
            va="center",
            color=_TEXT if not _is_error(r) else _DIM,
            fontsize=7.7,
            family="monospace",
        )
        ax.text(
            stage_x,
            y,
            _stage_text(r),
            transform=ax.get_yaxis_transform(),
            va="center",
            color=_TEXT if not _is_error(r) else _DIM,
            fontsize=7.5,
            family="monospace",
        )
        ax.text(
            details_x,
            y,
            detail,
            transform=ax.get_yaxis_transform(),
            va="center",
            color=_BAD if _is_error(r) else (_WARN if _is_suspect(r) else _TEXT),
            fontsize=7.5,
            linespacing=1.15,
        )

    for y in y_positions:
        ax.axhline(y, color="#30363d", linewidth=0.45, alpha=0.45, zorder=0)

    models = list(dict.fromkeys(str(r.get("model", "?")) for r in rows))
    system_parts = [
        sysinfo.get("gpu"),
        f"{sysinfo['cpu_count']} CPU threads" if sysinfo.get("cpu_count") else None,
        f"torch {sysinfo['torch']}" if sysinfo.get("torch") else None,
        f"ONNX Runtime: {', '.join(sysinfo['onnxruntime_providers'])}"
        if sysinfo.get("onnxruntime_providers")
        else None,
        f"Python {sysinfo['python']}" if sysinfo.get("python") else None,
        payload.get("timestamp"),
    ]
    system_line = "  |  ".join(str(part) for part in system_parts if part)
    fig.suptitle(
        title, x=0.255, y=0.97, ha="left", color=_TEXT, fontsize=17, fontweight="bold"
    )
    fig.text(
        0.255,
        0.925,
        textwrap.fill(f"Models: {', '.join(models)}    |    {system_line}", width=180),
        ha="left",
        va="top",
        color=_DIM,
        fontsize=8,
    )
    fig.text(
        0.255,
        0.045,
        "Green: fastest valid run   Blue: other valid runs   Gray hatched: suspect (<0.5 ms)   Red: failed",
        ha="left",
        color=_DIM,
        fontsize=8,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=_BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path
