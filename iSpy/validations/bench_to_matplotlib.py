"""Render the ispy-bench JSON payload as a clean, data-dense PNG report."""

import logging
import math
import os
import platform
import socket
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

_SUSPECT_MS = 0.5

PLOT_SECTIONS = (
    "summary",
    "throughput",
    "latency",
    "stages",
    "table",
    "system",
)

_THEMES = {
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "text": "#1f2328",
        "dim": "#59636e",
        "muted": "#8b949e",
        "rule": "#d0d7de",
        "grid": "#eaeef2",
        "accent": "#245c9c",
        "accent_dark": "#17406e",
        "accent_soft": "#a8bfd8",
        "good": "#2d6a4f",
        "bad": "#a33a35",
        "warn": "#8a6116",
        "alt": "#f6f8fa",
        "stage_pre": "#a8bfd8",
        "stage_dev": "#245c9c",
        "stage_post": "#d7dee6",
    },
    "dark": {
        "bg": "#111417",
        "panel": "#111417",
        "text": "#e6edf3",
        "dim": "#9da7b3",
        "muted": "#6e7781",
        "rule": "#30363d",
        "grid": "#22282e",
        "accent": "#7aa7d9",
        "accent_dark": "#a8c7e8",
        "accent_soft": "#3b5878",
        "good": "#7bc496",
        "bad": "#f08b85",
        "warn": "#d9b25f",
        "alt": "#171c21",
        "stage_pre": "#3b5878",
        "stage_dev": "#7aa7d9",
        "stage_post": "#39414a",
    },
}


def _cpu_name() -> str | None:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    name = platform.processor()
    return name.strip() if name and name.strip() else None


def collect_system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
    }
    cpu = _cpu_name()
    if cpu:
        info["cpu"] = cpu
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["memory_gb"] = round(
            psutil.virtual_memory().total / (1024**3),
            1,
        )
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
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

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        pass
    try:
        from importlib.metadata import version

        info["ispy"] = version("ispy-frc")
    except Exception:
        pass
    return info


def _is_error(r) -> bool:
    return bool(r.get("error")) or not r.get("fps") or r.get("ok") is False


def _is_suspect(r) -> bool:
    ms = _number(r.get("inference_ms"))
    return ms is not None and ms < _SUSPECT_MS


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "-"
    if abs(number) >= 1000:
        return f"{number:,.0f}{suffix}"
    return f"{number:.{digits}f}{suffix}"


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _truncate(value, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + "…"


def _readable_on(color: str) -> str:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return "#ffffff"
    try:
        red, green, blue = (
            int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)
        )
    except ValueError:
        return "#ffffff"
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111417" if luminance > 0.55 else "#ffffff"


def _row_label(r, multi_model: bool) -> str:
    parts = []
    if multi_model:
        parts.append(str(r.get("model") or "?"))
    backend = str(r.get("backend") or "?")
    extras = []
    model_format = r.get("format")
    device = r.get("device")
    if model_format and str(model_format).lower() not in backend.lower():
        extras.append(str(model_format))
    if device not in (None, "") and str(device).lower() not in backend.lower():
        extras.append(str(device))
    if extras:
        backend += " · " + " · ".join(extras)
    parts.append(backend)

    config = []
    if r.get("batch_size") not in (None, 1):
        config.append(f"b{r['batch_size']}")
    if r.get("mode") and str(r["mode"]) != "serial":
        config.append(str(r["mode"]))
    if r.get("streams") not in (None, 1):
        config.append(f"x{r['streams']}")
    if r.get("tpu_cores"):
        config.append(f"{r['tpu_cores']} cores")
    if config:
        parts.append(" · ".join(config))
    if _is_error(r):
        parts.append("failed")
    return "\n".join(
        textwrap.fill(part, width=27, break_long_words=True) for part in parts
    )


def _sort_rows(rows: list[dict], sort: str) -> list[dict]:
    if sort == "fps":
        return sorted(
            rows,
            key=lambda r: (
                1 if _is_error(r) else 0,
                -(_number(r.get("fps")) or 0.0),
            ),
        )
    if sort == "name":
        return sorted(
            rows,
            key=lambda r: (
                str(r.get("model") or ""),
                str(r.get("backend") or ""),
                r.get("batch_size") or 1,
                str(r.get("mode") or ""),
                r.get("streams") or 1,
            ),
        )
    return list(rows)


def _stage_values(r) -> tuple[float, float, float]:
    stage = r.get("stage_ms")
    if not isinstance(stage, dict):
        return (0.0, 0.0, 0.0)
    return tuple(
        _number(stage.get(name)) or 0.0
        for name in ("preprocess", "device", "postprocess")
    )


def _stage_total(r) -> float:
    return sum(_stage_values(r))


def _subtitle(payload: dict, rows: list[dict]) -> str:
    settings = payload.get("settings") or {}
    models = list(dict.fromkeys(str(r.get("model") or "?") for r in rows))
    bits = []
    if payload.get("timestamp"):
        bits.append(str(payload["timestamp"]))
    bits.append(f"{len(models)} model{'s' if len(models) != 1 else ''}")
    bits.append(f"{len(rows)} run{'s' if len(rows) != 1 else ''}")
    duration = settings.get("duration")
    if duration:
        bits.append(f"{float(duration):g} s/run")
    batch_sizes = settings.get("batch_sizes")
    if batch_sizes:
        bits.append("batch " + ",".join(str(v) for v in batch_sizes))
    streams = settings.get("streams")
    if streams and streams != 1:
        bits.append(f"{streams} streams")
    modes = settings.get("modes")
    if modes:
        if isinstance(modes, str):
            modes = [modes]
        bits.append(" + ".join(str(m) for m in modes))
    backends = settings.get("backends")
    if backends:
        bits.append(f"{len(backends)} backend{'s' if len(backends) != 1 else ''}")
    return "  ·  ".join(bits) + "  ·  " + ", ".join(models)


def _style_axis(ax, theme: dict) -> None:
    ax.set_facecolor(theme["panel"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["rule"])
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", colors=theme["dim"], labelsize=7, length=0)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.7)
    ax.set_axisbelow(True)


def _draw_header(ax, title: str, subtitle: str, theme: dict) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.72,
        title,
        ha="left",
        va="center",
        color=theme["text"],
        fontsize=20,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.2,
        textwrap.fill(subtitle, width=190),
        ha="left",
        va="center",
        color=theme["dim"],
        fontsize=8.4,
        transform=ax.transAxes,
    )
    ax.plot(
        [0.0, 1.0],
        [0.015, 0.015],
        transform=ax.transAxes,
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )


def _draw_summary(ax, rows: list[dict], theme: dict) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    valid = [r for r in rows if not _is_error(r)]
    fps_values = [
        _number(r.get("fps"))
        for r in valid
        if not _is_suspect(r) and _number(r.get("fps")) is not None
    ]
    if not fps_values:
        fps_values = [
            _number(r.get("fps")) for r in valid if _number(r.get("fps")) is not None
        ]
    best = max(valid, key=lambda r: _number(r.get("fps")) or -1, default=None)
    best_fps = _number(best.get("fps")) if best else None
    median_fps = sorted(fps_values)[len(fps_values) // 2] if fps_values else None
    fastest = min(
        (r for r in valid if _number(r.get("inference_ms")) is not None),
        key=lambda r: _number(r.get("inference_ms")) or math.inf,
        default=None,
    )
    fastest_ms = _number(fastest.get("inference_ms")) if fastest else None
    frames = sum(int(r.get("frames") or 0) for r in rows)
    detections = sum(int(r.get("detections") or 0) for r in rows)
    failures = len(rows) - len(valid)
    elapsed = sum(_number(r.get("elapsed")) or 0.0 for r in rows)

    cells = [
        (
            "BEST FPS",
            _fmt(best_fps),
            _truncate(best.get("backend") or "-", 26) if best else "-",
        ),
        ("MEDIAN FPS", _fmt(median_fps), f"{len(valid)} valid runs"),
        (
            "FASTEST LATENCY",
            _fmt(fastest_ms, 2, " ms"),
            _truncate(fastest.get("backend") or "-", 26) if fastest else "-",
        ),
        (
            "RUNS OK",
            f"{len(valid)}/{len(rows)}",
            f"{failures} failed" if failures else "all passed",
        ),
        ("FRAMES", _fmt_int(frames), f"{elapsed:.1f} s measured"),
        (
            "DETECTIONS",
            _fmt_int(detections),
            f"{detections / elapsed:.1f}/s" if elapsed > 0 else "-",
        ),
    ]

    ax.plot(
        [0.0, 1.0],
        [0.96, 0.96],
        transform=ax.transAxes,
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )
    ax.plot(
        [0.0, 1.0],
        [0.04, 0.04],
        transform=ax.transAxes,
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )
    count = len(cells)
    for index, (label, value, detail) in enumerate(cells):
        x = (index + 0.5) / count
        ax.text(
            x,
            0.78,
            label,
            ha="center",
            va="center",
            color=theme["dim"],
            fontsize=6.8,
            fontweight="bold",
        )
        ax.text(
            x,
            0.43,
            value,
            ha="center",
            va="center",
            color=theme["text"],
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            x,
            0.12,
            detail,
            ha="center",
            va="center",
            color=theme["muted"],
            fontsize=7,
        )
        if index:
            ax.plot(
                [index / count, index / count],
                [0.12, 0.88],
                transform=ax.transAxes,
                color=theme["rule"],
                linewidth=0.6,
            )


def _draw_throughput(ax, rows, labels, theme: dict) -> None:
    _style_axis(ax, theme)
    valid_fps = [
        _number(r.get("fps"))
        for r in rows
        if not _is_error(r) and not _is_suspect(r) and _number(r.get("fps"))
    ]
    best_fps = max(valid_fps, default=None)
    max_fps = max((_number(r.get("fps")) or 0.0 for r in rows), default=0.0)
    y_positions = list(range(len(rows)))
    colors = []
    for r in rows:
        fps = _number(r.get("fps"))
        if _is_error(r) or fps is None:
            colors.append(theme["bad"])
        elif _is_suspect(r):
            colors.append(theme["muted"])
        elif best_fps is not None and fps == best_fps:
            colors.append(theme["good"])
        else:
            colors.append(theme["accent"])

    values = [max(0.0, _number(r.get("fps")) or 0.0) for r in rows]
    ax.barh(
        y_positions,
        values,
        color=colors,
        height=0.58,
        linewidth=0,
        zorder=3,
    )
    for y, r, fps, color in zip(y_positions, rows, values, colors):
        if _is_error(r):
            ax.text(
                0.015,
                y,
                "failed",
                va="center",
                ha="left",
                color=theme["bad"],
                fontsize=7,
                transform=ax.get_yaxis_transform(),
            )
        else:
            ax.text(
                fps + max(max_fps, 1.0) * 0.012,
                y,
                _fmt(fps),
                va="center",
                ha="left",
                color=theme["good"] if color == theme["good"] else theme["text"],
                fontsize=7.2,
                fontweight="bold",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=theme["text"], fontsize=7.2, ha="right")
    ax.tick_params(axis="y", pad=8)
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.05)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max_fps * 1.2, 1.0))
    ax.set_title(
        "Throughput  ·  predict FPS, higher is better",
        loc="left",
        color=theme["text"],
        fontsize=9.2,
        fontweight="bold",
        pad=6,
    )


def _draw_latency(
    ax,
    rows,
    labels,
    theme: dict,
    scale: str,
    show_labels: bool = True,
) -> None:
    _style_axis(ax, theme)
    y_positions = list(range(len(rows)))
    points = []
    for r in rows:
        latency = r.get("latency_ms")
        latency = latency if isinstance(latency, dict) else {}
        p50 = _number(latency.get("p50"))
        p95 = _number(latency.get("p95"))
        p99 = _number(latency.get("p99"))
        mean = _number(latency.get("mean")) or _number(r.get("inference_ms"))
        values = [v for v in (p50, p95, p99, mean) if v is not None and v > 0]
        points.append((r, p50, p95, p99, mean, values))

    drawable = [
        (y, point)
        for y, point in zip(y_positions, points)
        if not _is_error(point[0]) and point[5]
    ]
    if not drawable:
        ax.text(
            0.5,
            0.5,
            "no latency samples",
            ha="center",
            va="center",
            color=theme["dim"],
            fontsize=8,
            transform=ax.transAxes,
        )
    for y, (r, p50, p95, p99, mean, values) in drawable:
        if p50 is not None and p99 is not None:
            ax.plot(
                [p50, p99],
                [y, y],
                color=theme["accent_soft"],
                linewidth=2.2,
                solid_capstyle="round",
                zorder=2,
            )
        if p95 is not None:
            ax.plot(
                [p95, p95],
                [y - 0.13, y + 0.13],
                color=theme["accent_dark"],
                linewidth=1.1,
                zorder=3,
            )
        if mean is not None:
            ax.scatter(
                [mean],
                [y],
                s=20,
                color=theme["accent"],
                edgecolors=theme["panel"],
                linewidths=0.6,
                zorder=4,
            )
        max_value = max(values)
        label_x = (
            max_value * 1.06
            if scale == "log"
            else max_value + max(max_value, 1.0) * 0.015
        )
        ax.text(
            label_x,
            y,
            _fmt(max_value, 1),
            va="center",
            ha="left",
            color=theme["muted"],
            fontsize=6.8,
        )

    ax.set_yticks(y_positions)
    if show_labels:
        ax.set_yticklabels(labels, color=theme["text"], fontsize=7.2, ha="right")
        ax.tick_params(axis="y", pad=8)
        for tick in ax.get_yticklabels():
            tick.set_linespacing(1.05)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    all_values = [value for _y, point in drawable for value in point[5]]
    if scale == "log":
        ax.set_xscale("log")
        try:
            from matplotlib.ticker import NullFormatter, ScalarFormatter

            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        except Exception:
            pass
        if all_values:
            ax.set_xlim(
                min(all_values) * 0.75,
                max(all_values) * 1.35,
            )
    else:
        max_value = max(all_values, default=1.0)
        ax.set_xlim(0, max(max_value * 1.25, 1.0))
    ax.set_title(
        "Latency  ·  milliseconds, lower is better",
        loc="left",
        color=theme["text"],
        fontsize=9.2,
        fontweight="bold",
        pad=6,
    )


def _draw_stages(ax, rows, labels, theme: dict) -> None:
    _style_axis(ax, theme)
    y_positions = list(range(len(rows)))
    for y, r in zip(y_positions, rows):
        pre, device, post = _stage_values(r)
        total = pre + device + post
        if _is_error(r) or total <= 0:
            ax.text(
                0.01,
                y,
                "failed" if _is_error(r) else "no stage data",
                va="center",
                ha="left",
                color=theme["bad"] if _is_error(r) else theme["dim"],
                fontsize=7,
                transform=ax.get_yaxis_transform(),
            )
            continue
        left = 0.0
        for value, color in (
            (pre, theme["stage_pre"]),
            (device, theme["stage_dev"]),
            (post, theme["stage_post"]),
        ):
            share = value / total * 100
            ax.barh(
                [y],
                [share],
                left=[left],
                height=0.58,
                color=color,
                linewidth=0,
                zorder=3,
            )
            if share >= 12:
                ax.text(
                    left + share / 2,
                    y,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    color=_readable_on(color),
                    fontsize=6.6,
                )
            left += share
        ax.text(
            101,
            y,
            _fmt(total, 1, " ms"),
            va="center",
            ha="left",
            color=theme["muted"],
            fontsize=6.8,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=theme["text"], fontsize=7.2, ha="right")
    ax.tick_params(axis="y", pad=8)
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.05)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.text(
        0.0,
        1.015,
        "Stage breakdown  ·  share of measured frame time",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=theme["text"],
        fontsize=9.2,
        fontweight="bold",
    )
    from matplotlib.patches import Rectangle

    for index, (label, color) in enumerate(
        (
            ("preprocess", theme["stage_pre"]),
            ("device", theme["stage_dev"]),
            ("postprocess", theme["stage_post"]),
        )
    ):
        x = 0.42 + index * 0.19
        ax.add_patch(
            Rectangle(
                (x, 1.001),
                0.012,
                0.012,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            x + 0.018,
            1.007,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=theme["dim"],
            fontsize=7,
        )


def _table_columns() -> list[tuple[str, float, str]]:
    return [
        ("#", 0.028, "left"),
        ("MODEL", 0.078, "left"),
        ("BACKEND", 0.09, "left"),
        ("FORMAT", 0.046, "left"),
        ("DEVICE", 0.058, "left"),
        ("CONFIG", 0.075, "left"),
        ("PROVIDER", 0.09, "left"),
        ("FPS", 0.052, "right"),
        ("MS", 0.052, "right"),
        ("P50", 0.052, "right"),
        ("P95", 0.052, "right"),
        ("P99", 0.052, "right"),
        ("PRE", 0.046, "right"),
        ("DEV", 0.046, "right"),
        ("POST", 0.046, "right"),
        ("FRAMES", 0.054, "right"),
        ("DET", 0.048, "right"),
        ("VISION", 0.054, "right"),
    ]


def _draw_table(ax, rows, theme: dict, sort: str) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    count = len(rows)
    ax.set_ylim(-0.5, count + 0.9)
    columns = _table_columns()
    total_weight = sum(weight for _label, weight, _align in columns)
    widths = [weight / total_weight for _label, weight, _align in columns]
    starts = []
    cursor = 0.0
    for width in widths:
        starts.append(cursor)
        cursor += width

    header_y = count + 0.25
    for index, (label, _weight, align) in enumerate(columns):
        x = starts[index] + 0.004
        if align == "right":
            x = starts[index] + widths[index] - 0.004
        ax.text(
            x,
            header_y,
            label,
            ha=align,
            va="center",
            color=theme["dim"],
            fontsize=6.4,
            fontweight="bold",
        )
    ax.plot(
        [0, 1],
        [count, count],
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )

    for row_index, r in enumerate(rows):
        y = count - 0.5 - row_index
        if row_index % 2:
            ax.axhspan(y - 0.5, y + 0.5, color=theme["alt"], linewidth=0)
        latency = r.get("latency_ms")
        latency = latency if isinstance(latency, dict) else {}
        pre, device, post = _stage_values(r)
        config = (
            f"{r.get('mode') or 'serial'}/b{r.get('batch_size') or 1}"
            f"/x{r.get('streams') or 1}"
        )
        if r.get("tpu_cores"):
            config += f"/{r['tpu_cores']}c"
        values = [
            _fmt_int(row_index + 1),
            _truncate(r.get("model") or "?", 14),
            _truncate(r.get("backend") or "?", 18),
            _truncate(r.get("format") or "-", 8),
            _truncate(r.get("device") or "-", 10),
            _truncate(config, 18),
            _truncate(r.get("provider") or "-", 18),
            "failed" if _is_error(r) else _fmt(r.get("fps")),
            _fmt(r.get("inference_ms"), 2),
            _fmt(latency.get("p50"), 2),
            _fmt(latency.get("p95"), 2),
            _fmt(latency.get("p99"), 2),
            _fmt(pre, 2) if pre or device or post else "-",
            _fmt(device, 2) if pre or device or post else "-",
            _fmt(post, 2) if pre or device or post else "-",
            _fmt_int(r.get("frames")),
            _fmt_int(r.get("detections")),
            _fmt(r.get("model_forward_fps")),
        ]
        for col_index, ((_label, _weight, align), value) in enumerate(
            zip(columns, values)
        ):
            x = starts[col_index] + 0.004
            if align == "right":
                x = starts[col_index] + widths[col_index] - 0.004
            color = theme["text"]
            if col_index == 7 and _is_error(r):
                color = theme["bad"]
            elif _is_suspect(r) and col_index in (7, 8):
                color = theme["warn"]
            elif col_index in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
                color = theme["dim"]
            ax.text(
                x,
                y,
                value,
                ha=align,
                va="center",
                color=color,
                fontsize=6.5,
                family="monospace" if col_index >= 7 else "sans-serif",
            )

    ax.plot(
        [0, 1],
        [0, 0],
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )
    ax.text(
        0.0,
        1.012,
        "Run detail",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=theme["text"],
        fontsize=9.2,
        fontweight="bold",
    )
    ax.text(
        1.0,
        1.012,
        f"order: {sort}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=theme["muted"],
        fontsize=7,
    )


def _environment_lines(sysinfo: dict) -> tuple[str, str]:
    hardware = []
    if sysinfo.get("platform"):
        hardware.append(str(sysinfo["platform"]))
    cpu = sysinfo.get("cpu")
    threads = sysinfo.get("cpu_count")
    if cpu or threads:
        bits = [str(cpu)] if cpu else []
        if threads:
            bits.append(f"{threads} threads")
        hardware.append(" · ".join(bits))
    if sysinfo.get("memory_gb"):
        memory = _number(sysinfo["memory_gb"])
        if memory is not None:
            hardware.append(f"{memory:g} GB RAM")
    if sysinfo.get("gpu"):
        gpu = str(sysinfo["gpu"])
        try:
            gpu_count = int(sysinfo.get("gpu_count") or 1)
        except (TypeError, ValueError):
            gpu_count = 1
        if gpu_count > 1:
            gpu += f" (x{gpu_count})"
        hardware.append(gpu)
    if sysinfo.get("cuda"):
        hardware.append(f"CUDA {sysinfo['cuda']}")
    if sysinfo.get("torch_xla"):
        hardware.append(f"torch_xla {sysinfo['torch_xla']}")

    software = []
    if sysinfo.get("python"):
        software.append(f"Python {sysinfo['python']}")
    if sysinfo.get("torch"):
        software.append(f"torch {sysinfo['torch']}")
    if sysinfo.get("onnxruntime"):
        software.append(f"ONNX Runtime {sysinfo['onnxruntime']}")
    providers = sysinfo.get("onnxruntime_providers")
    if providers:
        if isinstance(providers, str):
            providers = [providers]
        software.append("providers: " + ", ".join(str(p) for p in providers))
    if sysinfo.get("ispy"):
        software.append(f"iSpy {sysinfo['ispy']}")
    if sysinfo.get("hostname"):
        software.append(str(sysinfo["hostname"]))
    return " · ".join(hardware), " · ".join(software)


def _draw_footer(ax, payload: dict, theme: dict, show_system: bool) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(
        [0.0, 1.0],
        [0.97, 0.97],
        transform=ax.transAxes,
        color=theme["rule"],
        linewidth=0.8,
        clip_on=False,
    )
    legend = (
        "FPS = end-to-end predict throughput.  MS = mean predict latency.  "
        "PRE / DEV / POST = stage milliseconds.  VISION = model-forward-only "
        "FPS.  Latency panel: dot = mean, line = p50-p99, tick = p95.  "
        "Green = fastest.  Gray = suspect (<0.5 ms).  Red = failed."
    )
    ax.text(
        0.0,
        0.8,
        textwrap.fill(legend, width=200),
        ha="left",
        va="top",
        color=theme["muted"],
        fontsize=6.8,
        linespacing=1.35,
    )
    if show_system:
        hardware, software = _environment_lines(payload.get("system") or {})
        ax.text(
            0.0,
            0.46,
            textwrap.fill(hardware, width=200),
            ha="left",
            va="top",
            color=theme["dim"],
            fontsize=6.8,
            linespacing=1.35,
        )
        ax.text(
            0.0,
            0.19,
            textwrap.fill(software, width=200),
            ha="left",
            va="top",
            color=theme["dim"],
            fontsize=6.8,
            linespacing=1.35,
        )


def render_report(
    payload: dict,
    out_path,
    title: str = "iSpy Benchmark",
    *,
    sections=None,
    theme: str = "light",
    dpi: int = 180,
    sort: str = "run",
    latency_scale: str = "linear",
):
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless boards and Colab have no display
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed - skipping chart (pip install matplotlib)"
        )
        return None

    rows = list(payload.get("all") or [])
    if not rows:
        return None
    rows = _sort_rows(rows, sort)
    if sections is None:
        selected = set(PLOT_SECTIONS)
    elif isinstance(sections, str):
        raw_sections = {
            part.strip().lower() for part in sections.split(",") if part.strip()
        }
        if "all" in raw_sections:
            selected = set(PLOT_SECTIONS)
        elif "none" in raw_sections:
            selected = set()
        else:
            selected = raw_sections
    else:
        selected = set(sections)
    palette = _THEMES.get(theme) or _THEMES["light"]
    multi_model = len({str(r.get("model") or "?") for r in rows}) > 1
    labels = [_row_label(r, multi_model) for r in rows]
    count = len(rows)

    fig_width = 16.5
    side = 0.55
    chart_left = 1.95
    chart_right = 0.55
    gap = 0.4
    top_margin = 0.32
    bottom_margin = 0.32
    header_h = 1.12
    chart_h = max(2.4, count * 0.42 + 0.9)
    stage_h = max(1.6, count * 0.32 + 0.8)
    table_h = max(1.2, count * 0.235 + 0.55)
    footer_h = 1.35 if "system" in selected else 0.6

    has_charts = "throughput" in selected or "latency" in selected
    blocks = []
    if "summary" in selected:
        blocks.append(("summary", 0.92))
    if has_charts:
        blocks.append(("charts", chart_h))
    if "stages" in selected:
        blocks.append(("stages", stage_h))
    if "table" in selected:
        blocks.append(("table", table_h))
    blocks.append(("footer", footer_h))
    fig_height = max(
        6.4,
        top_margin
        + bottom_margin
        + header_h
        + sum(gap + height for _name, height in blocks),
    )

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.unicode_minus": False,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor=palette["bg"])
        content_left = side / fig_width
        content_width = (fig_width - 2 * side) / fig_width

        header_rect = [
            content_left,
            (fig_height - top_margin - header_h) / fig_height,
            content_width,
            header_h / fig_height,
        ]
        header_ax = fig.add_axes(header_rect)
        _draw_header(
            header_ax,
            title or "iSpy Benchmark",
            _subtitle(payload, rows),
            palette,
        )

        cursor = fig_height - top_margin - header_h
        for name, height in blocks:
            cursor -= gap
            rect = [
                content_left,
                (cursor - height) / fig_height,
                content_width,
                height / fig_height,
            ]
            if name == "summary":
                ax = fig.add_axes(rect)
                _draw_summary(ax, rows, palette)
            elif name == "charts":
                chart_gap = 0.55
                chart_width = fig_width - chart_left - chart_right
                half = (chart_width - chart_gap) / 2
                if "throughput" in selected and "latency" in selected:
                    left_rect = [
                        chart_left / fig_width,
                        (cursor - height) / fig_height,
                        half / fig_width,
                        height / fig_height,
                    ]
                    right_rect = [
                        (chart_left + half + chart_gap) / fig_width,
                        (cursor - height) / fig_height,
                        half / fig_width,
                        height / fig_height,
                    ]
                    _draw_throughput(
                        fig.add_axes(left_rect), rows, labels, palette
                    )
                    _draw_latency(
                        fig.add_axes(right_rect),
                        rows,
                        labels,
                        palette,
                        latency_scale,
                        show_labels=False,
                    )
                else:
                    chart_rect = [
                        chart_left / fig_width,
                        (cursor - height) / fig_height,
                        chart_width / fig_width,
                        height / fig_height,
                    ]
                    ax = fig.add_axes(chart_rect)
                    if "throughput" in selected:
                        _draw_throughput(ax, rows, labels, palette)
                    else:
                        _draw_latency(
                            ax, rows, labels, palette, latency_scale
                        )
            elif name == "stages":
                chart_rect = [
                    chart_left / fig_width,
                    (cursor - height) / fig_height,
                    (fig_width - chart_left - chart_right) / fig_width,
                    height / fig_height,
                ]
                ax = fig.add_axes(chart_rect)
                _draw_stages(ax, rows, labels, palette)
            elif name == "table":
                ax = fig.add_axes(rect)
                _draw_table(ax, rows, palette, sort)
            elif name == "footer":
                ax = fig.add_axes(rect)
                _draw_footer(ax, payload, palette, "system" in selected)
            cursor -= height

        fig.savefig(
            out_path,
            dpi=dpi,
            facecolor=palette["bg"],
            edgecolor="none",
        )
        plt.close(fig)
    return out_path
