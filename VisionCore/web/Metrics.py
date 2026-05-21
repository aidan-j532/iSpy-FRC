from collections import deque
import logging
import time

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Metrics:
    SERIES = {
        "loop_s": ("Loop time", "ms", 1000.0),
        "vision_s": ("Vision time", "ms", 1000.0),
        "camera_lag_s": ("Camera lag", "ms", 1000.0),
        "flask_s": ("Flask update", "ms", 1000.0),
        "network_s": ("Network Tables", "ms", 1000.0),
        "health_s": ("Health update", "ms", 1000.0),
    }

    def __init__(
        self, window: int = 30, log_every: int = 30, output_file: str = "metrics.html"
    ):
        self.window = window
        self.log_every = log_every
        self._itr = 0
        self._data: dict[str, deque] = {}
        self.logger = logging.getLogger(__name__)
        self._timeline: dict[str, list[tuple[float, float]]] = {
            k: [] for k in self.SERIES
        }
        self._rolling: dict[str, deque] = {}
        self.output_file = output_file
        self._start_wall = time.time()

    def record(self, **kwargs: float):
        t = time.time() - self._start_wall
        for key, val in kwargs.items():
            if val is None:
                continue
            if key not in self._data:
                self._data[key] = deque(maxlen=self.window)
            self._data[key].append(val)
            if key in self.SERIES:
                self._timeline[key].append((t, val))

    def avg(self, key: str) -> float | None:
        if key not in self._data or len(self._data[key]) == 0:
            return None
        return sum(self._data[key]) / len(self._data[key])

    def tick(self):
        self._itr += 1
        if self._itr % self.log_every == 0:
            self._log()

    def _fmt(self, key: str, unit: str = "ms", scale: float = 1000.0) -> str:
        v = self.avg(key)
        return f"{v * scale:.1f}{unit}" if v is not None else "n/a"

    def _log(self):
        loop_ms = self._fmt("loop_s")
        vision_ms = self._fmt("vision_s")
        cam_lag_ms = self._fmt("camera_lag_s")
        flask_ms = self._fmt("flask_s")
        fps = self.avg("loop_s")
        health = self._fmt("health_s")
        fps_str = f"{1/fps:.1f}" if fps else "n/a"

        # self.logger.info(
        #     f"[Metrics @{self._itr}] "
        #     f"loop={loop_ms}  fps={fps_str}  "
        #     f"vision={vision_ms}  "
        #     f"cam_lag={cam_lag_ms}  "
        #     f"flask={flask_ms}"
        #     f"health={health}"
        # )

    def destroy(self):
        self._log_final_summary()
        if not PLOTLY_AVAILABLE:
            self.logger.warning(
                "plotly not installed - skipping metrics export. "
                "Run: pip install plotly --break-system-packages"
            )
            return
        self._write_html()

    def _log_final_summary(self):
        self.logger.info("=== Final Metrics Summary ===")
        fps = self.avg("loop_s")
        self.logger.info(f"  Total iterations : {self._itr}")
        self.logger.info(
            f"  Avg FPS          : {1/fps:.1f}" if fps else "  Avg FPS          : n/a"
        )
        # for key, (label, unit, scale) in self.SERIES.items():
        #     v = self.avg(key)
        #     if v is not None:
        #         self.logger.info(f"  {label:<18}: {v * scale:.2f}{unit}")

    def _write_html(self):
        fig = go.Figure()

        for key, (label, unit, scale) in self.SERIES.items():
            points = self._timeline.get(key, [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] * scale for p in points]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=f"{label} ({unit})",
                    hovertemplate=f"%{{x:.2f}}s - %{{y:.2f}}{unit}<extra>{label}</extra>",
                )
            )

        # Add derived FPS trace from loop_s
        loop_points = self._timeline.get("loop_s", [])
        if loop_points:
            xs = [p[0] for p in loop_points]
            fps = [1.0 / p[1] if p[1] > 0 else 0 for p in loop_points]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=fps,
                    mode="lines",
                    name="FPS",
                    yaxis="y2",
                    line=dict(dash="dot"),
                    hovertemplate="%{x:.2f}s - %{y:.1f} fps<extra>FPS</extra>",
                )
            )

        fig.update_layout(
            title="Vision Pipeline Metrics",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Latency (ms)"),
            yaxis2=dict(title="FPS", overlaying="y", side="right", showgrid=False),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
            template="plotly_dark",
        )

        fig.write_html(self.output_file, include_plotlyjs=True)
        self.logger.info(f"Metrics timeline saved to: {self.output_file}")
        print(f"[Metrics] Timeline saved -> {self.output_file}")