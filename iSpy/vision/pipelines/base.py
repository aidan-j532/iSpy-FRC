import threading

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase

# ---------------------------------------------------------------------------
# Universal pipeline output schema
# ---------------------------------------------------------------------------
# Every VisionPipeline.run() returns (list[Object], frame).  Serializing an
# Object via Object.to_dict() (or the helpers below) yields the following
# JSON-safe shape -- identical keys regardless of which pipeline produced it:
#
# {
#   "id": int,                      # stable track id
#   "name": str,                    # class name / tag label / "depth_center"
#   "confidence": float,            # 0..1 detector confidence (0 if N/A)
#   "x": float, "y": float, "z": float,          # robot-frame position (meters)
#   "roll": float, "pitch": float, "yaw": float, # robot-frame rotation (radians)
#   "depth_source": str,            # monocular | pnp | optical_flow | depth_model
#   "vis_type": str,                # renderer hint: generic | planar | ...
#   "vis_meta": dict,               # pipeline-specific payload, see below
#   "keypoints_3d": list | None,    # pose models: [[x, y, z], ...]
#   "ray_origin": [x,y,z] | None,       # camera ray in robot frame, if known
#   "ray_direction": [x,y,z] | None,
# }
#
# vis_type contracts:
#   generic  -- position/rotation are meaningful; vis_meta free-form.
#               object_detection & yolo_world put {"kind": "detection", ...};
#               optical_flow puts its velocity dict {"kind": "velocity", vx, vy,
#               speed, heading_deg, ...}; depth_anything puts
#               {"kind": "depth", depth_estimate, max_depth, ...}.
#   planar   -- flat tag/code solved by PnP (april_tag / qr_code): vis_meta
#               carries {"tag_id" | "payload", "size", ...}; rotation is exact.
OUTPUT_SCHEMA_VERSION = 1


class VisionPipeline(Camera, VisionBase):
    plugin_name = "pipeline"
    output_schema_version = OUTPUT_SCHEMA_VERSION

    # Build-style messages (optimizing / building / downloading) can overlap
    # with run-style ones (calibration warnings), so keep them in separate
    # slots and surface both to the UI.
    _BUILD_PREFIXES = ("optimizing", "building", "converting", "download")

    def __init__(self, camera_config, input_size, grayscale):
        self._statuses = {"run": "initializing", "build": None}
        Camera.__init__(self, camera_config, input_size, grayscale)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self):
        pass

    def is_ready(self) -> tuple[bool, str]:
        level, msg = self.calibration_status()
        if level == "yellow":
            # runs, but approximate pose/depth until calibrated
            self._set_status(msg)
            return True, msg
        if level == "red":
            self._set_status(msg)
            return False, msg
        self._set_status("ready")
        return True, "ready"

    def get_status(self) -> str:
        build = getattr(self, "_statuses", {}).get("build")
        run = getattr(self, "_statuses", {}).get("run") or "initializing"
        if build:
            return f"{build}\n{run}"
        return run

    def get_state(self) -> str:
        status = self.get_status()
        lowered = status.lower()
        if status.startswith("error"):
            return "error"
        if lowered.startswith(("optimizing", "building", "converting")):
            return "optimizing"
        if lowered.startswith(("download", "loading", "preparing", "initializing")):
            return "downloading" if lowered.startswith("download") else "initializing"
        if lowered == "ready" or lowered.startswith("using"):
            return "ready"
        return "initializing"

    def _set_status(self, status: str):
        lowered = status.lower()
        if lowered.startswith(self._BUILD_PREFIXES):
            self._statuses["build"] = status
        else:
            self._statuses["run"] = status

    def get_health(self) -> dict:
        """Contribute a pipeline widget to the Health tab (optional hook).

        ``ok`` is False only when pipeline is genuinely blocked (e.g. red
        calibration level / error), never during transient build/optimize.
        """
        status = self.get_status()
        state = self.get_state()
        level = None
        ready = True
        try:
            level, _ = self.calibration_status()
            ready, _ = self.is_ready()
        except Exception:
            ready = False
        # hide the calibration line when it's just the default "ready"
        if not level:
            level = "n/a"
        return {
            "ok": bool(ready),
            "title": self.config.get("name", str(self.source)) if hasattr(self, "config") else str(getattr(self, "source", "camera")),
            "info": (status or "unknown").splitlines()[0],
            "rows": [
                {"label": "Pipeline", "value": getattr(self, "plugin_name", "pipeline")},
                {"label": "State", "value": state},
                {"label": "Calibration", "value": level},
                {"label": "Status", "value": status},
            ],
        }

    def process(self, frame):
        return self.run()

    def stop(self):
        self.destroy()

    # ------------------------------------------------------------------
    # Universal output serialization
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_detections(objects) -> list[dict]:
        """Serialize a pipeline's Object list to the universal schema.

        Pass-through for non-Object entries so partially-migrated consumers
        keep working.
        """
        return [
            o.to_dict() if hasattr(o, "to_dict") else o
            for o in (objects or [])
        ]

    @classmethod
    def serialize_frame_data(cls, frame_data: dict) -> dict:
        """JSON-safe view of frame_data for web/NT consumers.

        Detections are flattened to the universal schema; scalar metadata
        (fps, pipeline_name, ...) is passed through unchanged.
        """
        out = {
            k: v for k, v in frame_data.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        out["detections"] = cls.serialize_detections(
            frame_data.get("detections"))
        out["schema_version"] = OUTPUT_SCHEMA_VERSION
        return out

    # ------------------------------------------------------------------
    # Optimization (optional - only model-backed pipelines implement it)
    # ------------------------------------------------------------------

    def get_optimization_options(self) -> dict:
        return {}

    def optimize(self, **kwargs) -> str:
        return "not supported"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return False

    @classmethod
    def uses_user_model(cls) -> bool:
        return False

    @classmethod
    def show_common_fields(cls) -> bool:
        return True

    @classmethod
    def show_calibration(cls) -> bool:
        return True

    # Declares which calibration sections the calibration wizard should offer
    # (each section becomes a tab). Subclasses override to opt in/out. The
    # default is the universal ChArUco board intrinsics calibration - every
    # pipeline gets it unless it explicitly sets calibration_sections = [].
    # Valid tokens:
    #   "charuco" -> ChArUco board intrinsics (camera_matrix + dist_coeffs)
    #   "focal"   -> known-object focal/FOV measurement
    #   "pnp"     -> pose keypoint 3D positions (for pose models)
    calibration_sections: list[str] = ["charuco"]

    @classmethod
    def requires_calibration(cls) -> bool:
        """Whether this pipeline needs calibration configured before it will
        emit (3D) detections. Defaults to 'any calibration_sections declared';
        subclasses can override for finer control."""
        return bool(cls.calibration_sections)

    # ------------------------------------------------------------------
    # Hardware / compute backend
    # ------------------------------------------------------------------
    # Each pipeline declares every hardware target it can run inference on
    # (e.g. ("npu", "tpu", "gpu", "cpu") for a model-backed detector). A pure
    # CV pipeline like AprilTag runs on the CPU, which is already reported as
    # the CPU load, so it declares the empty tuple and gets no label.
    #
    # active_hardware() resolves (per instance, at runtime) which one the
    # pipeline's loaded backend is actually using. Subclasses override it;
    # the base reports None.
    hardware: tuple[str, ...] = ()

    @classmethod
    def hardware_options(cls) -> tuple[str, ...]:
        """Every hardware target this pipeline can run on (empty = none)."""
        return cls.hardware

    def active_hardware(self) -> str | None:
        """The hardware this pipeline's inference is currently using, or None
        when it has no dedicated accelerator (its CPU usage is already shown
        by the system CPU reading)."""
        return None

    @staticmethod
    def _section_satisfied(section: str, calibration: dict) -> bool:
        if section == "charuco":
            return bool(
                calibration.get("camera_matrix")
                and calibration.get("dist_coeffs") is not None
            )
        if section == "focal":
            try:
                return float(calibration.get("focal_length_pixels") or 0) > 0 or float(
                    calibration.get("fov") or 0
                ) > 0
            except (TypeError, ValueError):
                return False
        if section == "pnp":
            return bool(calibration.get("pnp"))
        return True

    def calibration_ready(self) -> bool:
        """True when the pipeline may run: either it needs no calibration, or at
        least one of its declared calibration sections is fully configured."""
        if not self.requires_calibration():
            return True
        config = getattr(self, "config", None)
        if config is None:
            # never a real pipeline instance (e.g. tests built via __new__) -
            # don't gate
            return True
        try:
            calibration = config.get("calibration") or {}
        except Exception:
            return True
        return any(
            self._section_satisfied(section, calibration)
            for section in self.calibration_sections
        )

    def _calibration_processable(self) -> bool:
        """Gate used at the top of run(): False means the camera goes back to a
        plain frame feed until its required calibration exists. The calibration
        wizard's live feed is always allowed through."""
        try:
            if self.in_calibration_mode():
                return True
        except Exception:
            pass
        return self.calibration_ready()

    def needs_calibration_to_run(self) -> bool:
        """True when the pipeline **cannot produce meaningful output** without
        calibration (e.g. pose models that need accurate 3D, or pipelines
        whose detection math depends on camera intrinsics).

        Returns False for pipelines that can still detect objects/tags/codes
        without calibration — those show a yellow warning instead of blocking."""
        return True

    def calibration_status(self) -> tuple[str, str | None]:
        """Classify calibration state for the UI dot + status text.

        Returns one of:
          ("ready", None)           — calibrated, green dot
          ("yellow", message)       — uncalibrated but detection still runs
          ("red", message)          — uncalibrated and pipeline is blocked
        """
        if self.calibration_ready():
            return "ready", None
        if self.needs_calibration_to_run():
            return "red", "Needs Calibration"
        return "yellow", "Needs Calibration for Better Accuracy"

    def _gate_uncalibrated(self, frame):
        """run() top helper: returns ([], frame) passthrough when the pipeline
        needs calibration that is not configured yet, else None."""
        level, msg = self.calibration_status()
        if level == "ready":
            return None
        if level == "red":
            self._set_status(msg)
            return [], frame
        # yellow: detection runs, just show a warning
        self._set_status(msg)
        return None


class BackgroundPreparedPipeline(VisionPipeline):
    def __init__(self, camera_config, input_size, grayscale):
        self._prep_thread: threading.Thread | None = None
        self._prep_started = False
        self._prep_lock = threading.Lock()
        super().__init__(camera_config, input_size, grayscale)

    def prepare(self):
        with self._prep_lock:
            if self._prep_started:
                return
            self._prep_started = True
            self._set_status("initializing")
            self._prep_thread = threading.Thread(
                target=self._prepare,
                daemon=True,
                name=f"Prepare-{getattr(self, 'plugin_name', 'pipeline')}",
            )
            self._prep_thread.start()

    def _prepare(self):
        raise NotImplementedError

    def _preparing(self) -> bool:
        thread = getattr(self, "_prep_thread", None)
        return bool(thread is not None and thread.is_alive())
