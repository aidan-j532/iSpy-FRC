import threading

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase

# All pipelines return (list[Object], frame) with the same JSON fields.
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
        if not build:
            return run
        lowered = run.lower()
        # build-style messages (optimizing / downloading / ...) already carry
        # the lifecycle story, so a generic run-slot message sitting on top of
        # them (the default "initializing", or a preparing/loading sibling) is
        # just noise - never surface e.g. "optimizing\ninitializing". Terminal
        # run states (ready / error) instead supersede a finished build message.
        # Real run info like a calibration warning still shows through next to
        # the build message, so the UI keeps that overlap.
        if lowered.startswith(
            self._BUILD_PREFIXES + ("initializing", "preparing", "loading", "working")
        ):
            return build
        if lowered.startswith(("ready", "using", "error", "failed", "stopped")):
            return run
        return f"{build}\n{run}"

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
        statuses = self.__dict__.setdefault(
            "_statuses", {"run": "initializing", "build": None}
        )
        lowered = status.lower()
        if lowered.startswith(self._BUILD_PREFIXES):
            statuses["build"] = status
        else:
            statuses["run"] = status

    def get_health(self) -> dict:
        state = self.get_state()
        status = (self.get_status() or "unknown").splitlines()[0]
        level = None
        try:
            level, _ = self.calibration_status()
        except Exception:
            level = None

        if state == "error" or level == "red":
            color = "red"
        elif level == "yellow" or state in (
            "optimizing",
            "downloading",
            "initializing",
        ):
            color = "yellow"
        else:
            color = "green"

        metrics = []
        try:
            count = len(getattr(self, "_last_objects", []) or [])
            metrics.append({"label": "Detections", "value": str(count)})
        except Exception:
            pass
        try:
            age_ms = round(self.get_frame_age() * 1000, 1)
            metrics.append({"label": "Frame age", "value": f"{age_ms}ms"})
        except Exception:
            pass

        return {
            "color": color,
            "state": status,
            "metrics": metrics,
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
        return [o.to_dict() if hasattr(o, "to_dict") else o for o in (objects or [])]

    @classmethod
    def serialize_frame_data(cls, frame_data: dict) -> dict:
        out = {
            k: v
            for k, v in frame_data.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        out["detections"] = cls.serialize_detections(frame_data.get("detections"))
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
        return cls.hardware

    def active_hardware(self) -> str | None:
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
                return (
                    float(calibration.get("focal_length_pixels") or 0) > 0
                    or float(calibration.get("fov") or 0) > 0
                )
            except (TypeError, ValueError):
                return False
        if section == "pnp":
            return bool(calibration.get("pnp"))
        return True

    def calibration_ready(self) -> bool:
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
        try:
            if self.in_calibration_mode():
                return True
        except Exception:
            pass
        return self.calibration_ready()

    def needs_calibration_to_run(self) -> bool:
        return True

    def calibration_status(self) -> tuple[str, str | None]:
        if self.calibration_ready():
            return "ready", None
        if self.needs_calibration_to_run():
            return "red", "Needs Calibration"
        return "yellow", "Needs Calibration for Better Accuracy"

    def _gate_uncalibrated(self, frame):
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
