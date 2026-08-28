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

    def __init__(self, camera_config, input_size, grayscale):
        self._status = "initializing"
        Camera.__init__(self, camera_config, input_size, grayscale)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self):
        pass

    def is_ready(self) -> tuple[bool, str]:
        self._set_status("ready")
        return True, "ready"

    def get_status(self) -> str:
        return getattr(self, "_status", "initializing")

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
        self._status = status

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
