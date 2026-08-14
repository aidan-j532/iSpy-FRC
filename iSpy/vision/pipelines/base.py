"""Common vision pipeline base.

Every vision capability in iSpy (object detection, AprilTag, QR codes, YOLO
World, Depth Anything, ...) shares one lifecycle and is treated generically:

    Pipeline created
        ↓
    prepare()  (background, pipeline-owned)
        ↓
    is_ready() -> (bool, status)
        ↓
    run()      (process frames)

The pipeline owns its config, prep, model downloads, optimization, readiness
and processing. Generic app code only knows the interface defined here.
"""

import threading

from iSpy.vision.Camera import Camera
from iSpy.plugins.bases import VisionBase


class VisionPipeline(Camera, VisionBase):
    plugin_name = "pipeline"

    def __init__(self, camera_config, input_size, grayscale):
        self._status = "initializing"
        Camera.__init__(self, camera_config, input_size, grayscale)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self):
        """start prep without blocking - runs on a background thread"""
        pass

    def is_ready(self) -> tuple[bool, str]:
        """(ready, status) for the whole pipeline. Only "ready" when
        everything needed to process frames is prepared. Never blocks."""
        self._set_status("ready")
        return True, "ready"

    def get_status(self) -> str:
        """short human-readable status (last known)"""
        return getattr(self, "_status", "initializing")

    def get_state(self) -> str:
        """coarse machine-readable state: ready/optimizing/downloading/
        initializing/error. UI uses this instead of parsing raw statuses."""
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
        """process a single frame; aliases run() by default"""
        return self.run()

    def stop(self):
        """stop bg work and release resources"""
        self.destroy()

    # ------------------------------------------------------------------
    # Optimization (optional - only model-backed pipelines implement it)
    # ------------------------------------------------------------------

    def get_optimization_options(self) -> dict:
        """describe supported optimization options, or {} if none"""
        return {}

    def optimize(self, **kwargs) -> str:
        """start a model optimization as a bg job; returns a status string"""
        return "not supported"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return False

    @classmethod
    def uses_user_model(cls) -> bool:
        """True if users can pick the model file (UI shows a picker)"""
        return False

    @classmethod
    def show_common_fields(cls) -> bool:
        """True if the UI should show the shared mount/calibration fields.
        Those feed every pipeline's camera-to-robot transform, so default
        on; pipelines that never use the transform may opt out."""
        return True

    @classmethod
    def show_calibration(cls) -> bool:
        """True if the UI should show the camera calibration fields (known
        distance, game piece size, object height, FOV) and the calibration
        wizard. Only pipelines that turn focal length / object size into a
        distance estimate need them; depth-estimation pipelines opt out."""
        return True


class BackgroundPreparedPipeline(VisionPipeline):
    """VisionPipeline whose prep runs on a background thread.

    Pipelines that download/build models shouldnt block construction on
    network/convert work: __init__ must be cheap so boot can construct every
    camera concurrently, then wait on is_ready(). Subclasses implement
    _prepare(); the thread starts automatically in prepare().
    """

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
        """actual prep work, runs once on a daemon thread"""
        raise NotImplementedError

    def _preparing(self) -> bool:
        thread = getattr(self, "_prep_thread", None)
        return bool(thread is not None and thread.is_alive())
