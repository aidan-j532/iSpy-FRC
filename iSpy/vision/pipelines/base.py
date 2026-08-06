"""Common vision pipeline base.

Every vision capability in iSpy is an equal peer: object detection,
AprilTag, QR codes, line tracking, YOLO World, Depth Anything, and any
future pipeline. They share one lifecycle and are treated generically by
the application:

    Pipeline created
        ↓
    prepare()  (background, pipeline-owned)
        ↓
    is_ready() -> (bool, status)
        ↓
    run()      (process frames)

The pipeline owns its configuration, preparation, model downloads,
optimization, quantization, initialization, readiness and processing.
Generic application code (boot, web, iSpy runtime) only knows the common
interface defined here.
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
        """Begin whatever preparation this pipeline requires, without
        blocking. Preparation runs in the background (a thread owned by
        the pipeline); ``is_ready()`` reports progress. The default
        implementation requires no preparation."""
        pass

    def is_ready(self) -> tuple[bool, str]:
        """(ready, status) for the whole pipeline.

        A pipeline is only "ready" when everything it needs to process
        frames has been prepared successfully:
        * ready       - fully prepared, can process frames
        * optimizing  - a model optimization/conversion is in flight
        * downloading - a required model/weights download is in flight
        * error: ...  - preparation failed (with a useful reason)

        Must never block on a multi-minute operation: if work is needed
        it should already be running in the background (see ``prepare``).
        """
        self._set_status("ready")
        return True, "ready"

    def get_status(self) -> str:
        """Short human-readable status of this pipeline (last known)."""
        return getattr(self, "_status", "initializing")

    def get_state(self) -> str:
        """Coarse machine-readable state derived from the status string:
        one of ``ready``, ``optimizing``, ``downloading``, ``initializing``
        or ``error``. Generic UI code uses this instead of parsing raw
        pipeline statuses."""
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
        """Process a single frame. By default aliases ``run()``."""
        return self.run()

    def stop(self):
        """Stop background work and release resources."""
        self.destroy()

    # ------------------------------------------------------------------
    # Optimization (optional - only model-backed pipelines implement it)
    # ------------------------------------------------------------------

    def get_optimization_options(self) -> dict:
        """Describe the optimization options this pipeline supports, or
        return {} if the pipeline has nothing to optimize. Used by generic
        UI code; the keys are pipeline-specific."""
        return {}

    def optimize(self, **kwargs) -> str:
        """Start an optimization/conversion of this pipeline's model as a
        background job. Returns a status string. The default implementation
        does nothing."""
        return "not supported"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return False

    @classmethod
    def uses_user_model(cls) -> bool:
        """True if this pipeline runs a user-selectable model file (e.g.
        object detection lets users pick any supported .pt). Pipelines that
        provision their own fixed models (YOLO World, Depth Anything) return
        False. Generic UI code uses this to decide whether to show a model
        picker."""
        return False

    @classmethod
    def show_common_fields(cls) -> bool:
        """True if the camera settings UI should show the shared mount/
        calibration fields (x/y/z/height/yaw/pitch) for this pipeline. These
        feed every pipeline's camera-to-robot transform, so they default to
        on; pipelines that never use the transform may opt out."""
        return True


class BackgroundPreparedPipeline(VisionPipeline):
    """VisionPipeline whose preparation runs on a background thread.

    Pipelines that download or build models (Depth Anything, YOLO World,
    ...) should not block construction on network/convert work: ``__init__``
    must be cheap so boot can construct every camera concurrently, then wait
    on ``is_ready()``. Subclasses implement ``_prepare()``; the thread is
    started automatically by ``prepare()``.
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
        """Do the actual preparation work. Runs once, on a daemon thread."""
        raise NotImplementedError

    def _preparing(self) -> bool:
        thread = getattr(self, "_prep_thread", None)
        return bool(thread is not None and thread.is_alive())
