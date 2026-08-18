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
