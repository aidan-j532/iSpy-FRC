from abc import ABC, abstractmethod

class StatusMixin:
    def __init__(self):
        self._status = "idle"

    def update_status(self, status: str) -> None:
        self._status = status

    def get_status(self) -> str:
        return getattr(self, "_status", "unknown")

class TrackerBase(StatusMixin):
    def __init__(self, config):
        StatusMixin.__init__(self)

    def start(self): pass
    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        return fuel_list
    def stop(self): pass


class FrameProcessorBase(StatusMixin):
    def __init__(self, config):
        StatusMixin.__init__(self)
    
    def start(self):
        pass

    def process(self, frame):
        return frame

    def stop(self):
        pass

class UtilityBase(StatusMixin):
    def __init__(self, context: dict):
        StatusMixin.__init__(self)
        
    def start(self):
        pass

    def update(self, frame_data: dict):
        pass

    def get_robot_pose(self):
        """Override in network utility to provide pose. Default returns None."""
        return None

    def stop(self):
        pass

class VisionBase(ABC):

    plugin_name = "base"
    
    def __init__(self, context: dict):
        self.context = context

    @classmethod
    def config_schema(cls) -> dict:
        """Return {} if this plugin needs no extra config beyond the
        standard camera fields."""
        return {}

    def is_ready(self) -> tuple[bool, str]:
        """Cheap, idempotent, safe to call every boot cycle. Returns
        (ready: bool, status: str) where status is a short human-readable
        state like "ready", "loading weights", "optimizing (rknn build)",
        "using unoptimized .pt fallback", "error: <reason>". Must NEVER
        block on a multi-minute operation - if work is needed and isn't
        already running, kick it off as a background job/subprocess and
        return (False, "..."). If already in flight, just report status.
        Default: ready immediately, no background work needed."""
        return True, "ready"

    @classmethod
    def needs_model_backend(cls) -> bool:
        """True if this pipeline requires a model file, download, or
        conversion step and should participate in the readiness scan
        beyond simple __init__ success."""
        return False

    def start(self):
        pass

    def get_debug_data(self) -> dict:
        return {}

    def get_debug_frame(self, frame):
        return None

    def plot(self, frame):
        """Return a frame annotated by the vision pipeline for display."""
        return frame

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def destroy(self):
        pass