"""base classes for all iSpy add-ons (trackers, utilities, frame processors).

an add-on is enabled by being present in the config (plugins.<type>.<name>);
the entry value is that add-on's own settings dict. add-ons get a context:
    config          -> iSpyAddonConfig view of THIS add-on's settings
    global_config   -> the global iSpyConfig
    cameras, flask_app, vision_instance
"""

from abc import ABC, abstractmethod
from iSpy.config.iSpyConfig import iSpyAddonConfig

# settings types the add-on settings editor understands
_SCHEMA_TYPES = ("text", "number", "toggle", "list")
_SCHEMA_TYPE_FALLBACK = {
    bool: "toggle",
    int: "number",
    float: "number",
}


def default_settings_from_schema(config_schema: dict) -> dict:
    defaults = {}
    for key, defn in config_schema.items():
        if isinstance(defn, dict) and "default" in defn:
            defaults[key] = defn["default"]
    return defaults


class StatusMixin:
    def __init__(self):
        self._status = "idle"

    def update_status(self, status: str) -> None:
        self._status = status

    def get_status(self) -> str:
        return getattr(self, "_status", "unknown")


class AddonBase(StatusMixin):
    def __init__(self, context: dict):
        StatusMixin.__init__(self)
        self.context: dict = context or {}
        raw = self.context.get("config")
        if not isinstance(raw, iSpyAddonConfig):
            raw = raw if isinstance(raw, dict) else {}
            raw = iSpyAddonConfig(raw)
        self.config = raw
        # merge schema defaults (absent keys only) so a config entry of {} still works
        for key, value in default_settings_from_schema(self.config_schema()).items():
            self.config.setdefault(key, value)

    @property
    def global_config(self):
        return self.context.get("global_config")

    @classmethod
    def config_schema(cls) -> dict:
        """declare this add-on's configurable settings (see examples for the format); {} if none needed"""
        return {}

    @classmethod
    def default_settings(cls) -> dict:
        return default_settings_from_schema(cls.config_schema())


class TrackerBase(AddonBase):
    def __init__(self, context: dict):
        AddonBase.__init__(self, context)

    def start(self):
        pass

    def update(self, detections, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        return detections

    def stop(self):
        pass


class FrameProcessorBase(AddonBase):
    def __init__(self, context: dict):
        AddonBase.__init__(self, context)

    def start(self):
        pass

    def process(self, frame):
        return frame

    def stop(self):
        pass


class UtilityBase(AddonBase):
    def __init__(self, context: dict):
        AddonBase.__init__(self, context)

    def start(self):
        pass

    def update(self, frame_data: dict):
        pass

    def get_robot_pose(self):
        """override in the network utility to give pose; defaults to None"""
        return None

    def stop(self):
        pass


class VisionBase(ABC):

    plugin_name = "base"

    def __init__(self, context: dict):
        self.context = context

    @classmethod
    def config_schema(cls) -> dict:
        return {}

    def is_ready(self) -> tuple[bool, str]:
        """(ready, status) checked every boot cycle. NEVER block on multi-minute
        work - kick it off as a bg job and report status. defaults to ready."""
        return True, "ready"

    @classmethod
    def needs_model_backend(cls) -> bool:
        """true if the pipeline needs a model/download/conversion and joins the readiness scan"""
        return False

    def start(self):
        pass

    def get_debug_data(self) -> dict:
        return {}

    def get_debug_frame(self, frame):
        return None

    def plot(self, frame):
        return frame

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def destroy(self):
        pass