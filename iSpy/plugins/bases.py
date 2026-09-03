"""base classes for all iSpy add-ons (trackers, utilities, frame processors).

an add-on is enabled by being present in the config (plugins.<type>.<name>);
the entry value is that add-on's own settings dict. add-ons get a context:
    config          -> iSpyAddonConfig view of THIS add-on's settings
    global_config   -> the global iSpyConfig
    cameras, flask_app, vision_instance
"""

import logging
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
    # vision pipeline plugin_names this add-on is known to work with.
    # None (default) means "works with any pipeline" - no compatibility
    # warning is shown on the Add-ons page. Set a tuple of pipeline
    # names to flag mismatches when a camera runs something else,
    # e.g. supported_pipelines = ("object_detection",).
    supported_pipelines: tuple | None = None

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

    @property
    def selection(self):
        return self.context.get("selection")

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

    def declared_output_key(self) -> str | None:
        """this utility's normalized output_key setting, or None if unset/invalid"""
        key, _err = validate_output_key(self.config.get("output_key"))
        return key

    def publish_output(self, frame_data: dict, value, output_key: str | None = None) -> bool:
        """expose a runtime value under frame_data["addon_data"][<output_key>].

        Values are namespaced under "addon_data" so user-configured keys can
        never clobber core frame_data entries (fps, detections, ...). Returns
        True if the value was written.
        """
        if not isinstance(frame_data, dict):
            return False
        key = output_key or self.declared_output_key()
        if not key:
            return False
        addon_data = frame_data.setdefault("addon_data", {})
        if key in addon_data:
            logging.getLogger(__name__).debug(
                "addon_data['%s'] overwritten by %s",
                key, type(self).__name__,
            )
        addon_data[key] = value
        return True

    def stop(self):
        pass


def validate_output_key(raw) -> tuple[str | None, str | None]:
    """validate a utility output_key setting -> (normalized_key, error_message).

    Normalizes surrounding whitespace. Keys must be non-empty strings without
    dots (dots are reserved for nested source paths like addon_data.<key>).
    """
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, f"Output Key must be a string, got {type(raw).__name__}"
    key = raw.strip()
    if not key:
        return None, "Output Key cannot be empty"
    if "." in key:
        return None, "Output Key cannot contain dots"
    return key, None


def find_duplicate_output_keys(utilities: dict) -> dict[str, list[str]]:
    """map conflicting output_key -> [utility names] across enabled utilities."""
    seen: dict[str, list[str]] = {}
    for name, inst in utilities.items():
        declared = getattr(inst, "declared_output_key", lambda: None)()
        if not declared:
            continue
        seen.setdefault(declared, []).append(name)
    return {key: names for key, names in seen.items() if len(names) > 1}


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