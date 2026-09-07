"""Base classes for iSpy add-ons."""

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

    # Code Breakdown opt-in. Set breakdown_label to a short display name to
    # include this add-on's per-tick work as its own series in the Metrics
    # page "Code Breakdown" chart. Small add-ons can leave it None - they
    # stay lumped into their aggregate (trackers/utilities/vision) and never
    # show a row of their own. breakdown_color is an optional CSS color
    # override; metrics picks one from its palette when left as None.
    breakdown_label: str | None = None
    breakdown_color: str | None = None

    def get_breakdown_parts(self) -> dict:
        if not self.breakdown_label:
            return {}
        key = getattr(self, "plugin_name", self.__class__.__name__)
        return {key: (self.breakdown_label, self.breakdown_color)}

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
        return None

    def declared_output_key(self) -> str | None:
        key, _err = validate_output_key(self.config.get("output_key"))
        return key

    def publish_output(self, frame_data: dict, value, output_key: str | None = None) -> bool:
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

    def get_code_parts(self) -> dict:
        return {}

    def get_code_times(self) -> dict:
        return {}

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