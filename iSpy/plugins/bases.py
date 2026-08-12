"""Base classes for all iSpy add-ons (trackers, utilities, frame processors).

An add-on is enabled by being present in the config
(``plugins.<type>.<name>``); the value of that entry is the add-on's OWN
settings dict. Add-ons are constructed with a context dict::

    context["config"]          -> iSpyAddonConfig view of this add-on's
                                  settings (its schema defaults are merged in)
    context["global_config"]   -> the global iSpyConfig
    context["cameras"]         -> list of active vision cameras
    context["flask_app"]       -> the Flask app when running in web mode,
                                  otherwise None
    context["vision_instance"] -> the running iSpy instance (set late)

Add-ons expose their configurable settings declaratively through
``config_schema()`` (mirroring vision pipelines) so the web UI can render a
settings editor and the loader can apply defaults.
"""

from abc import ABC, abstractmethod
from iSpy.config.iSpyConfig import iSpyAddonConfig

# Settings types the add-on settings editor understands.
_SCHEMA_TYPES = ("text", "number", "toggle")
_SCHEMA_TYPE_FALLBACK = {
    bool: "toggle",
    int: "number",
    float: "number",
}


def default_settings_from_schema(config_schema: dict) -> dict:
    """Extract ``{key: default}`` from an add-on's config_schema()."""
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
    """Shared ground for every add-on type: context handling, the per-add-on
    settings view, and the declarative settings schema."""

    def __init__(self, context: dict):
        StatusMixin.__init__(self)
        self.context: dict = context or {}
        raw = self.context.get("config")
        if not isinstance(raw, iSpyAddonConfig):
            raw = raw if isinstance(raw, dict) else {}
            raw = iSpyAddonConfig(raw)
        self.config = raw
        # Merge the add-on's schema defaults (absent keys only) so a config
        # entry of {} still yields working settings without persisting them.
        for key, value in default_settings_from_schema(self.config_schema()).items():
            self.config.setdefault(key, value)

    @property
    def global_config(self):
        return self.context.get("global_config")

    @classmethod
    def config_schema(cls) -> dict:
        """Declare this add-on's configurable settings:

            {"min_conf": {"type": "number", "label": "Min Confidence",
                          "default": 0.5, "hint": "..."},
             "enabled":  {"type": "toggle", "label": "...", "default": True}}

        Returns {} when the add-on needs no settings beyond being enabled."""
        return {}

    @classmethod
    def default_settings(cls) -> dict:
        """{key: default} resolved from config_schema()."""
        return default_settings_from_schema(cls.config_schema())


class TrackerBase(AddonBase):
    def __init__(self, context: dict):
        AddonBase.__init__(self, context)

    def start(self):
        pass

    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        return fuel_list

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