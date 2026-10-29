import logging
from abc import ABC, abstractmethod
from iSpy.config.iSpyConfig import iSpyAddonConfig
from iSpy.core import network_policy

_SCHEMA_TYPES = ("text", "number", "toggle", "list", "range")
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


# offline-mode plumbing shared by add-ons and pipelines: needs_internet = True
# or network_requirements(); require_network/try_online guard online-only actions
class NetworkMixin:
    # shortcut for add-ons whose every feature needs the internet. a subclass
    # that overrides network_requirements() replaces this entirely
    needs_internet = False

    # what this add-on needs the internet for, judged from its settings
    @classmethod
    def network_requirements(cls, settings: dict) -> list:
        if getattr(cls, "needs_internet", False):
            return [
                network_policy.NetworkRequirement(
                    name=getattr(cls, "plugin_name", cls.__name__),
                    reason="add-on uses the internet for everything it does",
                    critical=True,
                )
            ]
        return []

    @property
    def network_allowed(self) -> bool:
        return not network_policy.is_offline()

    # raises OfflineError when the feature would need the internet - call it
    # before an optional online action so it fails loud instead of hanging
    def require_network(self, feature: str) -> None:
        if network_policy.is_offline():
            raise network_policy.OfflineError(
                f"offline mode: {feature} needs the internet"
            )

    # runs fn when online, else records one degraded note and returns fallback.
    # flaky connections land here too so a dead network can't kill the tick
    def try_online(self, feature: str, fn, fallback=None):
        if not self.network_allowed:
            self._record_offline_feature(feature, "offline mode")
            return fallback
        try:
            return fn()
        except network_policy.network_exceptions() as exc:
            self._record_offline_feature(feature, f"network error: {exc}")
            return fallback

    def _record_offline_feature(self, feature: str, reason: str) -> None:
        network_policy.record_degraded(
            type(self).__name__,
            getattr(self, "plugin_name", type(self).__name__),
            feature,
            reason,
        )

    # schema fields marked requires_network get reset to their default while
    # offline, so an online-only knob can't stay on and keep reaching out
    def _ignore_offline_settings(self) -> None:
        if not network_policy.is_offline():
            return
        try:
            schema = self.config_schema()
        except Exception:
            return
        for key, defn in schema.items():
            if not (isinstance(defn, dict) and defn.get("requires_network")):
                continue
            default = defn.get("default")
            if self.config.get(key) != default:
                if hasattr(self.config, "set"):
                    self.config.set(key, default)
                else:
                    self.config[key] = default
                self._record_offline_feature(
                    key, "setting needs the internet - ignored while offline"
                )


class AddonBase(NetworkMixin, StatusMixin):
    # a camera assigned a pipeline not listed here is flagged incompatible
    supported_pipelines: tuple | None = None

    # show this add-on as its own series in Metrics
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
        for key, value in default_settings_from_schema(self.config_schema()).items():
            self.config.setdefault(key, value)
        self._ignore_offline_settings()

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

    # per-camera action buttons the web UI renders on the camera card. each
    # entry is {"id", "label", "kind": "button"|"toggle", "value", "title"}
    def get_actions(self) -> list[dict]:
        return []

    # the route only forwards ids that get_actions() declared, so an action
    # that is not currently offered can never be dispatched
    def invoke_action(self, action_id: str, value=None) -> dict:
        return {"ok": False, "error": "unsupported"}


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

    def publish_output(
        self, frame_data: dict, value, output_key: str | None = None
    ) -> bool:
        if not isinstance(frame_data, dict):
            return False
        key = output_key or self.declared_output_key()
        if not key:
            return False
        addon_data = frame_data.setdefault("addon_data", {})
        if key in addon_data:
            logging.getLogger(__name__).debug(
                "addon_data['%s'] overwritten by %s",
                key,
                type(self).__name__,
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
    seen: dict[str, list[str]] = {}
    for name, inst in utilities.items():
        declared = getattr(inst, "declared_output_key", lambda: None)()
        if not declared:
            continue
        seen.setdefault(declared, []).append(name)
    return {key: names for key, names in seen.items() if len(names) > 1}


class VisionBase(NetworkMixin, ABC):
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
        return True, "ready"

    @classmethod
    def needs_model_backend(cls) -> bool:
        return False

    def start(self):
        pass

    def get_debug_data(self) -> dict:
        return {}

    def get_debug_frame(self, frame):
        return None

    def plot(self, frame):
        return frame

    # per-camera action buttons - same contract as AddonBase.get_actions()
    def get_actions(self) -> list[dict]:
        return []

    def invoke_action(self, action_id: str, value=None) -> dict:
        return {"ok": False, "error": "unsupported"}

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def destroy(self):
        pass
