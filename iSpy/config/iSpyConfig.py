import json
import logging
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path.cwd()

_MODEL_BACKED_PIPELINES = ("object_detection",)

_CAMERA_CORE_KEYS = {
    "name", "source", "device_id", "subsystem", "grayscale",
    "x", "y", "z", "height", "yaw", "pitch", "calibration",
    "exposure_time", "gain", "fps_cap",
    "brightness", "contrast", "saturation", "gamma",
    "white_balance", "tint",
    "csi", "path",
}

_VISION_MODEL_SETTINGS_KEYS = (
    "min_conf", "quantize", "quantization_dataset", "optimize",
    "target_format",
)

_LEGACY_SETTING_ALIASES = {
    "auto_opt": "optimize",
    "quantized": "quantize",
}

_ADDON_TYPES = ("trackers", "utilities", "frame_processors")

# Conversion factors: how many inches does 1 of the given unit equal.
_UNIT_TO_INCHES = {
    "inch": 1.0,
    "inches": 1.0,
    "foot": 12.0,
    "feet": 12.0,
    "meter": 1.0 / 0.0254,
    "meters": 1.0 / 0.0254,
    "centimeter": 1.0 / 2.54,
    "centimeters": 1.0 / 2.54,
    # FRC/WPILib: inputs are in inches (calibration convention)
    "frc": 1.0,
}

_UNIT_LABELS = {
    "inch": "in", "inches": "in",
    "foot": "ft", "feet": "ft",
    "meter": "m", "meters": "m",
    "centimeter": "cm", "centimeters": "cm",
    "frc": "in",
}


def unit_to_inches(value: float, unit: str) -> float:
    """Convert *value* from *unit* to inches (the internal math unit)."""
    return value * _UNIT_TO_INCHES.get(unit.lower().strip(), 1.0)


def unit_label(unit: str) -> str:
    """Return a short display label for *unit* (e.g. 'in', 'm', 'ft')."""
    return _UNIT_LABELS.get(unit.lower().strip(), unit)

# legacy top-level keys folded into individual add-ons. (key, addon type,
# addon name, target setting key) - used by _migrate_addons.
_ADDON_LEGACY_FOLDS = (
    ("dbscan",           "trackers",    "object_tracker", None),  # special: dict
    ("distance_threshold", "trackers",  "object_tracker", "distance_threshold"),
    ("stale_threshold",  "trackers",    "object_tracker", "stale_threshold"),
)

# health reporting is a core web module (HealthModule) - these opt-in
# utilities were removed; their settings fold into top-level keys instead
_MERGED_HEALTH_ADDONS = ("health_reporter", "status_reporter")

# legacy enabled flags - the flag value is discarded once it becomes add-on presence
_ADDON_LEGACY_FLAGS = {
    "use_network_tables": ("utilities", "network_table_handler"),
    "record_mode": ("utilities", "video_recorder"),
}

_ADDON_LEGACY_SETTINGS = {
    "network_tables_ip": ("utilities", "network_table_handler", "network_tables_ip"),
    "record_dir": ("utilities", "video_recorder", "record_dir"),
}


def _normalize_vision_model_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        return
    vm = settings.get("vision_model")
    if not isinstance(vm, dict):
        return
    for key in _VISION_MODEL_SETTINGS_KEYS:
        if key in vm and key in settings:
            del vm[key]
    for legacy, canonical in _LEGACY_SETTING_ALIASES.items():
        if legacy in vm and (canonical in vm or canonical in settings):
            del vm[legacy]
        if legacy in settings and canonical in settings:
            del settings[legacy]


def default_vision_model() -> dict:
    pts = sorted((_REPO_ROOT / "YoloModels" / "pytorch").glob("*.pt"))
    user_pts = [p for p in pts if not p.name.startswith("_default")]
    if user_pts:
        rel = f"YoloModels/pytorch/{user_pts[0].name}"
    else:
        pose = next((p for p in pts if p.name == "_default_pose.pt"), None)
        rel = f"YoloModels/pytorch/{pose.name}" if pose else "YoloModels/pytorch/_default_pose.pt"
    return {"file_path": rel, "source_pt": rel, "min_conf": 0.5}


def is_model_backed_pipeline(pipeline: str) -> bool:
    return str(pipeline or "") in _MODEL_BACKED_PIPELINES


def get_pipeline_name(cam_entry: dict) -> str:
    if not isinstance(cam_entry, dict):
        return "object_detection"
    p = cam_entry.get("pipeline")
    if isinstance(p, str) and p:
        return p
    if isinstance(p, dict):
        name = p.get("name")
        if isinstance(name, str) and name:
            return name
    return "object_detection"


def get_pipeline_settings(cam_entry: dict) -> dict:
    if not isinstance(cam_entry, dict):
        return {}
    p = cam_entry.get("pipeline")
    if isinstance(p, dict) and isinstance(p.get("settings"), dict):
        return p["settings"]
    return {k: v for k, v in cam_entry.items()
            if k not in _CAMERA_CORE_KEYS and k != "pipeline"}


def normalize_camera_entry(cam_entry: dict) -> dict:
    if not isinstance(cam_entry, dict):
        return cam_entry
    p = cam_entry.get("pipeline")
    if isinstance(p, dict):
        settings = p.get("settings")
        if not isinstance(settings, dict):
            p["settings"] = settings = {}
        for k, v in list(cam_entry.items()):
            if k in _CAMERA_CORE_KEYS or k == "pipeline":
                continue
            settings.setdefault(k, v)
            cam_entry.pop(k)
        return cam_entry
    if isinstance(p, str):
        settings = {
            k: v for k, v in cam_entry.items()
            if k not in _CAMERA_CORE_KEYS and k != "pipeline"
        }
        for k in settings:
            cam_entry.pop(k)
        cam_entry["pipeline"] = {"name": p, "settings": settings}
        return cam_entry
    # no pipeline key at all (raw JSON editor) - tag with the default.
    settings = {
        k: v for k, v in cam_entry.items()
        if k not in _CAMERA_CORE_KEYS and k != "pipeline"
    }
    for k in settings:
        cam_entry.pop(k)
    cam_entry["pipeline"] = {"name": "object_detection", "settings": settings}
    return cam_entry


def ensure_camera_entries_ready(camera_configs: dict) -> None:
    if not isinstance(camera_configs, dict):
        return
    for cam_cfg in camera_configs.values():
        if not isinstance(cam_cfg, dict):
            continue
        normalize_camera_entry(cam_cfg)
        if is_model_backed_pipeline(get_pipeline_name(cam_cfg)) and not isinstance(
            get_pipeline_settings(cam_cfg).get("vision_model"), dict
        ):
            get_pipeline_settings(cam_cfg)["vision_model"] = default_vision_model()
        _normalize_vision_model_settings(get_pipeline_settings(cam_cfg))


class iSpyConfig:
    def __init__(
        self, file_path: str = None, create: bool = True
    ):
        self.logger = logging.getLogger(__name__)

        self.default_config = {
            "num_gpus": "auto",
            "device": 0,
            # "frc" = FRC/WPILib convention: outputs in meters (what robot code
            # expects, matches Limelight/PhotonVision), calibration inputs in inches.
            "unit": "frc",
            "debug_mode": True,
            "frame_sync": False,
            "optimize": False,
            "log_level": "INFO",
            "log_file": "Outputs/log.txt",
            # seconds without a fresh frame before /health reports degraded
            "health_stale_threshold": 1.0,
            "metrics": True,
            "app_mode": True,
            "max_fps": 0,
            # reset to False by every `boot -f` (fresh install) so the web UI
            # shows its first-run tutorial once until the user dismisses it.
            "onboarding": {
                "completed": False,
            },
            "camera_configs": {
                "default_cam": {
                    "name": "default_cam",
                    "source": 0,
                    "subsystem": "field",
                    "grayscale": False,
                    "calibration": {
                        "distance": 0.0,
                        "game_piece_size": 0.0,
                        "size": 0,
                        "fov": 0,
                    },
                    # mount offsets stay on the cam entry; pipeline stuff lives under pipeline.settings
                    "yaw": 0,
                    "pitch": 0,
                    "height": 1.0,
                    "x": 0,
                    "y": 0,
                    "pipeline": {
                        "name": "object_detection",
                        "settings": {
                            "vision_model": {
                                # input.*/output.* are auto-detected from the model's
                                # _metadata.yaml sidecar - only override if its wrong.
                                "file_path": "YoloModels/pytorch/_default_pose.pt",
                                "source_pt": "YoloModels/pytorch/_default_pose.pt",
                                "min_conf": 0.5,
                                # quantization/rknn builds are opt-in per cam (Optimize
                                # toggle in the web UI) so a fresh install boots fast.
                                "quantize": False,
                                # Optional PnP for pose (translation stored on
                                # Box; rotation stored as roll/pitch/yaw on
                                # Box). Enable to get 3D position + orientation
                                # from 2D keypoints, and to render a 3D human
                                # skeleton in the web viewer (when
                                # keypoints_3d is present on the Object).
                                # "pnp": {
                                #     # Canonical COCO 17-keypoint skeleton
                                #     # (~1.8 m tall, origin at mid-hip).
                                #     "object_points": [...],
                                #     "camera_matrix": [[fx, 0, cx], ...],
                                #     "dist_coeffs": [0, 0, 0, 0, 0],
                                #     "min_keypoint_conf": 0.5,
                                #     # "mode": "flexible" | "rigid"
                                # },
                            },
                        },
                    },
                }
            },
            "plugins": {
                # enabled add-ons only - presence == enabled, no flag. each entry maps
                # a name to that add-on's own settings (schema defaults apply at runtime).
                # "trackers": {"object_tracker": {"distance_threshold": 0.5}},
                # "utilities": {"network_table_handler": {"network_tables_ip": "10.0.0.2"}},
                "trackers": {},
                "utilities": {},
                "frame_processors": {}
            },
        }
        self.config = json.loads(json.dumps(self.default_config))
        self.file_path = file_path

        if create and file_path is not None and not Path(file_path).exists():
            self.save()

        if file_path:
            self.load_from_file(file_path)

        self._check_config()
        self._migrate_addons()
        self._migrate_camera_configs()
        self._rebuild_camera_configs()
        try:
            self._configure_logging()
        except Exception:
            self.logger.exception("Failed to configure logging from config")

    def search_for_config(self) -> str:
        config_dir = _REPO_ROOT / "Config"
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found at {config_dir}")
        config_files = list(config_dir.rglob("*.json"))
        if not config_files:
            raise FileNotFoundError("No .json config files found in Config/")
        chosen = str(config_files[0])
        self.logger.info("Found config files: %s  ->  using %s", config_files, chosen)
        return chosen

    def _check_config(self):
        self.config.setdefault("camera_configs", {})
        self.config.setdefault("plugins", {})
        self.config["plugins"].setdefault("trackers", {})
        self.config["plugins"].setdefault("utilities", {})
        self.config["plugins"].setdefault("frame_processors", {})

    def _migrate_addons(self):
        plugins = self.config.setdefault("plugins", {})
        for addon_type in _ADDON_TYPES:
            current = plugins.get(addon_type)
            if isinstance(current, list):
                plugins[addon_type] = {
                    name: {} for name in current if isinstance(name, str)
                }
            elif not isinstance(current, dict):
                plugins[addon_type] = {}

        trackers = plugins.setdefault("trackers", {})
        utilities = plugins.setdefault("utilities", {})

        dbscan = self.config.get("dbscan")
        if isinstance(dbscan, dict) and "path_planner" in trackers:
            for legacy_key, target_key in (("epsilon", "epsilon"),
                                           ("min_samples", "min_samples")):
                if legacy_key in dbscan:
                    trackers["path_planner"].setdefault(target_key, dbscan[legacy_key])

        for legacy_key, addon_type, addon_name, target_key in _ADDON_LEGACY_FOLDS:
            if target_key is None:
                continue  # dbscan handled above, folded into path_planner
            value = self.config.get(legacy_key)
            if value is not None:
                target = {"trackers": trackers, "utilities": utilities}[addon_type]
                if addon_name in target:
                    target[addon_name].setdefault(target_key, value)

        for flag, (addon_type, addon_name) in _ADDON_LEGACY_FLAGS.items():
            if self.config.get(flag):
                {"trackers": trackers,
                 "utilities": utilities}[addon_type].setdefault(addon_name, {})

        # the health_reporter/status_reporter add-ons were merged into the
        # always-on HealthModule web module; their stale_threshold setting
        # lives at the top level as health_stale_threshold now. An explicit
        # legacy value overrides the default; migration is idempotent since
        # the old keys are removed below and never re-written.
        migrated_stale = None
        for gone_name in _MERGED_HEALTH_ADDONS:
            gone = utilities.pop(gone_name, None)
            if isinstance(gone, dict) and isinstance(
                gone.get("stale_threshold"), (int, float)
            ):
                migrated_stale = gone["stale_threshold"]
        if migrated_stale is None:
            top_stale = self.config.get("stale_threshold")
            if isinstance(top_stale, (int, float)):
                migrated_stale = top_stale
        if migrated_stale is not None:
            self.config["health_stale_threshold"] = migrated_stale

        for legacy_key, (addon_type, addon_name, target_key) in _ADDON_LEGACY_SETTINGS.items():
            value = self.config.get(legacy_key)
            if value is not None:
                target = {"trackers": trackers, "utilities": utilities}[addon_type]
                if addon_name in target:
                    target[addon_name].setdefault(target_key, value)

        for legacy_key in (
            "dbscan", "distance_threshold", "stale_threshold",
            "record_mode", "record_dir", "use_network_tables",
            "network_tables_ip",
        ):
            self.config.pop(legacy_key, None)

    def _migrate_camera_configs(self):
        cams = self.config.get("camera_configs", {})
        if not isinstance(cams, dict):
            self.config["camera_configs"] = cams = {}
        for name, cam_cfg in cams.items():
            if not isinstance(cam_cfg, dict):
                continue
            normalize_camera_entry(cam_cfg)
            _normalize_vision_model_settings(get_pipeline_settings(cam_cfg))
            if is_model_backed_pipeline(get_pipeline_name(cam_cfg)) and not isinstance(
                get_pipeline_settings(cam_cfg).get("vision_model"), dict
            ):
                self.logger.info(
                    "Camera '%s' is missing a vision_model block - adding default.",
                    name,
                )
                get_pipeline_settings(cam_cfg)["vision_model"] = default_vision_model()

    def get_default_config(self) -> dict:
        return self.default_config

    def camera_config(self, cam_name: str) -> "iSpyCameraConfig":
        cfg = self.camera_configs.get(cam_name)
        if cfg is None:
            raise KeyError(
                f"No camera config named '{cam_name}'. "
                f"Available: {list(self.camera_configs)}"
            )
        return cfg

    def _rebuild_camera_configs(self):
        cams = self.config.get("camera_configs", {})
        if not isinstance(cams, dict):
            self.config["camera_configs"] = cams = {}
        self.camera_configs = {
            name: iSpyCameraConfig(cam_cfg)
            for name, cam_cfg in cams.items()
        }

    def load_from_file(self, file_path: str):
        try:
            # utf-8-sig strips a UTF-8 BOM, which editors like Notepad add on
            # save - json.load() would otherwise fail at char 0 with a bogus
            # "Expecting value: line 1 column 1" error.
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict) and "vision_model" in data:
                raise RuntimeError(
                    f"Config at {file_path} uses the legacy top-level "
                    "'vision_model' layout, which is no longer supported - "
                    "vision model settings now live inside each camera entry. "
                    "Run 'boot -f' to start from a fresh default configuration."
                )
            self._update_config(data)
        except json.JSONDecodeError as e:
            # bad config is an error - boot never silently regenerates it.
            # first-run setup is explicitly the job of `boot -f`.
            raise RuntimeError(
                f"Config at {file_path} is invalid JSON ({e}). Fix it or run "
                "'boot -f' to start from a fresh default configuration."
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                f"Config file not found at {file_path}. Run 'boot -f' to "
                "create a fresh default configuration."
            ) from None
        finally:
            try:
                self._configure_logging()
            except Exception:
                self.logger.exception("Failed to apply logging configuration after loading file")
    def save(self, quiet=False):
        try:
            if self.file_path:
                if not quiet:
                    self.logger.info("Config saved to %s", self.file_path)
            else:
                self.logger.info("No config file path set; saving to Config/config.json")
                self.file_path = str(_REPO_ROOT / "Config" / "config.json")

            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error("Failed to save config to %s: %s", self.file_path, e)
    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_nested(self, *keys, default=None):
        val = self.config
        try:
            for key in keys:
                val = val[key]
            return val
        except (KeyError, TypeError):
            return default

    # ---------------------------------------------------------------
    # add-on config helpers
    #
    # plugins.<type> is a dict of enabled add-on name -> its settings.
    # presence == enabled; the value is the add-on's own settings dict.
    # ---------------------------------------------------------------

    def addon_entries(self, addon_type: str) -> dict:
        if addon_type not in _ADDON_TYPES:
            return {}
        entries = self.get_nested("plugins", addon_type, default={})
        if not isinstance(entries, dict):
            return {}
        return entries

    def get_addon_settings(self, addon_type: str, addon_name: str) -> dict:
        entries = self.addon_entries(addon_type)
        if addon_name not in entries:
            return None
        settings = entries[addon_name]
        return settings if isinstance(settings, dict) else {}

    def get_addon_setting(self, addon_type, addon_name, key, default=None):
        settings = self.get_addon_settings(addon_type, addon_name)
        if settings is None:
            return default
        if key in settings:
            return settings[key]
        return default

    def is_addon_enabled(self, addon_type: str, addon_name: str) -> bool:
        return self.get_addon_settings(addon_type, addon_name) is not None

    def enable_addon(self, addon_type: str, addon_name: str,
                     settings: dict | None = None, save: bool = True) -> None:
        if addon_type not in _ADDON_TYPES:
            return
        entries = self.addon_entries(addon_type)
        if addon_name not in entries:
            entries[addon_name] = {}
        if isinstance(settings, dict):
            entries[addon_name].update(settings)
        if save:
            self.save()

    def disable_addon(self, addon_type: str, addon_name: str,
                      save: bool = True) -> None:
        if addon_type not in _ADDON_TYPES:
            return
        entries = self.addon_entries(addon_type)
        if addon_name in entries:
            del entries[addon_name]
        if save:
            self.save()

    def set_addon_settings(self, addon_type: str, addon_name: str,
                           settings: dict, save: bool = True) -> None:
        if addon_type not in _ADDON_TYPES:
            return
        entries = self.addon_entries(addon_type)
        if addon_name not in entries:
            return
        entries[addon_name] = settings if isinstance(settings, dict) else {}
        if save:
            self.save()

    def update_addon_settings(self, addon_type: str, addon_name: str,
                              settings: dict, save: bool = True) -> None:
        if addon_type not in _ADDON_TYPES:
            return
        entries = self.addon_entries(addon_type)
        if addon_name not in entries:
            return
        current = entries[addon_name]
        if not isinstance(current, dict):
            current = {}
        current.update(settings if isinstance(settings, dict) else {})
        entries[addon_name] = current
        if save:
            self.save()

    def set(self, *keys_and_value):
        if len(keys_and_value) < 2:
            return
        *keys, value = keys_and_value
        target = self.config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        if keys[-1] == "camera_configs":
            # whole-dict replace orphans the wrapper views - rebuild em
            self._rebuild_camera_configs()

    def _update_config(self, data: dict, current_dict: dict = None):
        if current_dict is None:
            current_dict = self.config
        if isinstance(data, dict) and "vision_model" in data and current_dict is self.config:
            # model settings live per-cam under pipeline.settings now - a top-level
            # key is either a stale UI payload or a legacy config.
            raise ValueError(
                "A top-level 'vision_model' key is no longer supported - "
                "model settings are configured per camera under "
                "pipeline.settings.vision_model (Camera Settings page)."
            )
        for key, value in data.items():
            if key == "camera_configs":
                if not isinstance(value, dict):
                    continue
                if current_dict is not self.config:
                    # _compare() merges on a scratch copy - keep the raw entries
                    # so the diff sees what was actually sent.
                    current_dict[key] = value
                else:
                    self.config[key] = json.loads(json.dumps(value))
                    ensure_camera_entries_ready(self.config[key])
                    # fresh deep copy orphaned the wrappers - rebuild em
                    self._rebuild_camera_configs()
            elif (
                isinstance(value, dict)
                and key in current_dict
                and isinstance(current_dict[key], dict)
            ):
                self._update_config(value, current_dict[key])
            else:
                current_dict[key] = value

    def _configure_logging(self):
        level_str = self.config.get("log_level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.NOTSET)  # let children decide

        fmt = logging.Formatter(
            "%(asctime)s [iSpy] %(levelname)s:%(name)s: %(message)s"
        )

        class _iSpyFilter(logging.Filter):
            def filter(self, record):
                return record.name.startswith("iSpy")

        sh = logging.StreamHandler()
        sh.setLevel(logging.NOTSET)
        sh.setFormatter(fmt)
        sh.addFilter(_iSpyFilter())
        root.addHandler(sh)

        log_file = self.config.get("log_file")
        if log_file:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = _REPO_ROOT / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            fh = logging.FileHandler(log_path, mode="a")
            fh.setLevel(logging.NOTSET)
            fh.setFormatter(fmt)
            fh.addFilter(_iSpyFilter())
            root.addHandler(fh)

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self.get_nested(*args)
        return self.get(args)

    def __call__(self, *keys):
        return self.get_nested(*keys)

    def __getattr__(self, item: str):
        if item.startswith("_") or item in {
            "config",
            "logger",
            "default_config",
            "camera_configs",
        }:
            raise AttributeError(item)
        val = self.get(item)
        if val is None:
            raise AttributeError(f"No config attribute or key named '{item}'")
        return val


class iSpyCameraConfig:
    DEFAULTS = {
        "name": "default",
        "x": 0,
        "y": 0,
        "z": 0,
        "height": 0,
        "pitch": 0,
        "yaw": 0,
        "grayscale": False,
        "brightness": 0,
        "contrast": 0,
        "saturation": 0,
        "white_balance": 0,
        "tint": 0,
        "gamma": 1.0,
        "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
        "source": "/dev/video0",
        "device_id": None,
        "subsystem": "field",
        "pipeline": {"name": "object_detection", "settings": {}},
    }

    def __init__(self, config_dict: dict = None):
        self.data = json.loads(json.dumps(self.DEFAULTS))
        if config_dict:
            self.data.update(config_dict)

    # ---------------------------------------------------------------
    # pipeline (pipeline: {name, settings}) accessors. legacy flat entries
    # (bare-string pipeline + settings spread on the cam) migrate lazily on
    # first touch.
    # ---------------------------------------------------------------

    def pipeline_entry(self) -> dict:
        p = self.data.get("pipeline")
        if isinstance(p, dict):
            if not isinstance(p.get("settings"), dict):
                p["settings"] = {}
            return p
        settings = {
            k: v for k, v in self.data.items()
            if k not in _CAMERA_CORE_KEYS and k != "pipeline"
        }
        for k in settings:
            self.data.pop(k)
        entry = {"name": get_pipeline_name(self.data), "settings": settings}
        self.data["pipeline"] = entry
        return entry

    def pipeline_name(self) -> str:
        return get_pipeline_name(self.data)

    def pipeline_settings(self) -> dict:
        return self.pipeline_entry()["settings"]

    def get_pipeline_setting(self, key, default=None):
        settings = self.pipeline_entry()["settings"]
        if key in settings:
            return settings[key]
        # legacy flat layout - the setting may still live directly on the cam entry
        if key not in _CAMERA_CORE_KEYS and key in self.data:
            return self.data[key]
        return default

    def set_pipeline_setting(self, key, value):
        self.pipeline_entry()["settings"][key] = value

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data


class iSpyAddonConfig:
    def __init__(self, settings: dict | None = None, defaults: dict | None = None):
        self.data: dict = {}
        if isinstance(settings, dict):
            self.data.update(settings)
        if isinstance(defaults, dict):
            for key, value in defaults.items():
                self.data.setdefault(key, value)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def get_nested(self, *keys, default=None):
        val = self.data
        try:
            for key in keys:
                val = val[key]
            return val
        except (KeyError, TypeError):
            return default

    def set(self, key, value):
        self.data[key] = value

    def setdefault(self, key, value):
        return self.data.setdefault(key, value)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.data))

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data