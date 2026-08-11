import json
import logging
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path.cwd()

_MODEL_BACKED_PIPELINES = ("object_detection",)

# Keys that belong to the camera itself (mount/capture/calibration) and stay
# at the camera level. Everything else on a camera entry is pipeline
# configuration and lives under ``pipeline.settings``.
_CAMERA_CORE_KEYS = {
    "name", "source", "device_id", "subsystem", "grayscale",
    "x", "y", "z", "height", "yaw", "pitch", "calibration",
    "auto_brightness", "exposure_time", "gain", "fps_cap",
    "csi", "path",
}

# User-facing model settings. These belong at pipeline.settings level, never
# duplicated inside pipeline.settings.vision_model - older builds merged them
# into the persisted vision_model block, so every camera save (or optimization
# activation) wrote each setting twice. The vision_model block holds model
# identity only (file_path, source_pt, input_size, device, ...).
_VISION_MODEL_SETTINGS_KEYS = (
    "min_conf", "quantize", "quantization_dataset", "optimize",
    "target_format",
)

# Legacy names for the same settings. 'optimize' was 'auto_opt' and 'quantize'
# was 'quantized' in older configs; the legacy key is dropped whenever the
# canonical one is present so old configs don't store both.
_LEGACY_SETTING_ALIASES = {
    "auto_opt": "optimize",
    "quantized": "quantize",
}


def _normalize_vision_model_settings(settings: dict) -> None:
    """Drop user-facing settings duplicated inside the vision_model block.

    When a key exists at pipeline.settings level AND inside vision_model, the
    settings copy wins and the vision_model copy is dropped, so the config
    never stores the same setting twice. Keys that live ONLY inside
    vision_model are left untouched - they are valid legacy fallbacks (the
    web UI and pipeline read through both locations). Idempotent."""
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
    """Pipeline-settings vision_model block for model-backed pipelines.

    Used when a config predates the per-camera vision_model restructure (or a
    camera was added without one). User-uploaded models (anything not prefixed
    with ``_default``) take priority; otherwise fall back to the same bundled
    pose model the default config ships with."""
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
    """Name of the pipeline a camera entry uses: ``"object_detection"`` for
    BOTH the legacy flat ``pipeline: "object_detection"`` layout and the new
    nested ``pipeline: {"name": ...}`` layout."""
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
    """Settings dict for a camera entry. For the legacy flat layout this
    coalesces every non-camera key into one settings dict; for the nested
    layout it returns ``pipeline.settings`` directly."""
    if not isinstance(cam_entry, dict):
        return {}
    p = cam_entry.get("pipeline")
    if isinstance(p, dict) and isinstance(p.get("settings"), dict):
        return p["settings"]
    return {k: v for k, v in cam_entry.items()
            if k not in _CAMERA_CORE_KEYS and k != "pipeline"}


def normalize_camera_entry(cam_entry: dict) -> dict:
    """Migrate a legacy FLAT camera entry into the nested pipeline layout:

        legacy:  {"pipeline": "object_detection", "min_conf": 0.7, ...}
        new:     {"pipeline": {"name": "object_detection",
                               "settings": {"min_conf": 0.7, ...}}, ...}

    Camera-level keys (mount, calibration, capture, source...) stay where they
    are; every other key is folded into ``pipeline.settings``. Idempotent -
    already-nested entries are left alone."""
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
    # No pipeline key at all (e.g. raw JSON editor) - tag with the default.
    settings = {
        k: v for k, v in cam_entry.items()
        if k not in _CAMERA_CORE_KEYS and k != "pipeline"
    }
    for k in settings:
        cam_entry.pop(k)
    cam_entry["pipeline"] = {"name": "object_detection", "settings": settings}
    return cam_entry


def ensure_camera_entries_ready(camera_configs: dict) -> None:
    """Normalize every camera entry into the nested pipeline layout and make
    sure model-backed pipelines have a per-camera vision_model block."""
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
            "unit": "meter",
            "debug_mode": True,
            "dbscan": {"epsilon": 0.3, "min_samples": 3},
            "distance_threshold": 0.5,
            "stale_threshold": 1.0,
            "record_mode": True,
            "record_dir": "VideoRecordings",
            "frame_sync": False,
            "optimize": False,
            "log_level": "INFO",
            "log_file": "Outputs/log.txt",
            "use_network_tables": False,
            "network_tables_ip": "10.0.0.2",
            "metrics": True,
            "app_mode": True,
            "max_fps": 0,
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
                    # Camera mount offsets are kept on the camera entry itself;
                    # everything that configures the *pipeline* lives under
                    # pipeline.settings.
                    "yaw": 0,
                    "pitch": 0,
                    "height": 1.0,
                    "x": 0,
                    "y": 0,
                    "pipeline": {
                        "name": "object_detection",
                        "settings": {
                            "vision_model": {
                                # output.* and input.* vision model fields are
                                # auto-detected from the model's _metadata.yaml
                                # sidecar file. You do not need to set them.
                                # Override here only if you know the metadata
                                # is wrong.
                                "file_path": "YoloModels/pytorch/_default_pose.pt",
                                "source_pt": "YoloModels/pytorch/_default_pose.pt",
                                "min_conf": 0.5,
                                # Optimization (quantization/RKNN builds) is
                                # opt-in per camera via the Optimize toggle in
                                # the web UI - a fresh install should boot
                                # instantly, not kick off a multi-minute
                                # backend build.
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
                # "trackers": ["object_tracker", "path_planner"],
                # "utilities": ["video_recorder", "health_reporter"],
                "trackers": [],
                "utilities": [],
                "frame_processors": []
            },
        }
        self.config = json.loads(json.dumps(self.default_config))
        self.file_path = file_path

        if create and file_path is not None and not Path(file_path).exists():
            self.save()

        if file_path:
            self.load_from_file(file_path)

        self._check_config()
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
        self.config["plugins"].setdefault("trackers", [])
        self.config["plugins"].setdefault("utilities", [])
        self.config["plugins"].setdefault("frame_processors", [])

    def _migrate_camera_configs(self):
        """Bring legacy camera entries up to date with the nested pipeline
        layout (pipeline: {name, settings}).

        Legacy entries stored everything flat on the camera entry (e.g.
        ``"pipeline": "object_detection"`` plus ``min_conf``, ``tag_size_inches``
        or ``vision_model`` at camera level). Every non-camera key is folded
        into ``pipeline.settings``. Model-backed pipelines (object_detection)
        additionally require a vision_model dict in their settings or the
        pipeline crashes at construction - inject a default so those configs
        keep working."""
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
        """Recreate the iSpyCameraConfig wrapper views from the current
        raw entries, keeping them in sync with self.config["camera_configs"].

        Web saves replace the whole camera_configs dict with a fresh deep
        copy (Settings._post -> _update_config, CamerasModule._update_camera
        -> set), which silently orphans the wrappers built in __init__ - the
        shared-reference link breaks on the first save and every later
        settings read through them returns stale values. The wrappers share
        their nested pipeline dicts with the raw entries, so in-place edits
        made by ensure_camera_entries_ready / _activate_optimized_model stay
        visible."""
        cams = self.config.get("camera_configs", {})
        if not isinstance(cams, dict):
            self.config["camera_configs"] = cams = {}
        self.camera_configs = {
            name: iSpyCameraConfig(cam_cfg)
            for name, cam_cfg in cams.items()
        }

    def load_from_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
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
            # Invalid config is an error - boot never silently regenerates it.
            # First-run initialization is explicitly the job of `boot -f`.
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
            # A whole-dict replacement orphans the wrapper views - rebuild
            # them so config.camera_configs stays a live view of the config.
            self._rebuild_camera_configs()

    def _update_config(self, data: dict, current_dict: dict = None):
        if current_dict is None:
            current_dict = self.config
        if isinstance(data, dict) and "vision_model" in data and current_dict is self.config:
            # Model settings live inside each camera's pipeline.settings now -
            # a top-level key is either a stale UI payload or a legacy config.
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
                    # _compare() runs the same merge on a scratch copy - keep
                    # the raw entries so the diff sees what was actually sent.
                    current_dict[key] = value
                else:
                    self.config[key] = json.loads(json.dumps(value))
                    ensure_camera_entries_ready(self.config[key])
                    # Every camera entry was replaced with a fresh deep copy -
                    # rebuild the wrapper views so they don't keep pointing at
                    # the orphaned pre-save dicts.
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
        "auto_brightness": True,
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

    # ------------------------------------------------------------------
    # Pipeline (nested pipeline: {name, settings}) accessors. Legacy flat
    # entries (pipeline as a bare string + settings spread on the camera)
    # are migrated lazily, in place, the first time they are touched.
    # ------------------------------------------------------------------

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
        # Legacy flat layout: the setting may still live directly on the
        # camera entry (configs constructed without a nested pipeline block).
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