import json
import logging
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path.cwd()

_MODEL_BACKED_PIPELINES = ("object_detection",)


def default_vision_model() -> dict:
    """Per-camera vision_model block for model-backed pipelines.

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
            "auto_opt": False,
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
                    "pipeline": "object_detection",
                    "vision_model": {
                        # output.* and input.* vision model fields are auto-detected
                        # from the model's _metadata.yaml sidecar file. You do not need to set them.
                        # Override here only if you know the metadata is wrong.
                        "file_path": "YoloModels/pytorch/_default_pose.pt",
                        "source_pt": "YoloModels/pytorch/_default_pose.pt",
                        "min_conf": 0.5,
                        # Optimization (quantization/RKNN builds) is opt-in per
                        # camera via the Optimize toggle in the web UI - a fresh
                        # install should boot instantly, not kick off a
                        # multi-minute backend build.
                        "quantized": False,
                    },
                    # Optional PnP for pose (translation stored on Box; rotation stored as roll/pitch/yaw on Box).
                    # Enable to get 3D position + orientation from 2D keypoints, and to render a 3D human
                    # skeleton in the web viewer (when keypoints_3d is present on the Object).
                    # "pnp": {
                    #     # Canonical COCO 17-keypoint skeleton (~1.8 m tall, origin at mid-hip).
                    #     # Indices match the model's kpt_shape ordering (typically COCO format).
                    #     "object_points": [
                    #         [0.0, 1.0, 0.1],       # 0  nose
                    #         [-0.03, 0.95, 0.1],    # 1  left_eye
                    #         [0.03, 0.95, 0.1],     # 2  right_eye
                    #         [-0.08, 0.93, 0.0],    # 3  left_ear
                    #         [0.08, 0.93, 0.0],     # 4  right_ear
                    #         [-0.2, 0.8, 0.0],      # 5  left_shoulder
                    #         [0.2, 0.8, 0.0],       # 6  right_shoulder
                    #         [-0.35, 0.55, -0.05],  # 7  left_elbow
                    #         [0.35, 0.55, -0.05],   # 8  right_elbow
                    #         [-0.4, 0.3, -0.1],     # 9  left_wrist
                    #         [0.4, 0.3, -0.1],      # 10 right_wrist
                    #         [-0.15, 0.0, 0.0],     # 11 left_hip
                    #         [0.15, 0.0, 0.0],      # 12 right_hip
                    #         [-0.15, -0.45, 0.0],   # 13 left_knee
                    #         [0.15, -0.45, 0.0],    # 14 right_knee
                    #         [-0.15, -0.9, 0.0],    # 15 left_ankle
                    #         [0.15, -0.9, 0.0],     # 16 right_ankle
                    #     ],
                    #     "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    #     "dist_coeffs": [0, 0, 0, 0, 0],
                    #     "min_keypoint_conf": 0.5,
                    #     # "mode": "flexible",  # "flexible" (deformable: keep detected 2D keypoint shape
                    #     #                       #   inflated to 3D at the PnP depth) or "rigid"
                    #     #                       #   (rigid object: keypoints = fitted model, full
                    #     #                       #   xyz + roll/pitch/yaw).  Solver is auto-selected.
                    # },
                    "yaw": 0,
                    "pitch": 0,
                    "height": 1.0,
                    "x": 0,
                    "y": 0,
                    "grayscale": False,
                    "subsystem": "field",
                    "calibration": {
                        "distance": 0.0,
                        "game_piece_size": 0.0,
                        "size": 0,
                        "fov": 0,
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
        self.camera_configs: dict[str, iSpyCameraConfig] = {
            name: iSpyCameraConfig(cam_cfg)
            for name, cam_cfg in self.config["camera_configs"].items()
        }
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
        """Bring legacy camera entries up to date with the per-camera
        vision_model restructure.

        Model-backed pipelines (object_detection) require a vision_model dict
        in every camera entry; before the restructure that block lived at the
        top level of the config, so existing configs (and cameras added by
        older web builds) boot with it missing and crash at pipeline
        construction. Inject a default so those configs keep working."""
        for name, cam_cfg in self.config.get("camera_configs", {}).items():
            if not isinstance(cam_cfg, dict):
                continue
            if is_model_backed_pipeline(cam_cfg.get("pipeline")) and not isinstance(
                cam_cfg.get("vision_model"), dict
            ):
                self.logger.info(
                    "Camera '%s' is missing a vision_model block - adding default.",
                    name,
                )
                cam_cfg["vision_model"] = default_vision_model()

        # required_trackers = ["path_planner"]
        # missing = False
        # for tracker in required_trackers:
        #     if tracker not in self.config["plugins"]["trackers"]:
        #         self.logger.warning(
        #             "%s not in trackers list. Re-adding required tracker.", tracker
        #         )
        #         self.config["plugins"]["trackers"].append(tracker)
        #         missing = True

        # if missing:
        #     self.logger.info("Required trackers missing. Saving updated config.")
        #     self.save()

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

    def _update_config(self, data: dict, current_dict: dict = None):
        if current_dict is None:
            current_dict = self.config
        for key, value in data.items():
            if key == "camera_configs":
                current_dict[key] = value
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
    }

    def __init__(self, config_dict: dict = None):
        self.data = json.loads(json.dumps(self.DEFAULTS))
        if config_dict:
            self.data.update(config_dict)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data