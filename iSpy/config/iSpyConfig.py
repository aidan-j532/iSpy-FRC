import json
import logging
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path.cwd()


class iSpyConfig:
    def __init__(self, file_path: str = None, create: bool = True):
        self.logger = logging.getLogger(__name__)

        self.default_config = {
            "vision_model": {
                "file_path": "YoloModels/pytorch/_default_pose.pt",
                "source_pt": "YoloModels/pytorch/_default_pose.pt",
                "input_size": [640, 640],
                "min_conf": 0.5,
                "margin": 10,
                "task": "detect",
                "num_classes": 1,
                "output": {
                    # "raw" = decode + optional software NMS; "hardware_nms" = baked NMS.
                    "format": "raw",
                    # "anchors_first" (N×D) or "features_first" (D×N, transposed export).
                    "layout": "features_first",
                    # Box encoding in raw tensor: "cxcywh" or "xyxy".
                    "box_format": "cxcywh",
                    # "multi_class" (per-class scores) or "objectness" (1 score, num_classes=1).
                    "score_mode": "multi_class",
                    # If true, apply sigmoid to score columns before threshold/NMS.
                    "scores_are_logits": False,
                    # Software NMS when format is "raw" (ignored for hardware_nms).
                    "apply_software_nms": True,
                    "nms_iou": 0.45,
                    # Dequantization: "none", "int8", or "uint8" (+ quant_scale if not none).
                    "quantization": "none",
                    # Pose-only (required when task is "pose"):
                    # "num_keypoints": 17,
                    # "keypoint_dims": 3,
                    # "keypoint_scores_are_logits": False,
                },
                "input": {
                    # "nhwc" (RKNN/TFLite) or "nchw" (typical ONNX export).
                    "layout": "nchw",
                    # "uint8" or "float32".
                    "dtype": "float32",
                    # Letterbox to input_size with pad_value (RKNN-style).
                    "letterbox": True,
                    "pad_value": 114,
                    # Divide by scale when true (common for float32 ONNX).
                    "normalize": True,
                    "scale": 255.0,  # required when normalize is true
                },
                # Optional PnP for pose (translation stored on Box; rotation not stored):
                # "pnp": {
                #     "object_points": [[0, 0, 0], ...],
                #     "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                #     "dist_coeffs": [0, 0, 0, 0, 0],
                #     "min_keypoint_conf": 0.5,
                # },
            },
            "num_gpus": "auto",
            "device": 0,
            "unit": "meter",
            "debug_mode": True,
            "dbscan": {"epsilon": 0.3, "min_samples": 3},
            "distance_threshold": 0.5,
            "stale_threshold": 1.0,
            "record_mode": True,
            "record_dir": "VideoRecordings",
            "auto_opt": True,
            "log_level": "INFO",
            "log_file": "Outputs/log.txt",
            "use_network_tables": False,
            "network_tables_ip": "10.0.0.2",
            "metrics": True,
            "app_mode": True,
            "camera_configs": {
                "default_cam": {
                    "name": "default_cam",
                    "source": 0,
                    "pipeline": "object_detection",
                    "fps_cap": 30,
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
                "trackers": ["object_tracker", "path_planner"],
                "utilities": ["video_recorder", "health_reporter"],
            },
        }
        self.config = json.loads(json.dumps(self.default_config))
        self.file_path = file_path

        if create and file_path is not None and not Path(file_path).exists():
            self.save()

        if file_path:
            self.load_from_file(file_path)

        self.camera_configs: dict[str, iSpyCameraConfig] = {
            name: iSpyCameraConfig(cam_cfg)
            for name, cam_cfg in self.config["camera_configs"].items()
        }

        self._check_config()
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
        if self.config.get("vision_model") and self.config.get("april_tag"):
            self.logger.warning(
                "Both vision_model and april_tag configs present - ensure this is intentional."
            )

        self.config.setdefault("plugins", {})
        self.config["plugins"].setdefault("trackers", [])
        self.config["plugins"].setdefault("utilities", [])

        required_trackers = ["path_planner"]
        missing = False
        for tracker in required_trackers:
            if tracker not in self.config["plugins"]["trackers"]:
                self.logger.warning(
                    "%s not in trackers list. Re-adding required tracker.", tracker
                )
                self.config["plugins"]["trackers"].append(tracker)
                missing = True

        if missing:
            self.logger.info("Required trackers missing. Saving updated config.")
            self.save()

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
            self._update_config(data)
        except Exception as e:
            self.logger.warning(
                "Failed to load config from %s: %s, searching for config", file_path, e
            )
            try:
                config_file = self.search_for_config()
                with open(config_file, "r") as f:
                    data = json.load(f)
                self._update_config(data)
            except Exception as e2:
                self.logger.warning(
                    "Failed to find and load config: %s. Using defaults.", e2
                )
        finally:
            try:
                self._configure_logging()
            except Exception:
                self.logger.exception(
                    "Failed to apply logging configuration after loading file"
                )

    def save(self, quiet=False):
        try:
            if self.file_path:
                if not quiet:
                    self.logger.info("Config saved to %s", self.file_path)
            else:
                self.logger.info(
                    "No config file path set; saving to Config/config.json"
                )
                self.file_path = str(_REPO_ROOT / "Config" / "config.json")
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
        "height": 0,
        "pitch": 0,
        "yaw": 0,
        "grayscale": False,
        "fps_cap": 30,
        "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
        "source": "/dev/video0",
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