import json
import logging
import subprocess
from pathlib import Path

_BOOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BOOT_DIR.parents[1] # two levels up

# Minimal basic config to ensure early logs appear; real level/handlers configured from config file
logging.basicConfig(level=logging.WARNING)

class VisionCoreConfig:
    def __init__(self, file_path: str = None):
        self.logger = logging.getLogger(__name__)

        self.default_config = {
            "unit": "meter",
            "dbscan": {"elipson": 0, "min_samples": 0},
            "distance_threshold": 0.5,
            "network_tables_ip": "10.22.7.2",
            "use_network_tables": True,
            "app_mode": True,
            "debug_mode": False,
            "record_mode": True,
            "stale_threshold": 1.0,
            "log_level": "INFO",
            "auto_opt": True,
            "log_file": "Outputs/log.txt",
            "metrics": False,
            "camera_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "dist_coeffs": [0, 0, 0, 0, 0],
            "camera_configs": {
                "default": {
                    "name": "default",
                    "x": 0, "y": 0, "height": 0, "pitch": 0, "yaw": 0,
                    "grayscale": False,
                    "fps_cap": -1,
                    "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
                    "source": "/dev/video0",
                    "subsystem": "field",
                    "pipeline": "object_detection",
                },
            },
            "vision_model": {
                "quantized": False,
                "file_path": "YoloModels/pytorch/nano/dummy.pt",
                "input_size": [640, 640],
                "min_conf": 0.7
            },
            "vision_modules": ["object_detection"],
            "trackers": ["fuel", "path_planner"],
            "utilities": ["network_table", "video_recorder"],
        }
        self.config = json.loads(json.dumps(self.default_config))  # deep copy
        self.file_path = file_path

        if file_path:
            self.load_from_file(file_path)

        self.camera_configs: dict[str, VisionCoreCameraConfig] = {
            name: VisionCoreCameraConfig(cam_cfg)
            for name, cam_cfg in self.config["camera_configs"].items()
        }

        self._check_config()
        # Apply logging configuration after config is loaded
        try:
            self._configure_logging()
        except Exception:
            # Never let logging configuration break initialization
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
        if self.config == self.default_config:
            self.logger.warning(
                "Using default configuration. Load a config file for proper operation."
            )
        else:
            if self.config.get("vision_model") and self.config.get("april_tag"):
                self.logger.warning(
                    "Both vision_model and april_tag configs present — ensure this is intentional."
                )

    def get_default_config(self) -> dict:
        return self.default_config

    def camera_config(self, cam_name: str) -> "VisionCoreCameraConfig":
        cfg = self.camera_configs.get(cam_name)
        if cfg is None:
            raise KeyError(f"No camera config named '{cam_name}'. "
                           f"Available: {list(self.camera_configs)}")
        return cfg

    def load_from_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._update_config(data)
        except Exception as e:
            self.logger.warning("Failed to load config from %s: %s, searching for config", file_path, e)
            config_file = self.search_for_config()

            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                self._update_config(data)
            except Exception as e:
                self.logger.warning("Failed to find and load config from %s: %s", config_file, e)
                self.logger.info("Using default configuration file: config.json")
        finally:
            # Reconfigure logging in case log settings changed from the file
            try:
                self._configure_logging()
            except Exception:
                self.logger.exception("Failed to apply logging configuration after loading file")

    def save(self):
        # Overwrite json file with current config
        if not self.file_path:
            self.logger.warning("No config file path set; saving to Config/config.json")
            self.file_path = str(_REPO_ROOT / "Config" / "config.json")
        
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.config, f, indent=4)
            self.logger.info("Config saved to %s", self.file_path)
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
                current_dict[key] = value  # always replace entirely
            elif (
                isinstance(value, dict)
                and key in current_dict
                and isinstance(current_dict[key], dict)
            ):
                self._update_config(value, current_dict[key])
            else:
                current_dict[key] = value

    def _configure_logging(self):
        """Configure root logging according to the loaded config.

        - Honor `log_level` (string like INFO, DEBUG)
        - Optionally add a file handler for `log_file` (path relative to repo root)
        """
        # Map level string to logging level
        level_str = self.config.get("log_level", "INFO")
        try:
            level = getattr(logging, str(level_str).upper(), logging.INFO)
        except Exception:
            level = logging.INFO

        root = logging.getLogger()
        # Remove existing handlers to avoid duplicate logs when reconfiguring
        for h in list(root.handlers):
            root.removeHandler(h)

        root.setLevel(level)

        fmt = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")

        # Stream handler
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        log_file = self.config.get("log_file")
        if log_file:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = _REPO_ROOT / log_path
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_path, mode="a")
                fh.setLevel(level)
                fh.setFormatter(fmt)
                root.addHandler(fh)
            except Exception:
                # If file handler can't be created, fall back to console only
                self.logger.exception("Unable to create log file handler at %s", log_path)

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self.get_nested(*args)
        return self.get(args)

    def __call__(self, *keys):
        return self.get_nested(*keys)

    def __getattr__(self, item: str):
        if item.startswith("_") or item in {"config", "logger", "default_config", "camera_configs"}:
            raise AttributeError(item)
        val = self.get(item)
        if val is None:
            raise AttributeError(f"No config attribute or key named '{item}'")
        return val

class VisionCoreCameraConfig:
    DEFAULTS = {
        "name":       "default",
        "x":          0,
        "y":          0,
        "height":     0,
        "pitch":      0,
        "yaw":        0,
        "grayscale":  False,
        "fps_cap":    30,
        "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
        "source":     "/dev/video0",
        "subsystem":  "field",
    }

    def __init__(self, config_dict: dict = None):
        import json
        self.data = json.loads(json.dumps(self.DEFAULTS)) # deep copy
        if config_dict:
            self.data.update(config_dict)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data