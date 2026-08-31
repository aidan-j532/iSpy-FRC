import logging
import json
from pathlib import Path
from iSpy.plugins.bases import UtilityBase


class DatasetManager(UtilityBase):
    plugin_name = "dataset_manager"

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        self.dataset_dir = Path.cwd() / "Dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        pass

    def update(self, frame_data: dict):
        """Store frame data with timestamp and optional annotations."""
        key = self.declared_output_key()
        if not key:
            return False
        addon_data = frame_data.setdefault("addon_data", {})
        entry = {
            "timestamp": frame_data.get("timestamp"),
            "data": addon_data,
        }
        dataset_file = self.dataset_dir / f"{key}_{entry['timestamp']}.json"
        try:
            dataset_file.write_text(json.dumps(entry, indent=2))
            self.logger.debug(f"Saved dataset entry to {dataset_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            return False

    def stop(self):
        pass
