import logging
from iSpy.plugins.bases import UtilityBase


class DatasetManager(UtilityBase):
    plugin_name = "dataset_manager"

    def __init__(self, context: dict):
        self.logger = logging.getLogger(__name__)

    def stop(self):
        pass
