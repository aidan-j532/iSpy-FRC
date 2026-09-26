from iSpy.plugins.bases import TrackerBase


class YourTracker(TrackerBase):
    plugin_name = "example/example_tracker"
    template = True

    @classmethod
    def config_schema(cls) -> dict:
        # settings the web ui renders an editor for; omitted keys get defaults
        return {
            "count_start": {
                "type": "number",
                "label": "Start Count",
                "hint": "Example setting - this tracker just counts updates.",
                "default": 0,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is this plugin's settings view, defaults already merged
        self.count = int(self.config.get("count_start", 0))

    def update(self, detections, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        self.count += 1
        return detections  # or modify and return
