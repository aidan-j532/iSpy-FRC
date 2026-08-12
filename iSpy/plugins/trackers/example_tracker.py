from iSpy.plugins.bases import TrackerBase

class YourTracker(TrackerBase):
    plugin_name = "example_tracker"

    @classmethod
    def config_schema(cls) -> dict:
        # Declare your add-on's settings so the web UI can render an editor
        # and the loader can apply defaults. Omitted keys get their default
        # at runtime, so a config entry of {} "just works".
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
        # self.config is an iSpyAddonConfig view of YOUR add-on's settings
        # (schema defaults already merged in). Presence in the config ==
        # enabled - there is no "enabled" flag.
        self.count = int(self.config.get("count_start", 0))

    def update(self, fuel_list, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        self.count += 1
        return fuel_list  # or modify and return
