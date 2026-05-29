from iSpy.plugins.bases import TrackerBase

class YourTracker(TrackerBase):
    plugin_name = "example_tracker"
    def __init__(self, config):
        super().__init__(config)
        self.count = 0

    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        self.count += 1
        return fuel_list  # or modify and return