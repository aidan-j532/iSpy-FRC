from abc import ABC, abstractmethod

class TrackerBase:
    """Base for tracker plugin"""
    def __init__(self, config):
        pass

    def update(self, fuel_list, robot_x, robot_y, robot_yaw):
        return fuel_list

    def stop(self):
        pass


class UtilityBase:
    """
    Base for utility plugin
    context keys available in __init__:
        config, camera_app, cameras, flask_app

    frame_data keys available in update():
        fuel_list, frame, fps, loop_s, vision_s,
        camera_lag_s, detections, cameras
    """
    def __init__(self, context: dict):
        pass

    def update(self, frame_data: dict):
        pass

    def get_robot_pose(self):
        """Override in network utility to provide pose. Default returns None."""
        return None

    def stop(self):
        pass

class VisionBase(ABC):

    plugin_name = "base"

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def destroy(self):
        pass