import ntcore
import logging
import dataclasses
import time
import wpiutil.wpistruct
from wpimath.geometry import Pose2d
from iSpy.plugins.bases import UtilityBase
from iSpy.vision.Object import Object


@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class NetworkTableHandler(UtilityBase):
    plugin_name = "network_table_handler"

    def __init__(self, context: dict):
        config = context["config"]
        self.logger = logging.getLogger(__name__)
        self._enabled = config.get("use_network_tables", False)

        if not self._enabled:
            return

        ip = config.get("network_tables_ip", "10.0.0.2")
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.setServer(ip)
        self.inst.startClient4("iSpy")

        for i in range(15):
            if self.inst.isConnected():
                break
            self.logger.warning(
                "NetworkTables not connected, retrying... (%d/15)", i + 1
            )
            time.sleep(1)
        else:
            self.logger.error("NetworkTables could not connect after 15s.")

        self._subscribers: dict = {}
        self._tables: dict = {}

    def isConnected(self) -> bool:
        return self._enabled and self.inst.isConnected()

    def get_robot_pose(self) -> Pose2d:
        if not self._enabled or not self.isConnected():
            return Pose2d()
        try:
            sub_key = "AdvantageKit/RealOutputs/Odometry/Robot"
            if sub_key not in self._subscribers:
                table = self._get_table("AdvantageKit/RealOutputs/Odometry")
                self._subscribers[sub_key] = table.getStructTopic(
                    "Robot", Pose2d
                ).subscribe(Pose2d())
            return self._subscribers[sub_key].get()
        except Exception as e:
            self.logger.error("Failed to get robot pose: %s", e)
            return Pose2d()

    def update(self, frame_data: dict):
        if not self._enabled or not self.isConnected():
            return
        fuel_list = frame_data.get("fuel_list", [])
        fps = frame_data.get("fps", 0)
        detections = frame_data.get("detections", 0)
        lag = frame_data.get("camera_lag_s", 0)
        cameras = frame_data.get("cameras", [])

        self._send_fuel_list(fuel_list)
        self._send_data(fps, "fps", "VisionData")
        self._send_data(detections, "num_detections", "VisionData")
        self._send_data(lag, "camera_lag", "VisionData")

        for cam in cameras:
            hopper = cam.get_data_for_subsystem("hopper")
            if hopper is not None:
                self._send_boolean(hopper, "hopper_sees_object", "VisionData")

    def _get_table(self, table_name: str):
        if table_name not in self._tables:
            self._tables[table_name] = self.inst.getTable(table_name)
        return self._tables[table_name]

    def _send_fuel_list(self, fuels: list):
        try:
            table = self._get_table("VisionData")
            pub_key = "pub/VisionData/vision_data"
            structs = [
                FuelStruct(
                    x=float(f.get_position_normally()[0]),
                    y=float(f.get_position_normally()[1]),
                    roll=f.roll,
                    pitch=f.pitch,
                    yaw=f.yaw,
                )
                for f in fuels
            ]
            if pub_key not in self._subscribers:
                self._subscribers[pub_key] = table.getStructArrayTopic(
                    "vision_data", FuelStruct
                ).publish()
            self._subscribers[pub_key].set(structs)
            table.putNumber("timestamp_ms", time.time() * 1000)
            self.inst.flush()
        except Exception as e:
            self.logger.error("Failed to send fuel list: %s", e)

    def _send_data(self, value, data_name: str, table_name: str):
        try:
            table = self._get_table(table_name)
            pub_key = f"pub/{table_name}/{data_name}"
            if pub_key not in self._subscribers:
                if isinstance(value, bool):
                    pub = table.getBooleanTopic(data_name).publish()
                elif isinstance(value, (int, float)):
                    pub = table.getDoubleTopic(data_name).publish()
                elif isinstance(value, str):
                    pub = table.getStringTopic(data_name).publish()
                else:
                    return
                self._subscribers[pub_key] = pub
            self._subscribers[pub_key].set(value)
            self.inst.flush()
        except Exception as e:
            self.logger.error("Failed to send data %s: %s", data_name, e)

    def _send_boolean(self, value: bool, data_name: str, table_name: str):
        self._send_data(value, data_name, table_name)

    def stop(self):
        pass
