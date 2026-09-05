import json
import logging
import ntcore
import dataclasses
import wpiutil.wpistruct
from wpimath.geometry import Pose2d
from iSpy.plugins.bases import UtilityBase


@wpiutil.wpistruct.make_wpistruct(name="Object")
@dataclasses.dataclass
class ObjectStruct:
    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


# vision_data is published as raw JSON (a single string topic holding the full
# universal Object.to_dict() list). Because every pipeline flattens to the same
# keys, JSON works for object_detection, april_tag, qr_code, optical_flow,
# depth, etc. - the robot picks the entries it cares about by "name" and
# "vis_type". The struct[] form remains available for back-compat via the
# data_type dropdown.
DEFAULT_PUBLISH = [
    {"name": "fps",            "data_type": "number",  "source": "fps",             "nt_topic": "fps"},
    {"name": "num_detections", "data_type": "number",  "source": "detection_count",  "nt_topic": "num_detections"},
    {"name": "camera_lag",     "data_type": "number",  "source": "camera_lag_s",     "nt_topic": "camera_lag"},
    {"name": "vision_data",    "data_type": "json",    "source": "detections",       "nt_topic": "vision_data"},
    {"name": "selected_target", "data_type": "json",   "source": "addon_data.selected_target", "nt_topic": "selected_target"},
]


class NetworkTableHandler(UtilityBase):
    plugin_name = "network_table_handler"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "network_tables_ip": {
                "type": "text",
                "label": "Robot IP",
                "hint": "IP address of the robot's NetworkTables server "
                        "(usually the roboRIO).",
                "default": "10.0.0.2",
            },
            "publish": {
                "type": "list",
                "label": "Publish to NetworkTables",
                "hint": "Data entries published every tick. Sources are "
                        "frame_data keys (fps, detection_count, camera_lag_s, "
                        "detections) plus utility outputs as "
                        "addon_data.<output_key>. Type 'auto' detects bool / "
                        "number / string from the value; dicts and lists are "
                        "published as a JSON string.",
                "default": DEFAULT_PUBLISH,
                "fields": {
                    "name": {
                        "type": "text",
                        "label": "Name",
                    },
                    "data_type": {
                        "type": "select",
                        "label": "Type",
                        "options": ["auto", "number", "boolean", "string", "json", "struct[]"],
                        "default": "auto",
                    },
                    "source": {
                        "type": "select",
                        "options_source": "publish_sources",
                        "label": "Source",
                    },
                    "nt_topic": {
                        "type": "text",
                        "label": "NT Topic",
                    },
                },
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)

        ip = self.config.get("network_tables_ip", "10.0.0.2")
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.setServer(ip)
        self.inst.startClient4("iSpy")
        # startClient4 connects asynchronously in the background - never block
        # boot for up to 15s waiting on a robot that may be off. connect state
        # is reported transitionally from update() instead.
        self._conn_state = None
        self.logger.info(
            "NetworkTables client started (server %s) - connecting in the "
            "background.", ip
        )

        self._subscribers: dict = {}
        self._tables: dict = {}
        self._viewer = self.context.get("viewer3d")

    def isConnected(self) -> bool:
        return self.inst.isConnected()

    def get_health(self) -> dict:
        """Contribute a NetworkTables widget to the Health tab.

        Implemented via the optional add-on health hook so the core Health
        page no longer hard-codes NetworkTables.
        """
        connected = self.isConnected()
        ip = self.config.get("network_tables_ip", "10.0.0.2")
        return {
            "ok": connected,
            "title": "NetworkTables",
            "info": "Connected" if connected else "Not connected",
            "rows": [
                {"label": "Robot IP", "value": str(ip)},
                {"label": "Status", "value": "Connected" if connected else "Disconnected"},
            ],
        }

    def _log_connection_state(self):
        connected = self.inst.isConnected()
        if connected and self._conn_state != "up":
            self._conn_state = "up"
            self.logger.info("NetworkTables connected.")
        elif not connected and self._conn_state != "down":
            self._conn_state = "down"
            self.logger.warning(
                "NetworkTables not connected yet - client is retrying in the "
                "background. Robot server may be off or the IP may be wrong."
            )

    def get_robot_pose(self) -> Pose2d:
        if not self.isConnected():
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
        self._log_connection_state()
        if not self.isConnected():
            return

        publish_entries = self.config.get("publish", DEFAULT_PUBLISH)
        if not isinstance(publish_entries, list):
            publish_entries = DEFAULT_PUBLISH

        for entry in publish_entries:
            self._publish_entry(entry, frame_data)

        cameras = frame_data.get("cameras", [])
        for cam in cameras:
            hopper = cam.get_data_for_subsystem("hopper")
            if hopper is not None:
                self._send_boolean(hopper, "hopper_sees_object", "VisionData")

        self.inst.flush()
        self._update_viewer_overlay(frame_data)

    def _publish_entry(self, entry: dict, frame_data: dict):
        """Publish a single configured entry to NetworkTables."""
        name = entry.get("name", "")
        source = entry.get("source", "")
        nt_topic = entry.get("nt_topic", name)
        data_type = entry.get("data_type", "number")

        if not source or not nt_topic:
            return

        value = self._resolve_source(source, frame_data)
        if value is None:
            return

        try:
            if data_type == "auto":
                self._publish_auto(value, nt_topic)
            elif data_type == "json":
                self._publish_json(value, nt_topic)
            elif data_type == "struct[]":
                self._send_detections(value)
            elif data_type == "number":
                self._send_data(float(value), nt_topic, "VisionData")
            elif data_type == "boolean":
                self._send_data(bool(value), nt_topic, "VisionData")
            elif data_type == "string":
                self._send_data(str(value), nt_topic, "VisionData")
        except Exception as e:
            self.logger.error("Failed to publish '%s': %s", name, e)

    def _publish_json(self, value, nt_topic: str):
        """Publish arbitrary structured data as a single JSON-string topic.

        Every pipeline flattens its detections to the universal schema, so a
        list of Objects JSON-serializes cleanly; robot code only needs to parse
        one string topic. Any value that can't be serialized is dropped with a
        warning instead of crashing the vision loop.
        """
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError) as e:
            self.logger.warning(
                "Could not serialize JSON topic '%s': %s", nt_topic, e,
            )
            return
        self._send_data(encoded, nt_topic, "VisionData")

    def _publish_auto(self, value, nt_topic: str):
        """Publish with automatic scalar type detection.

        bool must be tested before int (bool is a subclass of int). Dicts,
        lists, tuples, and anything else non-scalar fall back to a JSON
        string - arbitrary values are never turned into NT4 structs.
        """
        if isinstance(value, bool):
            self._send_data(bool(value), nt_topic, "VisionData")
        elif isinstance(value, (int, float)):
            self._send_data(float(value), nt_topic, "VisionData")
        elif isinstance(value, str):
            self._send_data(str(value), nt_topic, "VisionData")
        else:
            try:
                encoded = json.dumps(value, default=str)
            except (TypeError, ValueError) as e:
                self.logger.warning(
                    "Could not serialize addon output for topic '%s': %s",
                    nt_topic, e,
                )
                return
            self._send_data(encoded, nt_topic, "VisionData")

    def _resolve_source(self, source: str, frame_data: dict):
        """Resolve a dotted source key against frame_data.

        Supports dotted paths like ``debug_data.fps`` and utility outputs
        via ``addon_data.<output_key>``, plus special-case ``detections``
        which returns the raw detection list.
        """
        if source == "detections":
            return frame_data.get("detections", [])

        parts = source.split(".")
        obj = frame_data
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return obj

    # -- internal helpers --------------------------------------------------

    def _get_table(self, table_name: str):
        if table_name not in self._tables:
            self._tables[table_name] = self.inst.getTable(table_name)
        return self._tables[table_name]

    def _send_detections(self, detections: list):
        """Publish detections as struct[] using the universal output schema.

        Every entry is flattened via Object.to_dict() (or passed through if
        already a schema dict), so ANY pipeline's output -- object detection,
        april tags, qr codes, optical flow, depth -- publishes identically.
        """
        table = self._get_table("VisionData")
        pub_key = "pub/VisionData/vision_data"
        structs = []
        for entry in detections:
            data = entry.to_dict() if hasattr(entry, "to_dict") else entry
            if not isinstance(data, dict):
                continue
            structs.append(ObjectStruct(
                x=float(data.get("x", 0.0)),
                y=float(data.get("y", 0.0)),
                z=float(data.get("z", 0.0)),
                roll=float(data.get("roll", 0.0)),
                pitch=float(data.get("pitch", 0.0)),
                yaw=float(data.get("yaw", 0.0)),
            ))
        if pub_key not in self._subscribers:
            self._subscribers[pub_key] = table.getStructArrayTopic(
                "vision_data", ObjectStruct
            ).publish()
        self._subscribers[pub_key].set(structs)

    def _send_data(self, value, data_name: str, table_name: str):
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

    def _send_boolean(self, value: bool, data_name: str, table_name: str):
        self._send_data(value, data_name, table_name)

    # -- viewer overlay ----------------------------------------------------

    def _update_viewer_overlay(self, frame_data: dict):
        """Push a robot overlay to the 3D viewer via the generic overlay API."""
        if self._viewer is None:
            return
        pose = frame_data.get("robot_pose")
        if not pose or not isinstance(pose, dict):
            return
        self._viewer.add_overlay("robot", {
            "type": "box",
            "x": pose["x"],
            "y": pose["y"],
            "z": 0.15,
            "roll": 0,
            "pitch": 0,
            "yaw": pose["heading"],
            "color": "#4c8bf5",
            "label": "Robot",
            "data": {"width": 0.76, "height": 0.30, "depth": 0.69},
        })

    def stop(self):
        pass
