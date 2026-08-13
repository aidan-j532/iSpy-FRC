"""RobotDevice - writes alerts into NetworkTables for the robot to read.

The robot subscribes to the "iSpy/DeviceMessage" table. This is
the "human + hardware" half of Devices: the vision box can poke
the robot (or any NT-connected hardware) directly without going
through the normal VisionData stream.
"""

import logging
import time

import ntcore

from iSpy.Devices.device import Device


class RobotDevice(Device):
    def __init__(
        self,
        name: str = "Robot",
        server_ip: str = "10.0.0.2",
        table_name: str = "iSpy/DeviceMessage",
    ):
        super().__init__(name)
        self.server_ip = server_ip
        self.table_name = table_name
        self._inst = None
        self._pub = None

    def verify(self) -> bool:
        try:
            self._inst = ntcore.NetworkTableInstance.getDefault()
            if self._inst.getConnections() and self._inst.isConnected():
                return True
        except Exception as e:
            self.logger.warning("NetworkTables init failed: %s", e)
        # not connected yet - still register; it can come up later
        return False

    def notify(self, message: str, **payload) -> bool:
        if self._inst is None:
            return False
        table = self._inst.getTable(self.table_name)
        table.putString("message", message)
        if payload:
            table.putString("payload", str(payload))
        table.putNumber("timestamp_ms", time.time() * 1000)
        try:
            self._inst.flush()
            return True
        except Exception as e:
            self.logger.error("NT write failed: %s", e)
            return False

    def stop(self):
        if self._inst is not None:
            try:
                self._inst.stopClient()
            except Exception:
                pass