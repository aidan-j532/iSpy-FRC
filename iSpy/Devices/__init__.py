"""Devices - everything iSpy talks to that is not a camera.

Developer-controlled by design: devices are created in code, not
configured by users. See device.py (base class) and devices.py
(the DeviceManager).

    from iSpy.Devices import DeviceManager, EarpieceDevice
    devices = DeviceManager([EarpieceDevice()])
    vision = iSpy(cameras, config, devices=devices)
"""

from iSpy.Devices.device import Device
from iSpy.Devices.devices import DeviceManager
from iSpy.Devices.BuiltIn import EarpieceDevice, EmailDevice, PhoneDevice, RobotDevice

__all__ = [
    "Device",
    "DeviceManager",
    "EarpieceDevice",
    "EmailDevice",
    "PhoneDevice",
    "RobotDevice",
]