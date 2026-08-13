"""DeviceManager - the registry that holds every Device iSpy talks to."""

import logging
from typing import Iterable

from iSpy.Devices.device import Device


class DeviceManager:
    def __init__(self, devices: Iterable[Device] | None = None):
        self.logger = logging.getLogger(__name__)
        self._devices: dict[str, Device] = {}
        for device in devices or []:
            self.register(device)

    def register(self, device: Device) -> None:
        """Add a device. Its verify() result decides if it starts enabled."""
        if device.name in self._devices:
            self.logger.warning("Device '%s' already registered - replacing it.", device.name)
        device.verified = device.verify()
        self._devices[device.name] = device
        if not device.verified:
            self.logger.warning(
                "Device '%s' failed verification - it will not receive updates.", device.name
            )
        else:
            self.logger.info("Device '%s' verified and online.", device.name)

    def get(self, name: str) -> Device | None:
        return self._devices.get(name)

    @property
    def devices(self) -> list[Device]:
        return list(self._devices.values())

    def update(self, frame_data: dict) -> None:
        """Feed one vision frame to every verified device."""
        for device in self.devices:
            if getattr(device, "verified", False):
                try:
                    device.update(frame_data)
                except Exception:
                    self.logger.exception("Device '%s' update failed", device.name)

    def notify_all(self, message: str, **payload) -> None:
        """Push a message to every verified device at once."""
        for device in self.devices:
            if getattr(device, "verified", False):
                device._send_safe(message, **payload)

    def stop(self) -> None:
        """Stop every device."""
        for device in self.devices:
            try:
                if hasattr(device, "_stop_event"):
                    device._stop_event.set()
                device.stop()
            except Exception:
                self.logger.exception("Error stopping device '%s'", device.name)