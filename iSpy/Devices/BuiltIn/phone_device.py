import logging
from urllib.parse import urljoin

import requests

from iSpy.Devices.device import Device


class PhoneDevice(Device):
    def __init__(
        self,
        name: str = "Bluetooth - Android Phone",
        endpoint: str = "http://192.168.1.100:8000/message",
        ping_path: str = "ping",
        timeout: float = 5.0,
    ):
        super().__init__(name)
        self.endpoint = endpoint
        self.ping_path = ping_path
        self.timeout = timeout

    def verify(self) -> bool:
        try:
            r = requests.get(
                urljoin(self.endpoint.rstrip("/") + "/", self.ping_path),
                timeout=self.timeout,
            )
            return r.status_code == 200
        except Exception as e:
            self.logger.warning(
                "Phone endpoint %s not reachable: %s", self.endpoint, e
            )
            return False

    def notify(self, message: str, **payload) -> bool:
        try:
            body = {"message": message, "source": "iSpy"}
            if payload:
                body["payload"] = {k: str(v) for k, v in payload.items()}
            r = requests.post(self.endpoint, json=body, timeout=self.timeout)
            if r.status_code != 200:
                self.logger.error("Phone endpoint replied %s: %s", r.status_code, r.text[:200])
                return False
            return True
        except Exception as e:
            self.logger.error("Phone notify failed: %s", e)
            return False