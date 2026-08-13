import logging
import threading
import time
from typing import Any


class Device:
    def __init__(self, name: str = "Generic iSpy Device"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.last_notified = 0.0
        self._lock = threading.Lock()

    def verify(self) -> bool:
        return True

    def update(self, frame_data: dict) -> None:
        pass

    def notify(self, message: str, **payload: Any) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement notify()")

    def stop(self) -> None:
        pass

    # ---- helpers for subclasses ----

    def cooldown_ok(self, cooldown_s: float) -> bool:
        return (time.monotonic() - self.last_notified) >= cooldown_s

    def _schedule_every(self, interval_s: float, message: str, **payload: Any):
        def _run():
            while not self._stop_event.is_set():
                time.sleep(interval_s)
                self._send_safe(message, **payload)

        self._stop_event = threading.Event()
        t = threading.Thread(target=_run, daemon=True, name=f"{self.name}-scheduler")
        t.start()
        return t

    def _send_safe(self, message: str, **payload: Any) -> bool:
        with self._lock:
            try:
                ok = self.notify(message, **payload)
                if ok:
                    self.last_notified = time.monotonic()
                return ok
            except Exception:
                self.logger.exception("Device '%s' failed to notify", self.name)
                return False