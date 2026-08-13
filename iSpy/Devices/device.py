"""Base class for every iSpy Device.

A Device is anything iSpy can talk to that is NOT a camera:
a human (earpiece, phone, email) or hardware (the robot, a
controller, etc). Devices are developer-controlled - they are
not plugins, not user-configurable, and have no web UI.

Each device decides WHEN it fires. The manager only feeds it
vision data every frame and remembers when it last sent
something (for cooldowns).
"""

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
        """Return True if this device's transport is available right now.

        Called once at startup by the manager. Subclasses should check
        things like 'is the earpiece paired?' or 'can we reach SMTP?'.
        """
        return True

    def update(self, frame_data: dict) -> None:
        """Called every vision frame with the full frame_data dict.

        Subclasses that fire on events inspect frame_data here (e.g.
        'detections > 0 for 2 seconds') and call self.notify() when
        their condition is met. Subclasses that fire on a schedule
        can ignore frame_data and use _schedule_every().
        """
        pass

    def notify(self, message: str, **payload: Any) -> bool:
        """Actually send a message through this device's transport.

        Returns True on success. Subclasses must implement this.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement notify()")

    def stop(self) -> None:
        """Clean up (close connections, threads, bluetooth links)."""
        pass

    # ---- helpers for subclasses ----

    def cooldown_ok(self, cooldown_s: float) -> bool:
        """True if at least cooldown_s seconds passed since the last send.

        Use this inside update() to avoid spamming a device at 30 fps.
        """
        return (time.monotonic() - self.last_notified) >= cooldown_s

    def _schedule_every(self, interval_s: float, message: str, **payload: Any):
        """Fire notify() every interval_s seconds, on a background thread.

        Returns the thread handle so the subclass can keep/stop it.
        """
        def _run():
            while not self._stop_event.is_set():
                time.sleep(interval_s)
                self._send_safe(message, **payload)

        self._stop_event = threading.Event()
        t = threading.Thread(target=_run, daemon=True, name=f"{self.name}-scheduler")
        t.start()
        return t

    def _send_safe(self, message: str, **payload: Any) -> bool:
        """notify() wrapped in a lock + logging - safe to call from any thread."""
        with self._lock:
            try:
                ok = self.notify(message, **payload)
                if ok:
                    self.last_notified = time.monotonic()
                return ok
            except Exception:
                self.logger.exception("Device '%s' failed to notify", self.name)
                return False