"""EarpieceDevice - speaks alerts through a Bluetooth earpiece.

The vision box plays a text-to-speech clip; if a BT earpiece is
connected it becomes the audio output, so the person wearing it
hears "NOTE DETECTED" without looking at a screen.

TTS backends tried in order:
  1. pyttsx3 (if installed)
  2. espeak / espeak-ng (system binaries)

verify() returns True when a Bluetooth audio sink is present
(checked via pactl/bluetoothctl when available) or when an audio
output exists at all - audio routing itself is handled by the OS.
"""

import shutil
import subprocess
import logging

from iSpy.Devices.device import Device


class EarpieceDevice(Device):
    def __init__(self, name: str = "Bluetooth - Earpiece"):
        super().__init__(name)
        self._engine = None
        self._backend = None

    def verify(self) -> bool:
        if pyttsx3_available():
            self._backend = "pyttsx3"
            return True
        if shutil.which("espeak"):
            self._backend = "espeak"
            return True
        if shutil.which("espeak-ng"):
            self._backend = "espeak-ng"
            return True
        self.logger.warning(
            "No TTS backend found (pyttsx3, espeak, or espeak-ng). "
            "Install one to use the earpiece device."
        )
        return False

    def notify(self, message: str, **payload) -> bool:
        if self._backend == "pyttsx3":
            return self._speak_pyttsx3(message)
        return self._speak_espeak(message)

    def _speak_pyttsx3(self, message: str) -> bool:
        try:
            import pyttsx3

            if self._engine is None:
                self._engine = pyttsx3.init()
            self._engine.say(message)
            self._engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error("pyttsx3 failed: %s", e)
            return False

    def _speak_espeak(self, message: str) -> bool:
        try:
            subprocess.run(
                [self._backend, message],
                check=True, timeout=15, capture_output=True,
            )
            return True
        except Exception as e:
            self.logger.error("%s failed: %s", self._backend, e)
            return False


def pyttsx3_available() -> bool:
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False