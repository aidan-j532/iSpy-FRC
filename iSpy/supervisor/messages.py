"""User-editable page copy and image loader (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MESSAGES_FILE = ASSETS_DIR / "ui_messages.json"

DEFAULT_MESSAGES = {
    "restarting": [
        "Get a sip of water.",
        "Stretch your legs for a second.",
        "Check that the robot battery is still connected.",
        "This is a good time to make a note in the match log.",
        "Take a breath - the field will still be there.",
        "Look away from the screen for a moment.",
        "Make sure nobody is standing in the robot's path.",
        "One quick stretch, then back to it.",
        "Tell your drive team the vision system is rebooting.",
        "Almost there.",
    ],
    "fresh_boot": [
        "Fresh boots take a little longer than normal ones.",
        "Grab a snack - first-time setup downloads models.",
        "This is the install path, so it only happens once.",
        "Check that the board has a network connection.",
        "The camera configuration will be ready when this finishes.",
        "You can watch progress in Outputs/log.txt.",
        "Have a look at the robot while iSpy sets itself up.",
        "Take a walk and come back in a minute.",
        "First boot also prepares the vision models.",
        "Thanks for waiting - the web UI will be back shortly.",
    ],
}

_DEFAULT_IMAGE = ASSETS_DIR / "restarting.png"


def _clean_list(value, fallback):
    if not isinstance(value, list):
        return list(fallback)
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or list(fallback)


def _resolve_image(raw) -> str | None:
    candidates = []
    if isinstance(raw, str) and raw.strip():
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend((Path.cwd() / path, ASSETS_DIR / path))
    candidates.append(_DEFAULT_IMAGE)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def load_messages() -> dict:
    data = {}
    try:
        with open(MESSAGES_FILE, "r", encoding="utf-8-sig") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}

    return {
        "restarting": _clean_list(
            data.get("restarting"), DEFAULT_MESSAGES["restarting"]
        ),
        "fresh_boot": _clean_list(
            data.get("fresh_boot"), DEFAULT_MESSAGES["fresh_boot"]
        ),
        "image": _resolve_image(data.get("image")),
    }
