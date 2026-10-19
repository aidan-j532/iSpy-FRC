# performance profiles: one table trading luxury features for FPS. backed by the
# performance_level key (0 = Max FPS ... 4 = Full Features); detection is never trimmed

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_LEVEL = 0
MAX_LEVEL = 4
#: what an install without a performance_level (or an invalid one) gets
DEFAULT_LEVEL = 4


@dataclass(frozen=True)
class Profile:
    level: int
    label: str
    #: web UI on/off for the run (level 0 turns it off)
    web_enabled: bool
    #: frontend setInterval scale factor
    ui_poll_multiplier: float
    #: pause frontend polling while the browser tab is hidden
    pause_poll_hidden: bool
    #: max rate WebApp fans out to the web modules, in Hz (0 = every tick)
    web_update_hz: float
    #: skip building SSE payloads/frames while no client is connected
    skip_web_without_viewers: bool
    #: metrics recording of the vision numbers (level 0: off entirely)
    metrics_enabled: bool
    #: metrics sample rate in Hz (0 = every tick)
    metrics_sample_hz: float
    #: MJPEG preview encode rate / JPEG quality / downscale cap (None = capture
    #: resolution)
    mjpeg_fps: float
    mjpeg_quality: int
    mjpeg_max_width: int | None
    #: cache seconds for the psutil / hardware-load / temperature reads
    system_metrics_cache_s: float
    #: skip camera.plot()/debug frames unless a feed or the viewer is open
    annotate_only_when_watched: bool
    #: rollback recorder on/off (level 0: off)
    rollback_recorder: bool
    #: minimum rollback frame stride (1 = keep the configured stride)
    rollback_downsample_min: int
    #: print the live "FPS:" line on the console
    fps_console_print: bool
    #: only build the 3D viewer's per-tick object snapshot while its page polls
    viewer3d_open_only: bool


# the whole table. level 0's web knobs mirror level 1 so a forced-on UI stays
# the cheapest possible one
PROFILES: dict[int, Profile] = {
    4: Profile(
        level=4,
        label="Full Features",
        web_enabled=True,
        ui_poll_multiplier=1.0,
        pause_poll_hidden=False,
        web_update_hz=0,
        skip_web_without_viewers=False,
        metrics_enabled=True,
        metrics_sample_hz=0,
        mjpeg_fps=20,
        mjpeg_quality=70,
        mjpeg_max_width=None,
        system_metrics_cache_s=0,
        annotate_only_when_watched=False,
        rollback_recorder=True,
        rollback_downsample_min=1,
        fps_console_print=True,
        viewer3d_open_only=False,
    ),
    3: Profile(
        level=3,
        label="Smooth",
        web_enabled=True,
        ui_poll_multiplier=1.5,
        pause_poll_hidden=True,
        web_update_hz=20,
        skip_web_without_viewers=True,
        metrics_enabled=True,
        metrics_sample_hz=0,
        mjpeg_fps=20,
        mjpeg_quality=70,
        mjpeg_max_width=None,
        system_metrics_cache_s=1,
        annotate_only_when_watched=False,
        rollback_recorder=True,
        rollback_downsample_min=1,
        fps_console_print=True,
        viewer3d_open_only=True,
    ),
    2: Profile(
        level=2,
        label="Balanced",
        web_enabled=True,
        ui_poll_multiplier=2.0,
        pause_poll_hidden=True,
        web_update_hz=10,
        skip_web_without_viewers=True,
        metrics_enabled=True,
        metrics_sample_hz=5,
        mjpeg_fps=15,
        mjpeg_quality=60,
        mjpeg_max_width=None,
        system_metrics_cache_s=2,
        annotate_only_when_watched=True,
        rollback_recorder=True,
        rollback_downsample_min=2,
        fps_console_print=True,
        viewer3d_open_only=True,
    ),
    1: Profile(
        level=1,
        label="Performance",
        web_enabled=True,
        ui_poll_multiplier=4.0,
        pause_poll_hidden=True,
        web_update_hz=2,
        skip_web_without_viewers=True,
        metrics_enabled=True,
        metrics_sample_hz=1,
        mjpeg_fps=10,
        mjpeg_quality=50,
        mjpeg_max_width=960,
        system_metrics_cache_s=5,
        annotate_only_when_watched=True,
        rollback_recorder=True,
        rollback_downsample_min=4,
        fps_console_print=False,
        viewer3d_open_only=True,
    ),
    0: Profile(
        level=0,
        label="Max FPS",
        web_enabled=False,
        ui_poll_multiplier=4.0,
        pause_poll_hidden=True,
        web_update_hz=2,
        skip_web_without_viewers=True,
        metrics_enabled=False,
        metrics_sample_hz=0,
        mjpeg_fps=10,
        mjpeg_quality=50,
        mjpeg_max_width=960,
        system_metrics_cache_s=5,
        annotate_only_when_watched=True,
        rollback_recorder=False,
        rollback_downsample_min=4,
        fps_console_print=False,
        viewer3d_open_only=True,
    ),
}

#: per-level bullet list for the settings slider and the confirmation dialog,
#: describing what each level changes relative to Full Features
LEVEL_NOTES: dict[int, tuple[str, ...]] = {
    4: (
        "Everything on - the full web UI, live previews, overlays and recording.",
        "This is the default; nothing is trimmed.",
    ),
    3: (
        "UI polling 1.5x slower, paused while the browser tab is hidden.",
        "Web modules updated at most 20 times a second.",
        "Web payloads skipped while no browser is connected.",
        "System metrics cached for a second.",
        "3D viewer data only built while the viewer page is open.",
    ),
    2: (
        "UI polling 2x slower.",
        "Web modules updated at most 10 times a second.",
        "Annotated frames only drawn while a feed or the viewer is watching.",
        "Camera previews at 15 fps / q60.",
        "Recording keeps every 2nd frame at most.",
        "Vision metrics sampled at 5 Hz.",
        "System metrics cached for two seconds.",
    ),
    1: (
        "UI polling 4x slower.",
        "Web modules updated at most twice a second.",
        "Camera previews at 10 fps / q50, scaled to 960px wide.",
        "Recording keeps every 4th frame at most.",
        "Vision metrics sampled at 1 Hz.",
        "System metrics cached for five seconds.",
        "FPS console print off.",
    ),
    0: (
        "Web UI off for this run.",
        "Recording off - no new rollback sessions are created.",
        "Vision metrics off (nothing is written on shutdown).",
        "FPS console print off.",
        "Annotated/debug frames off.",
        "Detection, trackers, NetworkTables, robot pose and the watchdog are untouched.",
    ),
}

_RECOVERY = (
    "To get the web UI back: run 'ispy performance 4' on the board (or edit "
    "Config/config.json by hand) and restart, or run 'ispy start --web' / set "
    "ISPY_FORCE_WEB=1 to force the UI on for one run."
)

_lock = threading.Lock()
_runtime_level: int | None = None


# turns garbage into None so a bad config value just falls back to the default
def clamp(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    if level < MIN_LEVEL or level > MAX_LEVEL:
        return None
    return level


def get_level() -> int:
    with _lock:
        return _runtime_level if _runtime_level is not None else DEFAULT_LEVEL


def get_profile() -> Profile:
    return PROFILES[get_level()]


# in-memory only, never written anywhere. returns the applied value
def set_level_for_run(level: int) -> int:
    global _runtime_level
    if level is None:
        level = DEFAULT_LEVEL
    applied = clamp(level)
    if applied is None:
        logger.warning(
            "Invalid performance level %r - falling back to %d (%s).",
            level,
            DEFAULT_LEVEL,
            PROFILES[DEFAULT_LEVEL].label,
        )
        applied = DEFAULT_LEVEL
    with _lock:
        changed = _runtime_level != applied
        _runtime_level = applied
    if changed:
        log_profile(applied)
    return applied


# the boot log line, plus recovery help at level 0 (the one level that takes
# the UI away, so it warns loud on purpose)
def log_profile(level: int | None = None) -> None:
    profile = PROFILES[get_level() if level is None else level]
    logger.info("Performance profile: %s (%d)", profile.label, profile.level)
    if profile.level == 0:
        logger.warning(
            "Performance profile 0 (Max FPS): web UI OFF, rollback recorder "
            "OFF, metrics OFF, FPS console print OFF. %s",
            _RECOVERY,
        )


# the raw performance_level from a config object, None when unset/invalid
def configured_level(config) -> int | None:
    if config is None:
        return None
    try:
        raw = config.get("performance_level", DEFAULT_LEVEL)
    except Exception:
        return None
    level = clamp(raw)
    if level is None and raw is not None:
        logger.warning(
            "Invalid performance_level %r in config - using %d (%s) until it "
            "is fixed on the Settings page.",
            raw,
            DEFAULT_LEVEL,
            PROFILES[DEFAULT_LEVEL].label,
        )
    return level


# resolve the level out of the config into memory, called once at boot
def arm_from_config(config) -> int:
    level = configured_level(config)
    return set_level_for_run(DEFAULT_LEVEL if level is None else level)


def restart_pending(config) -> bool:
    level = configured_level(config)
    return level is not None and level != get_level()


def web_forced_on() -> bool:
    return os.environ.get("ISPY_FORCE_WEB", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off", "none")
    return bool(value)


# the one web on/off decision, shared by boot and iSpy.__init__. never writes
# app_mode; ISPY_FORCE_WEB=1 brings the UI back for a single run
def effective_app_mode(config) -> bool:
    if web_forced_on():
        return True
    try:
        configured = config.get("app_mode", True) if config is not None else True
    except Exception:
        configured = True
    if not _truthy(configured):
        return False
    return get_profile().web_enabled


def describe(level: int) -> list[str]:
    level = clamp(level)
    if level is None:
        level = DEFAULT_LEVEL
    return list(LEVEL_NOTES[level])


def recovery_help() -> str:
    return _RECOVERY


# everything the settings slider needs to render all five stops
def levels_payload() -> list[dict]:
    return [
        {
            "level": level,
            "label": PROFILES[level].label,
            "notes": describe(level),
            "web_enabled": PROFILES[level].web_enabled,
        }
        for level in range(MIN_LEVEL, MAX_LEVEL + 1)
    ]
