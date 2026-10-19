"""Performance profiles: one table that trades "luxury" features for FPS.

The Settings page exposes a 5-stop slider backed by the top-level
``performance_level`` config value (0 = Max FPS ... 4 = Full Features, the
default). Every consumer reads its knobs from the PROFILES table here - there
are deliberately no scattered ``if level == 2`` checks anywhere else.

What no level may ever trim: core detection, trackers, NetworkTables
publishing, robot pose reads, the watchdog/control channel and calibration
correctness. Those are not performance knobs and never appear in this table.

The level is resolved into memory once per boot (``arm_from_config``) so the
web-on/off decision, the loop and every web module all read the same value.
Changing the config value takes effect on the next boot;
``set_level_for_run`` exists for the bench harness and tests.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_LEVEL = 0
MAX_LEVEL = 4
#: what an install without any performance_level (or an invalid one) gets -
#: exactly today's full-feature behavior.
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


# the single source of truth. at level 0 the web is off, so its knobs mirror
# level 1: if the UI is forced back on (ispy start --web / ISPY_FORCE_WEB=1) it
# must be the cheapest possible UI, never a full-featured one.
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

#: per-level bullet list for the settings slider and the confirmation dialog.
#: describes what the level changes relative to Full Features.
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


def clamp(value) -> int | None:
    """Coerce value into 0..4, or None when it is not a usable number."""
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
    """The level in effect for this run (defaults to Full Features)."""
    with _lock:
        return _runtime_level if _runtime_level is not None else DEFAULT_LEVEL


def get_profile() -> Profile:
    return PROFILES[get_level()]


def set_level_for_run(level: int) -> int:
    """Set the in-memory level (never written anywhere). Returns the applied value."""
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


def log_profile(level: int | None = None) -> None:
    """The boot log line for the active profile (with recovery help at level 0)."""
    profile = PROFILES[get_level() if level is None else level]
    logger.info("Performance profile: %s (%d)", profile.label, profile.level)
    if profile.level == 0:
        # loud on purpose: 0 is the only level that takes the UI away
        logger.warning(
            "Performance profile 0 (Max FPS): web UI OFF, rollback recorder "
            "OFF, metrics OFF, FPS console print OFF. %s",
            _RECOVERY,
        )


def configured_level(config) -> int | None:
    """The raw performance_level from a config object, or None when unset/invalid."""
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


def arm_from_config(config) -> int:
    """Resolve the level out of the config into memory. Called once at boot."""
    level = configured_level(config)
    return set_level_for_run(DEFAULT_LEVEL if level is None else level)


def restart_pending(config) -> bool:
    """True when the configured level differs from the one the process runs."""
    level = configured_level(config)
    return level is not None and level != get_level()


def web_forced_on() -> bool:
    """True when the web UI was force-enabled for this run (recovery path)."""
    return os.environ.get("ISPY_FORCE_WEB", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off", "none")
    return bool(value)


def effective_app_mode(config) -> bool:
    """The one web-UI on/off decision, shared by boot and iSpy.__init__.

    Off when the user disabled app_mode, when Game Mode armed its one-shot run,
    or when the performance level turns the web off. ISPY_FORCE_WEB=1 forces it
    back on for this run only. This never writes app_mode anywhere.
    """
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
    """The bullet list for a level (falls back to the default level's)."""
    level = clamp(level)
    if level is None:
        level = DEFAULT_LEVEL
    return list(LEVEL_NOTES[level])


def recovery_help() -> str:
    return _RECOVERY


def levels_payload() -> list[dict]:
    """Everything the settings slider needs to render all five stops."""
    return [
        {
            "level": level,
            "label": PROFILES[level].label,
            "notes": describe(level),
            "web_enabled": PROFILES[level].web_enabled,
        }
        for level in range(MIN_LEVEL, MAX_LEVEL + 1)
    ]
