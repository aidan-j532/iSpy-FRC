"""Tiny config/status checks for the stdlib-only supervisor."""

from __future__ import annotations

import json
import os
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def project_root() -> Path:
    return Path.cwd()


def runtime_config_path() -> Path:
    return project_root() / "Config" / "config.json"


def read_runtime_config() -> dict:
    path = runtime_config_path()
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def allow_remote_restart(config: dict | None = None) -> bool:
    if config is None:
        config = read_runtime_config()
    # Missing key (older configs) keeps the documented LAN default.
    return _truthy(config.get("allow_remote_restart", True))


def in_ci() -> bool:
    for key in ("CI", "GITHUB_ACTIONS", "BUILDKITE", "TF_BUILD"):
        if _truthy(os.environ.get(key, "")):
            return True
    return False


def in_chroot() -> bool:
    """Best-effort Linux chroot detection.

    Inside a build chroot, ``/`` has a different inode than PID 1's root.  If
    /proc is unavailable (common in image-build chroots), rely on the explicit
    ISPY_NO_SUPERVISOR environment variable set by the build scripts instead.
    """
    if os.name != "posix":
        return False
    try:
        return os.stat("/").st_ino != os.stat("/proc/1/root").st_ino
    except OSError:
        return False


def supervisor_disabled() -> bool:
    if _truthy(os.environ.get("ISPY_NO_SUPERVISOR", "")):
        return True
    return in_ci() or in_chroot()


def write_normal_web_settings() -> tuple[bool, str]:
    """Set the persistent config back to a profile that serves the web UI.

    This is the action behind the web-off page's button.  It is deliberately a
    plain JSON rewrite rather than importing iSpyConfig: the supervisor has to
    stay cheap and must work before the vision process has ever booted.
    """
    path = runtime_config_path()
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Config root is not an object")
    except FileNotFoundError:
        return False, f"{path} does not exist yet - run 'ispy setup' first"
    except Exception as exc:
        return False, f"Could not read {path}: {exc}"

    data["performance_level"] = 4
    # Game Mode stores the user's previous app_mode here; restore it instead of
    # permanently forcing the UI on if they had deliberately turned it off.
    backup = data.get("game_mode_app_mode_backup")
    if backup is not None:
        data["app_mode"] = backup
        data["game_mode_app_mode_backup"] = None
    else:
        data["app_mode"] = True
    data["game_mode_web_off_next_run"] = False

    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        return False, f"Could not write {path}: {exc}"
    return True, ""
