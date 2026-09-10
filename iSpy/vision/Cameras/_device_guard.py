import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROC = Path("/proc")

# Distinct iSpy marker: the repo root of THIS install. Any process holding a
# camera whose command-line references this path belongs to iSpy and is
# protected. Falls back to matching the module package name.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])

# graceful-then-force timings
_TERM_GRACE_S = 2.0


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _devnode(target) -> Path:
    if isinstance(target, str) and target.startswith("/dev/video"):
        return Path(target)
    if isinstance(target, tuple) and len(target) == 2 and target[0] == "v4l2":
        return Path(target[1])
    if isinstance(target, int):
        return Path(f"/dev/video{target}")
    return None


def _process_cmdline(pid: int) -> str:
    try:
        raw = (_PROC / str(pid) / "cmdline").read_bytes().split(b"\x00")
        return " ".join(part.decode(errors="replace") for part in raw if part)
    except OSError:
        return ""


def _is_ispy_process(cmdline: str) -> bool:
    if not cmdline:
        return False
    low = cmdline.lower()
    if _REPO_ROOT and _REPO_ROOT.lower() in low:
        return True
    return "ispy" in low  # module/daemon names, watchdog, game_loop, boot, ...


def _own_pids() -> set:
    pids = set()
    pid = os.getpid()
    while pid and pid > 1:
        pids.add(pid)
        try:
            with open(_PROC / str(pid) / "stat", "r", encoding="utf8") as f:
                parts = f.read().split()
            pid = int(parts[3])
        except (OSError, IndexError, ValueError):
            break
    return pids


def _holders_via_fuser(devnode: Path) -> list[int]:
    try:
        result = subprocess.run(
            ["fuser", str(devnode)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids = []
    for tok in result.stdout.split():
        if tok.isdigit():
            pids.append(int(tok))
    return pids


def _holders_via_proc(devnode: Path) -> list[int]:
    try:
        dev_str = str(devnode)
    except Exception:
        return []
    holders = []
    for fd_dir in _PROC.glob("[0-9]*/fd"):
        pid_str = fd_dir.parts[-2]
        for link in fd_dir.iterdir():
            try:
                if os.path.realpath(link) == dev_str:
                    holders.append(int(pid_str))
                    break
            except OSError:
                continue
    return holders


def _kill_holder(pid: int) -> None:
    cmdline = _process_cmdline(pid)
    if _is_ispy_process(cmdline) or pid in _own_pids():
        logger.debug("device-guard: protecting iSpy pid %d (%s)", pid, cmdline)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        logger.warning(
            "device-guard: SIGTERM to pid %d holding the camera (%s)",
            pid,
            cmdline or "?",
        )
    except OSError:
        return
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        if not _killable(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning("device-guard: SIGKILL to pid %d (didn't exit in time)", pid)
    except OSError:
        pass


def _killable(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def free_camera_device(target, log_noop: bool = False) -> list[int]:
    if not _is_linux():
        return []
    devnode = _devnode(target)
    if devnode is None or not devnode.is_char_device():
        if log_noop:
            logger.debug("device-guard: %r is not a char devnode to guard", target)
        return []

    if shutil.which("fuser"):
        holders = _holders_via_fuser(devnode)
    else:
        holders = _holders_via_proc(devnode)

    own = _own_pids()
    killed = []
    for pid in holders:
        cmdline = _process_cmdline(pid)
        if pid in own or _is_ispy_process(cmdline):
            logger.info(
                "device-guard: keeping iSpy pid %d on %s (%s)",
                pid,
                devnode,
                cmdline or "?",
            )
            continue
        _kill_holder(pid)
        killed.append(pid)
    return killed
