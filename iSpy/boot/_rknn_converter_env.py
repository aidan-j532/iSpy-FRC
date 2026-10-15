"""Isolated, persistent RKNN model-conversion environment.

RKNN *conversion* (full ``rknn-toolkit2``) needs a different, incompatible
dependency set than iSpy's *runtime* environment: the toolkit pins NumPy 1.x,
SciPy, protobuf < 4 and ``pkg_resources`` (setuptools < 82), while the runtime
needs the ``rknn-toolkit2-lite`` inference wheel plus the project's normal
NumPy/PyTorch stack.  Installing the conversion stack into the main
environment both breaks runtime imports and mutates shared packages.

This module provisions and manages a dedicated, persistent Python virtual
environment that lives **outside** the project tree and outside the main
virtual environment.  It is created lazily, only when an actual RKNN
conversion is requested, and is reused across runs and reboots.

The environment's identity is keyed to the Python version/ABI, machine
architecture, RKNN Toolkit version and a dependency-set version, so an
incompatible environment is never silently reused.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Identity / versioning
# ---------------------------------------------------------------------------

#: Bumped whenever the pinned conversion dependency set changes.  An existing
#: converter environment built with an older depset version is rebuilt.
DEPSET_VERSION = 3

#: RKNN Toolkit 2 version targeted by this project.
RKNN_TOOLKIT_VERSION = "2.3.2"

#: Where the full toolkit2 wheels are hosted (same release the runtime
#: lite-wheel URLs and wheel-filename tests already reference).
_RKNN_FULL_BASE = (
    "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Wheels"
).rstrip("/")

#: Full toolkit2 wheel filenames keyed by (arch, python-tag).  Kept in sync
#: with the tables in iSpy.vision.optimizer.
_RKNN_FULL_WHEELS: dict[tuple[str, str], str] = {
    ("aarch64", "cp310"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("aarch64", "cp311"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("aarch64", "cp312"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ("x86_64", "cp310"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ("x86_64", "cp311"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ("x86_64", "cp312"): f"rknn_toolkit2-{RKNN_TOOLKIT_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
}

#: Pinned conversion dependency set.  These are the *minimum* packages the
#: RKNN conversion path actually imports (see _convert_rknn in optimizer.py):
#:
#: - numpy<2              toolkit2 2.3.2 is compiled against NumPy 1.x; NumPy 2
#:                          breaks the compiled ``numpy.core._multiarray_umath``
#:                          imports (the ``_center`` ImportError).
#: - scipy                 rknn toolkit imports scipy; must be ABI-compatible with
#:                          the pinned numpy (scipy < 1.15 supports numpy 1.x).
#: - onnx<1.17             onnx export + rknn.load_onnx; project convention.
#: - protobuf<4            rknn toolkit runtime dep.
#: - setuptools<82         setuptools 82 removed pkg_resources, which the toolkit
#:                          still imports at module load.
#: - torch                 ultralytics needs it to load .pt models for export.
#: - ultralytics           AGPL-3.0; build-time model export only, never runtime.
#: - onnxslim              export graph optimisation.
#: - opencv-python         calibration-image downscaling in _downscale_calib_images.
#:
#: numpy / scipy / protobuf / opencv-python are intentionally *not* installed
#: into the main runtime environment by conversion setup.
_CONVERTER_REQUIREMENTS: list[str] = [
    "numpy>=1.26.0,<2",
    "scipy>=1.10.0,<1.15",
    "onnx>=1.16.0,<1.17",
    "protobuf>=3.20.2,<4",
    # setuptools 82 removed pkg_resources; the toolkit still imports it.
    "setuptools>=64,<82",
    "torch>=2.0.0",
    "ultralytics>=8.3.0",
    "onnxslim>=0.1.87",
    "opencv-python>=4.8.0,<4.12",
]

#: Verification script run inside a freshly provisioned converter environment.
#: Every import here is required by the RKNN conversion path; a failure means
#: the environment is not usable and must be rebuilt.
_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy; assert numpy.__version__.split('.')[0] == '1', numpy.__version__",
        "import scipy",
        "import onnx",
        "import google.protobuf",
        "import setuptools; assert int(setuptools.__version__.split('.')[0]) < 82",
        "import pkg_resources",
        "import torch",
        "import ultralytics",
        "import onnxslim",
        "import cv2",
        "from rknn.api import RKNN",
        "print('OK')",
    ]
)

#: Lightweight re-verification run on *reuse*.  The full ``_VERIFY_SCRIPT``
#: imports ultralytics/torch/onnx which is expensive; a daily sanity check of
#: the two things most likely to silently break a cached environment (NumPy 1.x
#: and the toolkit import itself) is enough to keep stale environments from
#: being reused while not paying the full import tax on every launch.
_LIGHT_VERIFY_SCRIPT = "\n".join(
    [
        "import numpy; assert numpy.__version__.split('.')[0] == '1', numpy.__version__",
        "from rknn.api import RKNN",
        "print('OK')",
    ]
)

#: How often the light re-verification runs on an already-verified env (seconds).
_LIGHT_VERIFY_INTERVAL = 24 * 60 * 60

#: Minimum free disk space required before provisioning starts.  The full
#: toolkit + torch + ultralytics install easily reaches ~4 GB.
_MIN_FREE_BYTES = 2 * 1024 ** 3

#: Per-step timeouts so a wedged network never hangs a conversion forever.
_VENV_CREATE_TIMEOUT = 300
_PIP_INSTALL_TIMEOUT = 1800
_VERIFY_TIMEOUT = 300
_LIGHT_VERIFY_TIMEOUT = 120


class ConverterEnvError(RuntimeError):
    """Raised when the RKNN converter environment cannot be provisioned."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """Return the platform-appropriate base cache directory.

    Respects ``ISPY_RKNN_CONVERTER_HOME`` for testing / custom deployments,
    then falls back to the established per-platform cache locations:
    ``XDG_CACHE_HOME`` on Linux, ``LOCALAPPDATA`` on Windows, and
    ``~/Library/Caches`` on macOS.
    """
    override = os.environ.get("ISPY_RKNN_CONVERTER_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "iSpy-FRC" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ispy-frc"
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "ispy-frc"


def _arch_key() -> str:
    machine = platform.machine().lower()
    if "aarch64" in machine or "arm64" in machine:
        return "aarch64"
    if "x86_64" in machine or "amd64" in machine:
        return "x86_64"
    return machine


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _identity() -> dict[str, str]:
    return {
        "python_tag": _python_tag(),
        "arch": _arch_key(),
        "rknn_toolkit_version": RKNN_TOOLKIT_VERSION,
        "depset_version": str(DEPSET_VERSION),
    }


def _identity_hash(identity: dict[str, str]) -> str:
    payload = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _env_dir(identity: dict[str, str]) -> Path:
    return _cache_root() / "rknn-converter" / _identity_hash(identity)


def _manifest_path(env_dir: Path) -> Path:
    return env_dir / "env-manifest.json"


def _is_py(env_dir: Path) -> Path:
    """Return the python executable inside the converter env directory."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x0400, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _build_lock(lock_path: Path, timeout: float = 600.0):
    """Cross-process mutual exclusion for converter-environment provisioning.

    Uses an atomic ``O_CREAT|O_EXCL`` lock file whose content is the holder
    PID.  A lock whose holder is dead is reclaimed.  This keeps concurrent
    ``ispy-boot`` / ``ispy-bench`` / pipeline-optimize callers from racing
    to build the same environment.
    """
    deadline = time.monotonic() + timeout
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                pid = int(lock_path.read_text().strip())
            except (ValueError, OSError):
                pid = 0
            if not _pid_alive(pid):
                # stale lock from a crashed process - reclaim it
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            time.sleep(0.5)
    if not acquired:
        raise ConverterEnvError(
            "Timed out waiting for the RKNN converter-environment build lock "
            f"({lock_path}). Another process may still be provisioning the "
            "environment; wait for it to finish or remove the lock manually."
        )
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Manifest / validation
# ---------------------------------------------------------------------------


def _read_manifest(env_dir: Path) -> dict | None:
    try:
        return json.loads(_manifest_path(env_dir).read_text())
    except (OSError, ValueError):
        return None


def _env_is_current(env_dir: Path) -> bool:
    """True when *env_dir* holds a usable, identity-matching converter env."""
    if not _is_py(env_dir).exists():
        return False
    manifest = _read_manifest(env_dir)
    if manifest is None:
        return False
    for key, value in _identity().items():
        if manifest.get(key) != value:
            return False
    return True


def _cleanup_orphans(current_hash: str) -> None:
    """Remove converter-env directories that no longer match the identity.

    Keeps disk usage bounded when the Python version, architecture or pinned
    dependency set changes.  Never touches the current environment.
    """
    root = _cache_root() / "rknn-converter"
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name == current_hash:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        logger.info("Removed stale RKNN converter environment: %s", entry)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def _pip_failure_message(returncode: int, tail: str) -> str:
    """Turn a pip failure into an actionable message with targeted hints."""
    lowered = tail.lower()
    hints: list[str] = []
    if any(
        k in lowered
        for k in (
            "could not resolve",
            "timed out",
            "connection",
            "ssl",
            "tls",
            "network is unreachable",
            "no route to host",
            "[errno",
            "getaddrinfo",
        )
    ):
        hints.append(
            "This looks like a network/connection error - check internet "
            "access, firewall/proxy settings, and that the GitHub release "
            "URL is reachable, then retry."
        )
    if "no matching distribution" in lowered or "could not find a version" in lowered:
        hints.append(
            "pip could not find a compatible version - the pinned dependency "
            "set may not be published for this Python ABI. Check the wheel "
            "table for which cp<NN> tags are supported."
        )
    if "no space left" in lowered or ("disk" in lowered and "cache" in lowered):
        hints.append(
            "The cache disk is full - free space or point "
            "ISPY_RKNN_CONVERTER_HOME at a different location."
        )
    if "permission" in lowered:
        hints.append(
            "Permission denied - make sure the converter cache directory is "
            "writable."
        )
    message = (
        f"pip install failed for the RKNN converter environment "
        f"(exit {returncode}).\n{tail}"
    )
    if hints:
        message += "\n" + "\n".join(hints)
    message += (
        "\nFailed provisioning is cleaned up automatically - the next "
        "conversion attempt rebuilds the environment from scratch."
    )
    return message


def _pip_install(py: Path, *requirements: str, no_deps: bool = False) -> None:
    cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(requirements)
    logger.info("Installing into converter env: %s", " ".join(requirements))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_PIP_INSTALL_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise ConverterEnvError(
            f"pip install timed out for the RKNN converter environment "
            f"(>{_PIP_INSTALL_TIMEOUT}s). Check network/proxy settings and "
            "retry; environments can also be pre-built on an identical host "
            "and reused via the same cache path."
        ) from None
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-30:])
        raise ConverterEnvError(_pip_failure_message(proc.returncode, tail))


def _verify(py: Path) -> None:
    try:
        proc = subprocess.run(
            [str(py), "-c", _VERIFY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_VERIFY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ConverterEnvError(
            f"RKNN converter environment verification timed out "
            f"(>{_VERIFY_TIMEOUT}s). The install is likely too slow to be "
            "usable - retry, or provision on a faster host."
        ) from None
    if proc.returncode != 0 or "OK" not in proc.stdout:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-30:])
        raise ConverterEnvError(
            "RKNN converter environment verification failed - the installed "
            "dependency set is not usable.\n"
            f"{tail}\n"
            "Delete the converter environment directory and retry to rebuild it."
        )


def _light_verify(py: Path, env_dir: Path) -> bool:
    """Cheap daily sanity check before reusing a cached converter env.

    Returns ``True`` when the environment passed a lightweight import check or
    was verified within ``_LIGHT_VERIFY_INTERVAL``.  Never raises: reuse
    failures degrade to a full rebuild instead of crashing the caller.
    """
    stamp = env_dir / ".verified"
    if stamp.exists():
        try:
            if time.time() - stamp.stat().st_mtime < _LIGHT_VERIFY_INTERVAL:
                return True
        except OSError:
            pass
    try:
        proc = subprocess.run(
            [str(py), "-c", _LIGHT_VERIFY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_LIGHT_VERIFY_TIMEOUT,
        )
        ok = proc.returncode == 0 and "OK" in proc.stdout
    except Exception:
        ok = False
    if ok:
        try:
            stamp.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        except OSError:
            pass
    else:
        logger.warning(
            "RKNN converter env failed light verification (%s) - will rebuild",
            py,
        )
    return ok


def _toolkit_wheel_url() -> str:
    key = (_arch_key(), _python_tag())
    fn = _RKNN_FULL_WHEELS.get(key)
    if fn is None:
        supported = sorted(f"{a} {v}" for (a, v) in _RKNN_FULL_WHEELS if a == key[0])
        raise ConverterEnvError(
            f"No rknn-toolkit2 {RKNN_TOOLKIT_VERSION} wheel for {key[0]} "
            f"(Python {key[1]}). Supported: {', '.join(supported)}. "
            "RKNN conversion is not possible on this architecture/Python ABI."
        )
    return f"{_RKNN_FULL_BASE}/{fn}"


def _check_disk_space(env_dir: Path) -> None:
    """Refuse to start a ~4 GB provisioning on a nearly-full disk."""
    parent = env_dir.parent
    for base in (parent, *parent.parents):
        try:
            usage = shutil.disk_usage(base)
            break
        except OSError:
            continue
    else:
        return
    if usage.free < _MIN_FREE_BYTES:
        raise ConverterEnvError(
            f"Not enough free disk space to build the RKNN converter "
            f"environment: {usage.free / 1024 ** 3:.1f} GiB free on {base}, "
            f"need at least {_MIN_FREE_BYTES / 1024 ** 3:.1f} GiB. The full "
            "toolkit + torch + ultralytics install is ~4 GB. Free space or "
            "point ISPY_RKNN_CONVERTER_HOME at a different location."
        )


def _provision(env_dir: Path) -> Path:
    """Create and populate the converter environment at *env_dir*."""
    _cache_root().mkdir(parents=True, exist_ok=True)
    _check_disk_space(env_dir)
    # A previous failed provision may have left a partial environment behind -
    # never let venv/pip grow a poisoned tree on top of it.
    if env_dir.exists():
        shutil.rmtree(env_dir, ignore_errors=True)
        logger.info(
            "Removed incomplete RKNN converter env before rebuild: %s", env_dir
        )
    try:
        logger.info("Creating RKNN converter virtual environment ...")
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(env_dir)],
            capture_output=True,
            text=True,
            timeout=_VENV_CREATE_TIMEOUT,
        )
        if proc.returncode != 0:
            raise ConverterEnvError(
                "Failed to create the converter virtual environment "
                f"(exit {proc.returncode}).\n{proc.stderr}"
            )

        py = _is_py(env_dir)

        # 1. The full toolkit wheel first (--no-deps: the pinned deps come
        #    next, from PyPI, at versions the runtime environment must not
        #    receive).
        _pip_install(py, _toolkit_wheel_url(), no_deps=True)

        # 2. The pinned conversion dependency set.
        _pip_install(py, *_CONVERTER_REQUIREMENTS)

        # 3. Verify the environment can actually import the conversion stack.
        _verify(py)

        # 4. Record the manifest only after a fully successful build.
        manifest = _identity()
        manifest["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _manifest_path(env_dir).write_text(json.dumps(manifest, indent=2))
        # The full verify just passed - stamp it so reuse skips the light check.
        try:
            (env_dir / ".verified").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        except OSError:
            pass

        logger.info("RKNN converter environment ready at %s", env_dir)
        return py
    except Exception:
        # Never leave a half-built environment that could be mistaken for a
        # healthy one - remove it so the next attempt starts clean.
        shutil.rmtree(env_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ispy_source_root() -> Path:
    """Return the directory that contains the ``iSpy`` package.

    Works for a source checkout, an editable installation and a regular
    installed deployment: the installed package layout mirrors the source
    layout, so the parent of the ``iSpy`` package directory is the right
    ``PYTHONPATH`` entry in every case.
    """
    return Path(__file__).resolve().parents[2]


def converter_python() -> Path:
    """Return the Python executable of a ready RKNN converter environment.

    Provisions the environment on first use; reuses it thereafter.  Raises
    :class:`ConverterEnvError` with an actionable message when provisioning
    is impossible (unsupported arch/ABI, no network, disk space, ...).
    """
    identity = _identity()
    env_dir = _env_dir(identity)
    py = _is_py(env_dir)

    if _env_is_current(env_dir):
        if _light_verify(py, env_dir):
            logger.debug("Reusing RKNN converter environment at %s", env_dir)
            _cleanup_orphans(_identity_hash(identity))
            return py
        # Light verification failed (or the stamp is stale and the env is
        # broken) - fall through to a locked rebuild instead of reusing a
        # corruption-prone environment.
        logger.warning(
            "RKNN converter environment at %s failed light verification - "
            "rebuilding it.",
            env_dir,
        )

    lock_path = _cache_root() / "rknn-converter" / "build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _build_lock(lock_path):
        # Re-check under the lock - another process may have just built it.
        if _env_is_current(env_dir) and _light_verify(py, env_dir):
            return py
        py = _provision(env_dir)
        _cleanup_orphans(_identity_hash(identity))
        return py


def converter_env_status() -> dict:
    """Return a JSON-serialisable status dict (for diagnostics / tests)."""
    identity = _identity()
    env_dir = _env_dir(identity)
    return {
        "env_dir": str(env_dir),
        "env_python": str(_is_py(env_dir)),
        "exists": env_dir.exists(),
        "current": _env_is_current(env_dir),
        "identity": identity,
        "toolkit_wheel_url": _RKNN_FULL_WHEELS.get(
            (identity["arch"], identity["python_tag"])
        ),
    }
