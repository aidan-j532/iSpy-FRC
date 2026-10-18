"""Process-wide offline mode policy.

Offline mode blocks traffic to the *public internet* only. Everything that
talks to the robot LAN (NetworkTables, the dashboard, loopback, mDNS, the
Tello drone) keeps working, because field boards sit on an isolated LAN and
the whole point of the mode is to skip the online-only niceties without
taking the robot link down with them.

Two ways in:

- the top-level ``offline_mode`` config key (a normal setting, saved to
  config.json), and
- Game Mode, which forces offline for its one-shot run without persisting
  anything (see boot._consume_game_mode_web_off).

``ISPY_OFFLINE=1`` in the environment mirrors the state into every
subprocess (pip, the RKNN convert worker, the yolo-world reparam worker),
so no child can quietly start downloading while the parent is offline.
"""

import ipaddress
import logging
import os
import socket
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OFFLINE_ENV = "ISPY_OFFLINE"

# set while offline so huggingface_hub / transformers fail fast instead of
# hanging on a retry storm, and cleared again when offline mode turns off (a
# stray value must not keep HF offline afterwards).
_HF_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class OfflineError(RuntimeError):
    """Offline mode blocked an attempt to reach the internet."""


@dataclass
class NetworkRequirement:
    """One internet need an add-on declares.

    ``critical=True`` means the add-on cannot do its job without it and is
    skipped while offline; ``critical=False`` means only some optional
    feature goes away and the add-on still loads.
    """

    name: str
    reason: str
    critical: bool = True


# ---------------------------------------------------------------------------
# state


_lock = threading.RLock()
_offline = os.environ.get(OFFLINE_ENV) == "1"
_reason = "environment" if _offline else ""

# what got skipped / degraded this process; the web UI reads this through
# get_report(). deduplicated on (bucket, kind, name, feature).
_events: dict[str, list] = {"skipped": [], "degraded": []}
_seen: set = set()


def set_offline(on: bool, reason: str = "") -> None:
    """Turn offline mode on/off for this process (and its children).

    Idempotent - flipping it twice does not double-log or leave half of the
    guard installed.
    """
    global _offline, _reason
    on = bool(on)
    with _lock:
        changed = on != _offline
        _offline = on
        if changed:
            _reason = str(reason or "config") if on else ""
            # never leak skips/degradeds across a flip - the next run rebuilds
            # its own report from the add-ons it actually loaded
            _events["skipped"].clear()
            _events["degraded"].clear()
            _seen.clear()

    if on:
        os.environ[OFFLINE_ENV] = "1"
        for key in _HF_ENV_KEYS:
            os.environ[key] = "1"
        _install_socket_guard()
        if changed:
            logger.info("Offline mode ON (reason: %s)", _reason or "config")
    else:
        if os.environ.get(OFFLINE_ENV) == "1":
            os.environ.pop(OFFLINE_ENV, None)
        for key in _HF_ENV_KEYS:
            if os.environ.get(key) == "1":
                os.environ.pop(key, None)
        _remove_socket_guard()
        if changed:
            logger.info("Offline mode off")


def is_offline() -> bool:
    if os.environ.get(OFFLINE_ENV) == "1":
        return True
    with _lock:
        return _offline


def offline_reason() -> str:
    if os.environ.get(OFFLINE_ENV) == "1" and not _offline:
        # a child process inherits the flag but not the in-process reason
        return "environment"
    with _lock:
        return _reason


# ---------------------------------------------------------------------------
# report registry


def record_skip(kind: str, name: str, reason: str) -> None:
    """An add-on was not loaded because offline mode blocks it."""
    _record_event("skipped", {"kind": kind, "name": name, "reason": reason})


def record_degraded(kind: str, name: str, feature: str, reason: str) -> None:
    """An add-on loaded, but one optional online feature is disabled."""
    _record_event(
        "degraded",
        {"kind": kind, "name": name, "feature": feature, "reason": reason},
    )


def _record_event(bucket: str, entry: dict) -> None:
    key = (bucket, entry.get("kind"), entry.get("name"), entry.get("feature"))
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
        _events[bucket].append(entry)


def get_report() -> dict:
    with _lock:
        return {
            "offline": is_offline(),
            "reason": offline_reason(),
            "skipped": [dict(e) for e in _events["skipped"]],
            "degraded": [dict(e) for e in _events["degraded"]],
        }


def reset_report() -> None:
    with _lock:
        _events["skipped"].clear()
        _events["degraded"].clear()
        _seen.clear()


# ---------------------------------------------------------------------------
# address policy


#: Everything offline mode is allowed to reach: loopback, RFC1918,
#: link-local and the IPv6 private ranges. Deliberately a fixed list - the
#: spec is "the public internet is off", not "whatever is_private says".
_ALLOWED_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


def _ip_is_local(ip) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return any(
        net.version == ip.version and ip in net for net in _ALLOWED_NETWORKS
    )


def is_private_address(host) -> bool:
    """True when *host* points at this machine or the local network.

    Accepts an IP literal or a hostname; hostnames are resolved (through the
    real getaddrinfo, never the guarded one). A hostname that cannot be
    resolved is *not* private - while offline we cannot prove it is local,
    so it counts as blocked.
    """
    if not host:
        return False
    text = str(host).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    literal = text.split("%", 1)[0]  # strip an IPv6 zone id like fe80::1%eth0
    try:
        return _ip_is_local(ipaddress.ip_address(literal))
    except ValueError:
        pass

    try:
        infos = _original_getaddrinfo(text, None)
    except OSError:
        return False
    if not infos:
        return False
    resolved = []
    for info in infos:
        sockaddr = info[4]
        try:
            resolved.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    if not resolved:
        return False
    return all(_ip_is_local(ip) for ip in resolved)


# ---------------------------------------------------------------------------
# socket guard
#
# Defense in depth for the code we do not control (huggingface_hub,
# requests, urllib, pip invoked in-process). Plugins are supposed to gate
# themselves through the policy, but a friendly `requests.get` in a third
# party helper would otherwise sail straight out of the LAN.

_guard_lock = threading.Lock()
_guard_installed = False

_original_getaddrinfo = socket.getaddrinfo
_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection


def _blocked_message(host) -> str:
    return (
        f"Offline mode blocked a connection to {host!r} - only loopback and "
        "LAN addresses are reachable while offline."
    )


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) and address else None
    if host is not None and not is_private_address(host):
        raise OfflineError(_blocked_message(host))
    return _original_connect(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) and address else None
    if host is not None and not is_private_address(host):
        raise OfflineError(_blocked_message(host))
    return _original_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    infos = _original_getaddrinfo(host, port, *args, **kwargs)
    if host is None or host == "":
        # passive/bind-all lookup, not an outbound destination
        return infos
    if is_private_address(host):
        return infos
    raise OfflineError(_blocked_message(host))


def _install_socket_guard() -> None:
    global _guard_installed
    with _guard_lock:
        if _guard_installed:
            return
        socket.socket.connect = _guarded_connect
        socket.create_connection = _guarded_create_connection
        socket.getaddrinfo = _guarded_getaddrinfo
        _guard_installed = True
        logger.debug("Offline socket guard installed.")


def _remove_socket_guard() -> None:
    global _guard_installed
    with _guard_lock:
        if not _guard_installed:
            return
        socket.socket.connect = _original_connect
        socket.create_connection = _original_create_connection
        socket.getaddrinfo = _original_getaddrinfo
        _guard_installed = False
        logger.debug("Offline socket guard removed.")


# ---------------------------------------------------------------------------
# error classification for try_online() callers


_network_exceptions: tuple | None = None


def network_exceptions() -> tuple:
    """Exception types that mean 'the network is missing or flaky'.

    OfflineError plus the connection-level errors; requests/urllib wrap
    socket errors in their own types, so add those when importable.
    """
    global _network_exceptions
    if _network_exceptions is not None:
        return _network_exceptions

    excs: list[type] = [
        OfflineError,
        ConnectionError,
        TimeoutError,
        socket.gaierror,
    ]
    try:
        import urllib.error

        excs.append(urllib.error.URLError)
    except Exception:
        pass
    try:
        import requests

        excs.append(requests.RequestException)
    except Exception:
        pass
    _network_exceptions = tuple(excs)
    return _network_exceptions


# a child process (the RKNN convert worker, the yolo-world reparam worker,
# pip itself) inherits ISPY_OFFLINE=1 - the flag alone is not enough, so arm
# the guard at import time too. the module-level _offline above already
# picked the environment up.
if os.environ.get(OFFLINE_ENV) == "1":
    _install_socket_guard()
