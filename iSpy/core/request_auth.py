
from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Mapping

ADMIN_TOKEN_ENV = "ISPY_ADMIN_TOKEN"
ADMIN_TOKEN_HEADER = "X-iSpy-Admin-Token"

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}

_LAN_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

def _normalize(remote_addr: str | None) -> str:
    value = (remote_addr or "").strip()
    if value.startswith("[") and "]" in value:
        value = value[1 : value.index("]")]
    if "%" in value:
        value = value.split("%", 1)[0]
    return value

def is_loopback(remote_addr: str | None) -> bool:
    value = _normalize(remote_addr)
    if value.lower() in _LOOPBACK_NAMES:
        return True
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback

def is_lan_address(remote_addr: str | None) -> bool:
    value = _normalize(remote_addr)
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return any(
        net.version == addr.version and addr in net for net in _LAN_NETWORKS
    )

def provided_admin_token(headers: Mapping | None) -> str:
    if not headers:
        return ""
    return str(headers.get(ADMIN_TOKEN_HEADER, "") or "")

def valid_admin_token(headers: Mapping | None) -> bool:
    expected = os.environ.get(ADMIN_TOKEN_ENV)
    if not expected:
        return False
    provided = provided_admin_token(headers)
    return bool(provided) and hmac.compare_digest(provided, expected)

def restart_allowed(
    remote_addr: str | None,
    headers: Mapping | None = None,
    allow_remote: bool = True,
) -> bool:
    if is_loopback(remote_addr):
        return True
    if valid_admin_token(headers):
        return True
    return bool(allow_remote and is_lan_address(remote_addr))

def remote_denied_message(allow_remote: bool) -> str:
    if allow_remote:
        return (
            "Restarting iSpy is allowed from the local network only. "
            "Set ISPY_ADMIN_TOKEN to allow other remote callers."
        )
    return (
        "Remote restart is disabled. Use localhost, enable "
        "allow_remote_restart, or set ISPY_ADMIN_TOKEN."
    )
