"""UDP broadcast announcer for iSpy board discovery.

Beams a small UDP packet on the local subnet every ~5 seconds so that
tools/find_ispy.py (and similar clients) can find the board even when
mDNS (.local) and DHCP hostname resolution both fail.

The packet is:
    b"ISPY_DISCOVER:" + json({"hostname": <str>, "ip": <str>, "port": 5000})

This module can be run as a standalone service (``python -m iSpy.boot.announce``)
or imported and started in a background thread via :func:`start_announcer`.
"""
import json
import logging
import socket
import struct
import sys
import threading
import time as _time

logger = logging.getLogger(__name__)

MAGIC = b"ISPY_DISCOVER:"
BROADCAST_PORT = 37429          # arbitrary high port unlikely to collide
BROADCAST_INTERVAL_S = 5.0
WEB_PORT = 5000                 # iSpy Flask default


def _local_ip() -> str:
    """Return the preferred outbound IP without sending any traffic."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _hostname() -> str:
    return socket.gethostname()


def _broadcast_addr(ip: str) -> str:
    """Derive the broadcast address for a /24 subnet."""
    parts = ip.split(".")
    if len(parts) != 4:
        return "255.255.255.255"
    parts[-1] = "255"
    return ".".join(parts)


def build_payload() -> bytes:
    """Build the announce payload bytes."""
    data = {
        "hostname": _hostname(),
        "ip": _local_ip(),
        "port": WEB_PORT,
    }
    return MAGIC + json.dumps(data).encode("utf-8")


def send_announcement(sock: socket.socket, payload: bytes) -> None:
    """Send one broadcast datagram."""
    ip = _local_ip()
    bcast = _broadcast_addr(ip)
    try:
        sock.sendto(payload, (bcast, BROADCAST_PORT))
    except OSError as exc:
        logger.debug("announce: sendto %s failed: %s", bcast, exc)


def announce_loop(stop_event: threading.Event | None = None) -> None:
    """Blocking loop that broadcasts every BROADCAST_INTERVAL_S seconds."""
    stop = stop_event or threading.Event()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    payload = build_payload()
    logger.info(
        "announce: broadcasting on port %d every %.0fs — %s",
        BROADCAST_PORT, BROADCAST_INTERVAL_S, payload.decode("utf-8", errors="replace"),
    )
    try:
        while not stop.is_set():
            send_announcement(sock, payload)
            stop.wait(BROADCAST_INTERVAL_S)
    finally:
        sock.close()


def start_announcer(daemon: bool = True) -> threading.Event:
    """Start the announce loop in a background thread.

    Returns a :class:`threading.Event` that can be set to stop the announcer.
    """
    stop_event = threading.Event()
    t = threading.Thread(target=announce_loop, args=(stop_event,), daemon=daemon)
    t.start()
    return stop_event


def main() -> None:
    """Standalone entry point for running as a systemd service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("announce: starting UDP broadcast service")
    try:
        announce_loop()
    except KeyboardInterrupt:
        logger.info("announce: shutting down")


if __name__ == "__main__":
    main()
