#!/usr/bin/env python3
"""find_ispy.py — Discover iSpy boards on the local network.

Listens for UDP broadcast announcements sent by iSpy's announce service
(iSpy.boot.announce) and prints the hostname, IP, and web port of each
board it hears from.

This is a **last-resort fallback** for when neither mDNS
(http://ispy.local:5000) nor DHCP hostname resolution (http://ispy:5000)
works — for example on Windows pit laptops without Bonjour installed and
routers that don't publish DHCP hostnames into DNS.

The expected discovery path is still:
    http://ispy.local:5000   (mDNS)
    http://ispy:5000          (DHCP/local DNS)

Usage:
    python tools/find_ispy.py            # listen until Ctrl-C
    python tools/find_ispy.py --once     # print first board found and exit
    python tools/find_ispy.py --timeout 10   # listen for at most 10 seconds

Requirements: Python 3.10+ (stdlib only — no pip installs needed).
"""
import argparse
import json
import socket
import sys
import time

MAGIC = b"ISPY_DISCOVER:"
BROADCAST_PORT = 37429


def listen(once: bool = False, timeout: float | None = None) -> None:
    """Listen for iSpy announce broadcasts and print discoveries."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass  # Windows doesn't support SO_REUSEPORT
    sock.bind(("", BROADCAST_PORT))
    sock.settimeout(1.0)

    print(f"Listening for iSpy boards on UDP port {BROADCAST_PORT}...")
    print("Press Ctrl-C to stop.\n")

    seen: dict[str, float] = {}  # ip -> last-seen timestamp
    deadline = time.monotonic() + timeout if timeout else None

    try:
        while True:
            if deadline and time.monotonic() >= deadline:
                break
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue

            if not data.startswith(MAGIC):
                continue

            try:
                payload = json.loads(data[len(MAGIC):])
            except json.JSONDecodeError:
                continue

            ip = payload.get("ip", addr[0])
            hostname = payload.get("hostname", "unknown")
            port = payload.get("port", 5000)

            # Deduplicate within 30 seconds
            now = time.monotonic()
            if ip in seen and now - seen[ip] < 30:
                continue
            seen[ip] = now

            print(f"  Found: {hostname} @ http://{ip}:{port}")
            print(f"         also try http://{hostname}.local:{port}")
            print()

            if once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover iSpy boards on the local network via UDP broadcast.",
        epilog=(
            "This is a last-resort fallback for when mDNS "
            "(http://ispy.local:5000) and DHCP hostname resolution "
            "(http://ispy:5000) both fail.  Requires Python but no "
            "admin rights or software installs."
        ),
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print the first board found and exit.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, metavar="SECS",
        help="Stop listening after SECS seconds (default: listen forever).",
    )
    args = parser.parse_args()
    listen(once=args.once, timeout=args.timeout)


if __name__ == "__main__":
    main()
