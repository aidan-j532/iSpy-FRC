"""pip-installable `ispy` command line entry point (`setup` / `start`).

Thin wrappers around iSpy.boot.boot.on_boot() so the pre-existing
`python -m iSpy.boot.boot [-f]` invocation keeps working exactly as-is for
anyone with it in a systemd unit, install script, or docs. Helpers beyond
setup/start (status, stop, ...) are intentionally out of scope - the service
daemon already exposes those over HTTP (`/service/*`).
"""

import argparse
import logging
import os
import sys

from iSpy.boot import boot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ispy",
        description="iSpy control surface (trivial wrappers around the boot sequence).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -s/--service and -w/--wait come from boot.add_boot_arguments() so they
    # can never drift from `python -m iSpy.boot.boot`.
    setup = subparsers.add_parser(
        "setup",
        help="Fresh first-time install (equivalent to `python -m iSpy.boot.boot -f`)",
    )
    boot.add_boot_arguments(setup)

    start = subparsers.add_parser(
        "start",
        help="Normal boot against the existing Config/config.json (equivalent to `python -m iSpy.boot.boot`)",
    )
    boot.add_boot_arguments(start)

    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    boot.on_boot(
        install_service=args.service,
        fresh=args.command == "setup",
        wait=args.wait,
    )

    # mirror boot.py's main(): flush everything and hard-exit to dodge
    # RKNN/OpenCV native-extension segfaults during interpreter teardown on ARM
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    return 0


if __name__ == "__main__":
    main()