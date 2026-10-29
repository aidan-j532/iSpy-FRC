from __future__ import annotations

import argparse
import sys

from iSpy.supervisor.process import Supervisor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m iSpy.supervisor",
        description="Always-on iSpy status page and vision process supervisor.",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Web port to own (default: 5000)"
    )
    parser.add_argument(
        "--child-arg",
        action="append",
        default=[],
        help="Argument passed to `ispy _vision` (repeatable)",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the vision child immediately instead of waiting on the idle page",
    )
    args = parser.parse_args(argv)
    supervisor = Supervisor(
        child_args=args.child_arg,
        port=args.port,
        autostart=args.start,
    )
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
