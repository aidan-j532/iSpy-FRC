"""Boot/CLI argument definitions, kept free of the vision stack.

`iSpy.cli` imports this instead of `iSpy.boot.boot`, so `ispy start` can hand
port 5000 to the supervisor before torch/OpenCV/Flask are imported.
"""

from __future__ import annotations

import argparse


def add_boot_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-s",
        "--service",
        action="store_true",
        help="Install and start the watchdog service",
    )
    parser.add_argument(
        "-w",
        "--wait",
        action="store_true",
        help="Wait for all pipelines to be ready before running vision",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help=(
            "Force the web UI on for this run (recovery from performance "
            "level 0). Nothing is saved - the configured level is unchanged."
        ),
    )
    parser.add_argument(
        "--replay",
        metavar="SESSION_OR_CLIP",
        default="",
        help="Replay a rollback session directory or a recorded video clip",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--replay-cam",
        default="",
        help="Recorded camera name to replay (defaults to the first camera)",
    )
    parser.add_argument(
        "--replay-no-loop",
        action="store_true",
        help="Stop vision when the replay reaches its end",
    )
    return parser
