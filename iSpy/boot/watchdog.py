import subprocess
import time
import logging
import sys
from pathlib import Path

# make the iSpy package importable when this file is run directly with a bare
# system python (no pip-installed iSpy)
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iSpy.boot._venv import read_marked_python, resolve_launch_python

logger = logging.getLogger(__name__)

MAX_RESTARTS = 5


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        print("Usage: python watchdog.py <script.py>")
        return 1

    script = argv[1]

    # Single-source the interpreter: the marker written at install time wins
    # over whatever interpreter launched this script, so the supervised child
    # always runs under the exact venv iSpy was installed into.
    launcher = sys.executable
    python = resolve_launch_python(fallback=launcher)
    marked = read_marked_python()
    if marked and launcher != marked:
        logger.warning(
            "watchdog launched under %r (prefix=%r) but the install marker "
            "says %r; children will run under the marked interpreter.",
            launcher, sys.prefix, marked,
        )
    logger.info(
        "watchdog interpreter: launcher=%r prefix=%r child_python=%r",
        launcher, sys.prefix, python,
    )

    restarts = 0
    while restarts < MAX_RESTARTS:
        logger.info(f"Starting {script}... (attempt {restarts + 1}/{MAX_RESTARTS})")
        result = subprocess.run([python, script])

        if result.returncode == 0:
            logger.info("Script exited cleanly, stopping watchdog.")
            return 0

        logger.warning(f"Script crashed (code {result.returncode}), restarting in 5s...")
        restarts += 1
        if restarts < MAX_RESTARTS:
            time.sleep(5)
    return 1


if __name__ == "__main__":
    sys.exit(main())