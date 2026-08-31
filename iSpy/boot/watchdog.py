import subprocess
import time
import logging
import sys

logger = logging.getLogger(__name__)

if len(sys.argv) < 2:
    print("Usage: python watchdog.py <script.py>")
    sys.exit(1)

script = sys.argv[1]

MAX_RESTARTS = 5
restarts = 0

while restarts < MAX_RESTARTS:
    logger.info(f"Starting {script}... (attempt {restarts + 1}/{MAX_RESTARTS})")
    result = subprocess.run([sys.executable, script])

    if result.returncode == 0:
        logger.info("Script exited cleanly, stopping watchdog.")
        break

    logger.warning(f"Script crashed (code {result.returncode}), restarting in 5s...")
    restarts += 1
    if restarts < MAX_RESTARTS:
        time.sleep(5)