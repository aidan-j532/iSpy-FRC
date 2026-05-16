import subprocess
import time
import logging
import sys

logger = logging.getLogger(__name__)

if len(sys.argv) < 2:
    print("Usage: python watchdog.py <script.py>")
    sys.exit(1)

script = sys.argv[1]

while True:
    logger.info(f"Starting {script}...")
    result = subprocess.run([sys.executable, script])

    if result.returncode == 0:
        logger.info("Script exited cleanly, stopping watchdog.")
        break

    logger.warning(f"Script crashed (code {result.returncode}), restarting in 5s...")
    time.sleep(5)