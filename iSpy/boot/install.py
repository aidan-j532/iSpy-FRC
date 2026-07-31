import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_service import setup

if __name__ == "__main__":
    setup("watchdog.py iSpy/boot/service_daemon.py")