"""First-boot service: creates Config/config.json if missing, then exits.

Run by ispy-first-boot.service (Type=oneshot, RemainAfterExit=yes) on every
boot.  Idempotent - if Config/config.json already exists this is a no-op.
Exit non-zero on failure so systemd reports the error and skips starting
ispy.service.
"""
import logging
import sys
from pathlib import Path

from iSpy.boot.boot import on_boot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

_PROJECT_ROOT = Path.cwd().resolve()
_CONFIG_PATH = _PROJECT_ROOT / "Config" / "config.json"


def main() -> None:
    if _CONFIG_PATH.exists():
        logger.info("first_boot: Config/config.json already exists, nothing to do.")
        return

    logger.info("first_boot: Config/config.json not found — running fresh install.")
    on_boot(install_service=False, fresh=True, wait=False)
    logger.info("first_boot: fresh install complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("first_boot: failed: %s", exc, exc_info=True)
        sys.exit(1)
