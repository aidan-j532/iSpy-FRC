"""Standalone web service for iSpy.

This runs the Flask web interface independently of the vision loop.
It can start/stop/restart the vision process via subprocess management.
The web UI is always available, even when iSpy vision isn't running.
"""

import sys
import logging
from pathlib import Path

from iSpy.web.Backend.WebApp import iSpyWebApp
from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.web.process_manager import VisionProcessManager

logger = logging.getLogger(__name__)


def main():
    repo_root = Path.cwd()
    config_path = repo_root / "Config" / "config.json"

    if not config_path.exists():
        print(f"Error: config not found at {config_path}")
        print("Run 'ispy-boot' first to set up iSpy.")
        sys.exit(1)

    config = iSpyConfig(str(config_path))

    process_manager = VisionProcessManager()

    web_app = iSpyWebApp(
        cameras=[],
        config=config,
        standalone=True,
        process_manager=process_manager,
    )

    port = config.get("web_port", 5000)
    logger.info("Starting iSpy web service on http://0.0.0.0:%d", port)
    print(f"iSpy web interface: http://localhost:{port}/")

    web_app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
