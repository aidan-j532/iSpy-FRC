"""Standalone web service for iSpy.

This runs the Flask web interface independently of the vision loop.
It can start/stop/restart the vision process via subprocess management.
The web UI is always available, even when iSpy vision isn't running.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    repo_root = Path.cwd()
    config_path = repo_root / "Config" / "config.json"

    if not config_path.exists():
        print(f"Error: config not found at {config_path}")
        print("Run 'ispy-boot' first to set up iSpy.")
        sys.exit(1)

    from iSpy.config.iSpyConfig import iSpyConfig
    from iSpy.web.process_manager import VisionProcessManager
    from iSpy.web.Backend.WebApp import iSpyWebApp

    config = iSpyConfig(str(config_path))

    process_manager = VisionProcessManager()

    web_app = iSpyWebApp(
        cameras=[],
        config=config,
        standalone=True,
        process_manager=process_manager,
    )

    port = config.get("web_port", 5000)
    host = "0.0.0.0"

    print(f"iSpy web interface: http://localhost:{port}/")

    try:
        from waitress import serve
        logger.info("Starting iSpy web service with Waitress on http://%s:%d", host, port)
        serve(web_app.flask_app, host=host, port=port, threads=4)
    except ImportError:
        logger.info("Waitress not available, using Flask dev server (not recommended for production)")
        web_app.flask_app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
