import os
import sys
from pathlib import Path
from logging import getLogger, WARNING
import unittest
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch


def unit_tests(verbosity: int = 2) -> bool:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Save current log levels and set to WARNING to suppress informational output
    root_logger = getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(WARNING)

    try:
        logger = getLogger(__name__)
        prev_logger_level = logger.level
        logger.setLevel(WARNING)

        test_dir = str(Path(__file__).parent)
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=test_dir, pattern="unit_tests.py")
        with open(os.devnull, "w") as devnull:
            runner = unittest.TextTestRunner(verbosity=verbosity, stream=devnull)
            result = runner.run(suite)
        return result.wasSuccessful()
    finally:
        # Restore original log levels
        root_logger.setLevel(prev_level)
        logger.setLevel(prev_logger_level)


def main() -> int:
    return 0 if unit_tests() else 1

if __name__ == "__main__":
    raise SystemExit(main())