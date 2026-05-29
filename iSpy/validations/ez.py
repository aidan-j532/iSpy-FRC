import os
import sys
import unittest
from pathlib import Path

def unit_tests(verbosity: int = 2) -> bool:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    test_dir = str(Path(__file__).parent)
    loader   = unittest.TestLoader()
    suite    = loader.discover(start_dir=test_dir, pattern="unit_tests.py")
    runner   = unittest.TextTestRunner(verbosity=verbosity)
    result   = runner.run(suite)
    return result.wasSuccessful()

def main() -> int:
    return 0 if unit_tests() else 1

if __name__ == "__main__":
    raise SystemExit(main())