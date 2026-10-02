"""Tests for `ispy` (iSpy/cli.py) - the wrapper the install scripts tell people
to run.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from iSpy import cli


class StartRunsTheVisionLoopTests(unittest.TestCase):
    def _run(self, argv):
        """Run the CLI with both the boot step and the vision loop stubbed."""
        loop = types.ModuleType("iSpy.core.game_loop")
        calls = []
        loop.main = lambda config=None: calls.append(config)

        exited = []
        with mock.patch.object(cli.boot, "on_boot", return_value="CONFIG") as boot_step:
            with mock.patch.object(cli.os, "_exit", side_effect=exited.append):
                with mock.patch.dict(sys.modules, {"iSpy.core.game_loop": loop}):
                    cli.main(argv)
        return boot_step, calls, exited

    def test_start_runs_the_vision_loop(self):
        # "ispy start" booted, flushed, and exited - the install scripts point
        # at it, so it has to be the whole thing
        _, calls, _ = self._run(["start"])
        self.assertEqual(calls, ["CONFIG"], "vision never ran")

    def test_setup_runs_the_vision_loop_too(self):
        # setup is documented as "fresh first-time install", which is boot.py's
        # main() - that boots and then runs the loop
        boot_step, calls, _ = self._run(["setup"])
        self.assertTrue(boot_step.call_args.kwargs["fresh"])
        self.assertEqual(calls, ["CONFIG"], "vision never ran")

    def test_the_boot_config_is_handed_to_the_loop(self):
        # passing it avoids the loop re-reading a config.json from the cwd,
        # which is not necessarily the one boot just prepared
        _, calls, _ = self._run(["start"])
        self.assertEqual(calls, ["CONFIG"])

    def test_the_process_exits_hard_after_the_loop_returns(self):
        # the vision loop normally never returns; when it does (an error that
        # escapes), boot.py's flush-and-_exit is still what we want
        _, _, exited = self._run(["start"])
        self.assertEqual(exited, [0])

    def test_game_loop_is_not_imported_at_module_scope(self):
        # importing iSpy.core.game_loop configures logging and pulls in the
        # whole vision stack (torch and all), which "ispy setup" has no use for
        source = Path(cli.__file__).read_text(encoding="utf-8")
        module_level = source.split("\ndef main(", 1)[0]
        self.assertNotIn("game_loop", module_level)
        self.assertIn(
            "from iSpy.core.game_loop import main", source.split("\ndef main(", 1)[1]
        )

    def test_service_flag_is_passed_through(self):
        boot_step, _, _ = self._run(["start", "--service"])
        self.assertTrue(boot_step.call_args.kwargs["install_service"])


class ParserTests(unittest.TestCase):
    def test_a_command_is_required(self):
        with self.assertRaises(SystemExit):
            cli._build_parser().parse_args([])

    def test_start_does_not_pass_fresh(self):
        args = cli._build_parser().parse_args(["start"])
        self.assertFalse(args.command == "setup")
        self.assertFalse(args.wait)


if __name__ == "__main__":
    unittest.main()