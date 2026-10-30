"""Tests for `ispy` (iSpy/cli.py) - the wrapper the install scripts tell people
to run.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from iSpy import cli


class StartLaunchesSupervisorTests(unittest.TestCase):
    def test_start_launches_the_supervisor_not_the_loop(self):
        # "ispy start" owns port 5000 itself now, then launches `_vision` as a
        # child. It must not import boot/torch in the supervisor process.
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=False
        ):
            with mock.patch(
                "iSpy.supervisor.process.run_supervisor", return_value=0
            ) as supervisor:
                self.assertEqual(cli.main(["start"]), 0)
        supervisor.assert_called_once_with([], autostart=True)

    def test_service_flag_is_passed_to_the_child(self):
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=False
        ):
            with mock.patch(
                "iSpy.supervisor.process.run_supervisor", return_value=0
            ) as supervisor:
                cli.main(["start", "--service"])
        supervisor.assert_called_once_with(["-s"], autostart=True)

    def test_replay_flags_are_passed_to_the_child(self):
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=False
        ):
            with mock.patch(
                "iSpy.supervisor.process.run_supervisor", return_value=0
            ) as supervisor:
                cli.main(
                    ["start", "--replay", "session", "--replay-speed", "2.0"]
                )
        supervisor.assert_called_once_with(
            ["--replay", "session", "--replay-speed", "2.0"], autostart=True
        )

    def test_no_supervisor_falls_back_to_the_inline_vision_loop(self):
        loop = types.ModuleType("iSpy.core.game_loop")
        calls = []
        loop.main = lambda config=None: calls.append(config)

        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=True
        ):
            with mock.patch("iSpy.boot.boot.on_boot", return_value="CONFIG"):
                with mock.patch.object(cli.os, "_exit"):
                    with mock.patch(
                        "iSpy.core.control_channel.update_service_state"
                    ):
                        with mock.patch.dict(
                            sys.modules, {"iSpy.core.game_loop": loop}
                        ):
                            cli.main(["start"])
        self.assertEqual(calls, ["CONFIG"])


class VisionChildTests(unittest.TestCase):
    def _run_vision(self, argv):
        loop = types.ModuleType("iSpy.core.game_loop")
        calls = []
        loop.main = lambda config=None: calls.append(config)
        exited = []
        with mock.patch("iSpy.boot.boot.on_boot", return_value="CONFIG") as boot_step:
            with mock.patch.object(cli.os, "_exit", side_effect=exited.append):
                with mock.patch(
                    "iSpy.core.control_channel.update_service_state"
                ):
                    with mock.patch.dict(
                        sys.modules, {"iSpy.core.game_loop": loop}
                    ):
                        cli.main(argv)
        return boot_step, calls, exited

    def test_vision_runs_the_loop(self):
        _, calls, _ = self._run_vision(["_vision"])
        self.assertEqual(calls, ["CONFIG"], "vision never ran")

    def test_the_boot_config_is_handed_to_the_loop(self):
        _, calls, _ = self._run_vision(["_vision"])
        self.assertEqual(calls, ["CONFIG"])

    def test_the_process_exits_hard_after_the_loop_returns(self):
        _, _, exited = self._run_vision(["_vision"])
        self.assertEqual(exited, [0])

    def test_service_install_defers_the_systemd_start(self):
        loop = types.ModuleType("iSpy.core.game_loop")
        loop.main = lambda config=None: None
        seen = []

        def fake_boot(**kwargs):
            seen.append(os.environ.get("ISPY_DEFER_SERVICE_START"))
            return "CONFIG"

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ISPY_DEFER_SERVICE_START", None)
            with mock.patch(
                "iSpy.boot.boot.on_boot", side_effect=fake_boot
            ):
                with mock.patch.object(cli.os, "_exit"):
                    with mock.patch(
                        "iSpy.core.control_channel.update_service_state"
                    ):
                        with mock.patch.dict(
                            sys.modules, {"iSpy.core.game_loop": loop}
                        ):
                            cli.main(["_vision", "--service"])
            self.assertNotIn("ISPY_DEFER_SERVICE_START", os.environ)
        self.assertEqual(seen, ["1"])


class SetupTests(unittest.TestCase):
    def test_setup_does_not_run_the_vision_loop_and_never_binds_a_port(self):
        # install.sh and image chroots call `ispy setup` inline. With
        # ISPY_NO_SUPERVISOR/CI/chroot we must not even create the status
        # server, and setup must still finish installing.
        exited = []
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=True
        ):
            with mock.patch(
                "iSpy.supervisor.status_server.StatusServer"
            ) as server_cls:
                with mock.patch(
                    "iSpy.boot.boot.on_boot", return_value="CONFIG"
                ) as boot_step:
                    with mock.patch.object(
                        cli.os, "_exit", side_effect=exited.append
                    ):
                        cli.main(["setup"])
        server_cls.assert_not_called()
        self.assertTrue(boot_step.call_args.kwargs["fresh"])
        self.assertEqual(exited, [0])

    def test_setup_serves_fresh_boot_page_when_enabled(self):
        server = mock.Mock()
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=False
        ):
            with mock.patch(
                "iSpy.supervisor.status_server.StatusServer", return_value=server
            ) as server_cls:
                with mock.patch(
                    "iSpy.core.control_channel.update_supervisor_state"
                ):
                    with mock.patch(
                        "iSpy.supervisor.state.mark_supervisor_exit"
                    ):
                        with mock.patch(
                            "iSpy.boot.boot.on_boot", return_value="CONFIG"
                        ):
                            with mock.patch.object(cli.os, "_exit"):
                                cli.main(["setup"])
        server_cls.assert_called_once()
        server.start.assert_called_once()
        server.stop.assert_called_once()

    def test_setup_service_does_not_hold_port_while_installing(self):
        # The systemd unit starts at the end of setup -s. Holding 5000 for a
        # fresh-boot page would make the new supervisor fail to bind.
        with mock.patch(
            "iSpy.supervisor.config.supervisor_disabled", return_value=False
        ):
            with mock.patch(
                "iSpy.supervisor.status_server.StatusServer"
            ) as server_cls:
                with mock.patch(
                    "iSpy.boot.boot.on_boot", return_value="CONFIG"
                ):
                    with mock.patch.object(cli.os, "_exit"):
                        cli.main(["setup", "--service"])
        server_cls.assert_not_called()


class ParserTests(unittest.TestCase):
    def test_a_command_is_required(self):
        with self.assertRaises(SystemExit):
            cli._build_parser().parse_args([])

    def test_start_does_not_pass_fresh(self):
        args = cli._build_parser().parse_args(["start"])
        self.assertFalse(args.command == "setup")
        self.assertFalse(args.wait)

    def test_game_loop_is_not_imported_at_module_scope(self):
        # importing iSpy.core.game_loop configures logging and pulls in the
        # whole vision stack (torch and all), which "ispy start" has no use for
        import ast

        source = Path(cli.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_level.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_level.append(node.module or "")
        self.assertNotIn("iSpy.core.game_loop", module_level)
        self.assertIn("from iSpy.core.game_loop import main as game_loop_main", source)


if __name__ == "__main__":
    unittest.main()
