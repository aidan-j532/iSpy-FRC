"""Cheap, stdlib-only process supervisor for iSpy.

Importing this package must not pull in torch, OpenCV, Flask, or the boot
module.  The CLI deliberately imports it before doing any heavy work so
``ispy start`` can hold port 5000 even while the vision process is booting or
has crashed.
"""

from iSpy.supervisor.config import allow_remote_restart, supervisor_disabled
from iSpy.supervisor.messages import load_messages
from iSpy.supervisor.process import Supervisor, run_supervisor

__all__ = [
    "Supervisor",
    "allow_remote_restart",
    "load_messages",
    "run_supervisor",
    "supervisor_disabled",
]
