import logging
import threading

_STANDARD_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def repair_standard_log_levels() -> None:
    name_to_level = logging._nameToLevel
    for name, level in _STANDARD_LEVEL_NAMES.items():
        if name_to_level.get(name) != level:
            logging.addLevelName(level, name)

_torch_lock = threading.Lock()
_torch_imported = False

def ensure_torch_imported() -> None:
    global _torch_imported
    if _torch_imported:
        return
    with _torch_lock:
        if _torch_imported:
            return
        repair_standard_log_levels()
        import torch  # noqa: F401

        _torch_imported = True

def import_rknnlite():
    repair_standard_log_levels()
    from rknnlite.api import RKNNLite

    repair_standard_log_levels()
    return RKNNLite
