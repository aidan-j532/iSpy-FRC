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
    # rknnlite (and friends) replace logging._nameToLevel with single-letter
    # shorthands at import time. that makes stdlib setLevel('WARNING') throw
    # "Unknown level: 'WARNING'", which torch's fx bootstrap calls during
    # import - so a clobbered table kills every torch import. re-adding the
    # canonical names is idempotent, so just re-assert them.
    name_to_level = logging._nameToLevel
    for name, level in _STANDARD_LEVEL_NAMES.items():
        if name_to_level.get(name) != level:
            logging.addLevelName(level, name)


_torch_lock = threading.Lock()
_torch_imported = False


def ensure_torch_imported() -> None:
    # scipy.stats triggers `import torch` from inside its own module init.
    # if a bg thread is mid-import of torch at that moment, scipy can grab
    # the half-initialized module: "partially initialized module 'torch' has
    # no attribute 'Tensor'". importing up front on the calling thread makes
    # every later `import torch` a no-op and kills the race.
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
    # rknnlite clobbers the logging tables on import - put them back.
    repair_standard_log_levels()
    from rknnlite.api import RKNNLite

    repair_standard_log_levels()
    return RKNNLite
