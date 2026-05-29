import importlib.util
import inspect
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_plugins(plugin_dir: Path, base_class) -> dict[str, type]:
    plugins = {}

    if not plugin_dir.exists():
        logger.warning("Plugin directory not found: %s", plugin_dir)
        return plugins

    for path in sorted(plugin_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue

        module_name = f"iSpy_plugins.{plugin_dir.name}.{path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                cls = getattr(module, attr_name)

                if not inspect.isclass(cls):
                    continue

                if cls is base_class:
                    continue

                if not issubclass(cls, base_class):
                    continue

                if not hasattr(cls, "plugin_name"):
                    raise ValueError(f"Plugin {cls.__name__} missing plugin_name")

                if not isinstance(cls.plugin_name, str):
                    raise ValueError(
                        f"Plugin {cls.__name__} plugin_name must be a string"
                    )

                name = cls.plugin_name

                if name in plugins:
                    raise ValueError(f"Duplicate plugin_name detected: {name}")

                plugins[name] = cls

                # logger.info("Loaded plugin '%s' (%s) from %s", name, cls.__name__, path)

        except Exception:
            logger.exception("Failed to load plugin from %s", path)

    return plugins
