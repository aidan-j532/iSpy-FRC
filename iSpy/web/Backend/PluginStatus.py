import logging
import re
from flask import jsonify, render_template, request
from pathlib import Path
from iSpy.web.Backend.WebModule import WebModule
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase, FrameProcessorBase, VisionBase

import iSpy.plugins as _plugins_pkg

logger = logging.getLogger(__name__)

_PLUGIN_ROOT = Path(_plugins_pkg.__file__).resolve().parent

# Vision pipelines use a different loading strategy than the other plugin
# types: they are imported directly and registered in a static PIPELINES dict
# (iSpy/vision/pipelines/__init__.py), not discovered by scanning a directory
# with load_plugins(). The _TYPE_MAP entry below makes them visible in /addons
# as read-only built-ins: they are listed from the static registry, never
# toggled/uploaded/created/deleted from the web UI (config selects one per
# camera via the 'pipeline' key instead).
_VISION_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent / "vision" / "pipelines"

_TYPE_MAP = {
    "tracker": ("trackers", TrackerBase, "TrackerBase", "update"),
    "utility": ("utilities", UtilityBase, "UtilityBase", "update"),
    "frame_processor": ("frame_processors", FrameProcessorBase, "FrameProcessorBase", "process"),
    "vision_pipeline": ("pipelines", VisionBase, "VisionBase", "run"),
}

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_ADDON_TYPES_FROM_PTYPE = {
    "tracker": "trackers", "utility": "utilities",
    "frame_processor": "frame_processors",
}


def _coerce_setting_value(value, defn: dict):
    """Validate + coerce one add-on setting against its schema definition.
    Raises ValueError with a human-readable message on bad input."""
    if not isinstance(defn, dict):
        return value
    stype = defn.get("type")
    if stype == "number":
        if isinstance(value, bool):
            raise ValueError(f"'{defn.get('label', 'value')}' must be a number")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'{defn.get('label', 'value')}' must be a number, got {value!r}"
            ) from None
        # numbers stay ints when the JSON payload was an int
        return int(num) if isinstance(value, int) and not isinstance(value, bool) else num
    if stype == "toggle":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if stype == "text":
        return str(value)
    return value


def _build_vision_pipeline_payloads():
    from iSpy.vision.pipelines import get_pipeline_classes
    pipelines = []
    try:
        vision_classes = get_pipeline_classes()
    except Exception:
        logger.exception("Failed to load vision pipelines")
        return pipelines

    for name, cls in sorted(vision_classes.items()):
        try:
            schema = cls.config_schema()
        except Exception:
            logger.warning("Failed to load config schema for vision pipeline '%s'", name)
            continue
        if schema is None:
            schema = {}
        payload = {
            "name": name,
            "class_name": cls.__name__,
            "config_schema": schema,
            "show_common_fields": bool(getattr(cls, "show_common_fields", lambda: True)()),
        }
        if hasattr(cls, "recommended_format"):
            try:
                payload["recommended_format"] = cls.recommended_format()
            except Exception:
                pass
        pipelines.append(payload)
    return pipelines


class PluginStatusModule(WebModule):
    plugin_name = "plugin_status"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/addons", "addons_page", lambda: render_template("addons.html"))
        # Back-compat alias
        flask_app.add_url_rule("/plugins", "plugins_page", lambda: render_template("addons.html"))

        flask_app.add_url_rule("/api/plugins/status", "api_plugins_status", self._status)
        flask_app.add_url_rule("/api/plugins/available", "api_plugins_available", self._available)
        flask_app.add_url_rule("/api/plugins/toggle", "api_plugins_toggle", self._toggle, methods=["POST"])
        flask_app.add_url_rule("/api/plugins/settings", "api_plugins_settings", self._save_settings, methods=["POST"])
        flask_app.add_url_rule("/api/plugins/upload", "api_plugins_upload", self._upload, methods=["POST"])
        flask_app.add_url_rule("/api/plugins/create", "api_plugins_create", self._create, methods=["POST"])
        flask_app.add_url_rule("/api/plugins/<ptype>/<name>", "api_plugins_delete", self._delete, methods=["DELETE"])
        flask_app.add_url_rule("/api/plugins/<ptype>/<name>/source", "api_plugins_source", self._source, methods=["GET"])

    # ---------- read ----------

    def _status(self):
        vision = self.context.get("vision_instance")
        if not vision:
            return jsonify(plugins=[])
        out = []
        for group, items in (
            ("tracker", vision.trackers),
            ("utility", vision.utilities),
            ("frame_processor", vision.frame_processors),
        ):
            for name, inst in items.items():
                out.append({
                    "name": name,
                    "type": group,
                    "status": inst.get_status() if hasattr(inst, "get_status") else "unknown",
                })
        return jsonify(plugins=out)

    def _vision_pipelines(self):
        return jsonify(pipelines=_build_vision_pipeline_payloads())

    def _available(self):
        config = self.context.get("config")

        available = []
        for ptype, (subdir, base_cls, _, _) in _TYPE_MAP.items():
            if ptype == "vision_pipeline":
                # Built-in pipelines are core code, listed from the static
                # registry instead of a directory scan - shown read-only.
                from iSpy.vision.pipelines import get_pipeline_classes
                for name, cls in sorted(get_pipeline_classes().items()):
                    available.append({
                        "name": name,
                        "type": ptype,
                        "enabled": False,
                        "builtin": True,
                        "doc": (cls.__doc__ or "").strip()[:200],
                        "filename": f"{name}.py",
                    })
                continue
            discovered = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
            for name, cls in discovered.items():
                settings = {}
                enabled = False
                if config:
                    addon_settings = config.get_addon_settings(_TYPE_MAP[ptype][0], name)
                    if addon_settings is not None:
                        enabled = True
                        settings = addon_settings
                try:
                    schema = cls.config_schema()
                except Exception:
                    logger.warning("Failed to load config schema for add-on '%s'", name)
                    schema = {}
                filename = self._filename_for(subdir, name)
                # Files under <subdir>/BuiltIn/ are iSpy's bundled add-ons:
                # they can be enabled/disabled and configured, but not
                # deleted, and are labelled as built-in in the UI.
                is_builtin = bool(filename and filename.startswith("BuiltIn/"))
                available.append({
                    "name": name,
                    "type": ptype,
                    "enabled": enabled,
                    "builtin": is_builtin,
                    "doc": (cls.__doc__ or "").strip()[:200],
                    "filename": filename,
                    "config_schema": schema if isinstance(schema, dict) else {},
                    "settings": settings,
                })
        return jsonify(available=available)

    def _filename_for(self, subdir: str, plugin_name: str) -> str | None:
        # Best-effort: scan for the file whose plugin_name matches, so the
        # UI can offer delete/edit without re-parsing every module itself.
        base_cls_map = {v[0]: v[1] for v in _TYPE_MAP.values()}
        base_cls = base_cls_map.get(subdir)
        if not base_cls:
            return None
        d = _PLUGIN_ROOT / subdir
        for path in sorted(d.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                text = path.read_text(errors="ignore")
                if f'"{plugin_name}"' in text or f"'{plugin_name}'" in text:
                    return str(path.relative_to(d))
            except Exception:
                continue
        return None

    def _source(self, ptype, name):
        info = _TYPE_MAP.get(ptype)
        if not info:
            return jsonify(error="Unknown addon type"), 400
        if ptype == "vision_pipeline":
            path = (_VISION_PIPELINE_DIR / f"{name}.py").resolve()
            try:
                path.relative_to(_VISION_PIPELINE_DIR)
            except ValueError:
                return jsonify(error="Invalid filename"), 400
            if not path.exists():
                return jsonify(error="Not found"), 404
            return jsonify(source=path.read_text(errors="ignore"), filename=path.name)
        subdir = info[0]
        # Resolve the real file by plugin name (handles custom add-ons in
        # <subdir>/ and bundled ones under <subdir>/BuiltIn/ uniformly).
        rel = self._filename_for(subdir, name)
        if not rel:
            return jsonify(error="Not found"), 404
        base = (_PLUGIN_ROOT / subdir).resolve()
        path = (base / rel).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            return jsonify(error="Invalid filename"), 400
        if not path.exists():
            return jsonify(error="Not found"), 404
        return jsonify(source=path.read_text(errors="ignore"), filename=rel)

    # ---------- toggle (enable/disable in config) ----------

    def _toggle(self):
        data = request.get_json(force=True)
        name = data.get("name")
        plugin_type = data.get("type")
        enable = data.get("enable", True)

        if not name or plugin_type not in _TYPE_MAP:
            return jsonify(error="Missing/invalid name or type"), 400

        if plugin_type == "vision_pipeline":
            return jsonify(
                error="Vision pipelines are selected per camera in Camera "
                      "Settings - they are not toggled here."
            ), 400

        subdir, base_cls, _, _ = _TYPE_MAP[plugin_type]
        discovered = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
        if name not in discovered:
            return jsonify(error=f"Unknown {plugin_type} addon: '{name}'"), 404

        config = self.context.get("config")
        if not config:
            return jsonify(error="No config available"), 500

        # plugins.<type> is a dict of enabled add-on -> settings; presence IS
        # the enabled state, so toggling adds/removes the entry.
        config_type = {"tracker": "trackers", "utility": "utilities",
                       "frame_processor": "frame_processors"}[plugin_type]

        if enable:
            # Write the add-on's schema defaults into the config entry so the
            # settings are visible in Config/config.json immediately (not an
            # empty {} that only gets values after a manual save). Explicit
            # user settings are preserved if the entry already has them.
            defaults = {}
            try:
                schema = discovered[name].config_schema() or {}
            except Exception:
                logger.warning("Failed to load config schema for add-on '%s'", name)
                schema = {}
            for _key, _defn in schema.items():
                if isinstance(_defn, dict) and "default" in _defn:
                    defaults[_key] = _defn["default"]
            config.enable_addon(config_type, name, settings=defaults, save=False)
        else:
            config.disable_addon(config_type, name, save=False)
        config.save()

        return jsonify(success=True, enabled=config.is_addon_enabled(config_type, name),
                       needs_restart=True)

    # ---------- settings (edit an enabled add-on's settings) ----------

    def _save_settings(self):
        data = request.get_json(force=True)
        name = data.get("name")
        plugin_type = data.get("type")
        settings = data.get("settings") or {}

        if not name or plugin_type not in _TYPE_MAP:
            return jsonify(error="Missing/invalid name or type"), 400
        if plugin_type == "vision_pipeline":
            return jsonify(
                error="Vision pipelines are configured per camera - their "
                      "settings are not edited here."
            ), 400
        if not isinstance(settings, dict):
            return jsonify(error="settings must be a JSON object"), 400

        subdir, base_cls, _, _ = _TYPE_MAP[plugin_type]
        discovered = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
        cls = discovered.get(name)
        if cls is None:
            return jsonify(error=f"Unknown {plugin_type} addon: '{name}'"), 404

        config = self.context.get("config")
        if not config:
            return jsonify(error="No config available"), 500

        config_type = {"tracker": "trackers", "utility": "utilities",
                       "frame_processor": "frame_processors"}[plugin_type]
        if not config.is_addon_enabled(config_type, name):
            return jsonify(
                error=f"'{name}' is not enabled - enable it first, then edit "
                      f"its settings."
            ), 409

        # Validate + coerce against the add-on's schema so bad values never
        # reach the config. Unknown keys are rejected - this config is
        # forward-only and typo-proof.
        try:
            schema = cls.config_schema() or {}
        except Exception:
            schema = {}
        clean = {}
        for key, value in settings.items():
            defn = schema.get(key)
            if defn is None:
                return jsonify(error=f"Unknown setting '{key}' for add-on '{name}'"), 400
            try:
                clean[key] = _coerce_setting_value(value, defn)
            except ValueError as e:
                return jsonify(error=str(e)), 400

        config.update_addon_settings(config_type, name, clean, save=False)
        config.save()

        return jsonify(success=True, settings=config.get_addon_settings(config_type, name),
                       needs_restart=True)

    # ---------- create / upload / delete ----------

    def _resolve_safe_path(self, subdir: str, filename: str) -> Path | None:
        base = (_PLUGIN_ROOT / subdir).resolve()
        # never touch BuiltIn/ (reserved for iSpy's own bundled add-ons)
        stem = Path(filename).stem
        if not _NAME_RE.match(stem):
            return None
        target = (base / f"{stem}.py").resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target

    def _validate_addon_source(self, ptype: str, code: str, expected_class_name: str | None) -> str | None:
        """Returns an error string, or None if OK. Cheap static checks only -
        we don't execute untrusted code here."""
        subdir, base_cls, base_name, _ = _TYPE_MAP[ptype]
        if base_name not in code:
            return f"Add-on must subclass {base_name} (import it from iSpy.plugins.bases)."
        if "plugin_name" not in code:
            return "Add-on class must define a 'plugin_name' class attribute."
        if len(code) > 200_000:
            return "File too large."
        return None

    def _create(self):
        data = request.get_json(force=True) or {}
        ptype = data.get("type")
        code = data.get("code", "")
        filename = data.get("filename") or data.get("name")

        if ptype not in _TYPE_MAP:
            return jsonify(error="type must be tracker, utility, or frame_processor"), 400
        if ptype == "vision_pipeline":
            return jsonify(
                error="Vision pipelines are built into iSpy and selected per "
                      "camera - user-authored pipelines are not supported."
            ), 400
        if not filename:
            return jsonify(error="filename/name required"), 400
        if not code.strip():
            return jsonify(error="code is empty"), 400

        err = self._validate_addon_source(ptype, code, None)
        if err:
            return jsonify(error=err), 400

        subdir = _TYPE_MAP[ptype][0]
        path = self._resolve_safe_path(subdir, filename)
        if path is None:
            return jsonify(error="Invalid filename"), 400
        if path.exists():
            return jsonify(error=f"'{path.name}' already exists"), 409

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

        return jsonify(
            success=True,
            filename=path.name,
            note="Add-on saved. Restart vision to detect and enable it.",
        )

    def _upload(self):
        ptype = request.form.get("type") or request.args.get("type")
        f = request.files.get("file")

        if ptype not in _TYPE_MAP:
            return jsonify(error="type must be tracker, utility, or frame_processor"), 400
        if ptype == "vision_pipeline":
            return jsonify(
                error="Vision pipelines are built into iSpy and selected per "
                      "camera - user-authored pipelines are not supported."
            ), 400
        if not f or not f.filename.endswith(".py"):
            return jsonify(error="Upload a .py file"), 400

        code = f.read().decode("utf-8", errors="replace")
        err = self._validate_addon_source(ptype, code, None)
        if err:
            return jsonify(error=err), 400

        subdir = _TYPE_MAP[ptype][0]
        path = self._resolve_safe_path(subdir, f.filename)
        if path is None:
            return jsonify(error="Invalid filename"), 400
        if path.exists():
            return jsonify(error=f"'{path.name}' already exists. Delete it first or rename your file."), 409

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

        return jsonify(
            success=True,
            filename=path.name,
            note="Add-on uploaded. Restart vision to detect and enable it.",
        )

    def _delete(self, ptype, name):
        if ptype not in _TYPE_MAP:
            return jsonify(error="Unknown addon type"), 400
        if ptype == "vision_pipeline":
            return jsonify(error="Cannot delete a built-in add-on."), 403

        subdir, base_cls, _, _ = _TYPE_MAP[ptype]
        path = self._resolve_safe_path(subdir, name)
        if path is None or not path.exists():
            return jsonify(error="Not found"), 404
        if "BuiltIn" in path.parts:
            return jsonify(error="Cannot delete a built-in add-on."), 403

        # If it's currently enabled in config, disable it first so we don't
        # leave a dangling reference the loader can't resolve.
        config = self.context.get("config")
        discovered = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
        plugin_name = None
        for pname, cls in discovered.items():
            try:
                if Path(cls.__module__.replace(".", "/") + ".py").name == path.name:
                    plugin_name = pname
                    break
            except Exception:
                continue

        if config and plugin_name:
            config_key = _ADDON_TYPES_FROM_PTYPE[ptype]
            config.disable_addon(config_key, plugin_name, save=False)
            config.save()

        path.unlink()
        return jsonify(success=True, note="Add-on deleted. Restart vision to apply.")