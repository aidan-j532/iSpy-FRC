from flask import jsonify, render_template, request
from pathlib import Path
from iSpy.web.Backend.WebModule import WebModule
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import TrackerBase, UtilityBase

import iSpy.plugins as _plugins_pkg

_PLUGIN_ROOT = Path(_plugins_pkg.__file__).resolve().parent


class PluginStatusModule(WebModule):
    plugin_name = "plugin_status"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/plugins", "plugins_page", lambda: render_template("plugins.html"))
        flask_app.add_url_rule("/api/plugins/status", "api_plugins_status", self._status)
        flask_app.add_url_rule("/api/plugins/available", "api_plugins_available", self._available)
        flask_app.add_url_rule("/api/plugins/toggle", "api_plugins_toggle", self._toggle, methods=["POST"])

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

    def _available(self):
        tracker_classes = load_plugins(_PLUGIN_ROOT / "trackers", TrackerBase)
        utility_classes = load_plugins(_PLUGIN_ROOT / "utilities", UtilityBase)

        config = self.context.get("config")
        enabled_trackers = list(config.get_nested("plugins", "trackers", default=[])) if config else []
        enabled_utilities = list(config.get_nested("plugins", "utilities", default=[])) if config else []

        available = []
        for name, cls in tracker_classes.items():
            available.append({
                "name": name,
                "type": "tracker",
                "enabled": name in enabled_trackers,
                "doc": (cls.__doc__ or "").strip()[:200],
            })
        for name, cls in utility_classes.items():
            available.append({
                "name": name,
                "type": "utility",
                "enabled": name in enabled_utilities,
                "doc": (cls.__doc__ or "").strip()[:200],
            })

        return jsonify(available=available)

    def _toggle(self):
        data = request.get_json(force=True)
        name = data.get("name")
        plugin_type = data.get("type")
        enable = data.get("enable", True)

        if not name or not plugin_type:
            return jsonify(error="Missing name or type"), 400
        if plugin_type not in ("tracker", "utility"):
            return jsonify(error="type must be 'tracker' or 'utility'"), 400

        base_cls = TrackerBase if plugin_type == "tracker" else UtilityBase
        subdir = "trackers" if plugin_type == "tracker" else "utilities"
        discovered = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
        if name not in discovered:
            return jsonify(error=f"Unknown {plugin_type} plugin: '{name}'"), 404

        config = self.context.get("config")
        if not config:
            return jsonify(error="No config available"), 500

        config_key = "trackers" if plugin_type == "tracker" else "utilities"
        current = list(config.get_nested("plugins", config_key, default=[]))

        if enable and name not in current:
            current.append(name)
        elif not enable and name in current:
            current.remove(name)

        config.config.setdefault("plugins", {})[config_key] = current
        config.save()

        return jsonify(success=True, enabled_list=current, needs_restart=True)