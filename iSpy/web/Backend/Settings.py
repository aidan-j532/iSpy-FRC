import json
import copy
from flask import jsonify, render_template, request
from iSpy.web.Backend.WebModule import WebModule
from iSpy.validations.recommendations import get_structured_recommendations

_RESTART_REQUIRED_KEYS = {
    "unit", "debug_mode", "dbscan", "distance_threshold",
    "stale_threshold", "record_mode", "record_dir", "frame_sync",
    "optimize", "log_level", "use_network_tables", "network_tables_ip",
    "metrics", "plugins", "camera_configs", "device", "num_gpus",
}


class SettingsModule(WebModule):
    plugin_name = "settings"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/settings", "settings_page", lambda: render_template("settings.html"))
        flask_app.add_url_rule("/api/settings", "api_settings_get", self._get, methods=["GET"])
        flask_app.add_url_rule("/api/settings", "api_settings_post", self._post, methods=["POST"])
        flask_app.add_url_rule("/api/settings/compare", "api_settings_compare", self._compare, methods=["POST"])
        flask_app.add_url_rule("/api/settings/snapshot", "api_settings_snapshot", self._snapshot, methods=["POST"])
        flask_app.add_url_rule("/api/settings/restore", "api_settings_restore", self._restore, methods=["POST"])

    def _get(self):
        config = self.context["config"]
        return jsonify(config=config.config, defaults=config.default_config)

    def _snapshot(self):
        from iSpy.web.Backend.save_store import write
        config = self.context["config"]
        write("config_snapshot", {"config": config.config, "taken": True})
        return jsonify(success=True)

    def _restore(self):
        from iSpy.web.Backend.save_store import read, write
        snap = read("config_snapshot")
        if not snap or not snap.get("taken"):
            return jsonify(error="No snapshot available (already used, or none was taken)"), 404
        config = self.context["config"]
        config.config = snap["config"]
        config.save()
        write("config_snapshot", {"config": None, "taken": False})  # one-shot: consume it
        return jsonify(success=True, config=config.config)

    def _post(self):
        try:
            data = request.get_json(force=True)
            config = self.context["config"]

            config_data = data.get("config", data)
            frontend_restart_keys = set(data.get("restart_keys", []))

            old_config = copy.deepcopy(config.config)
            config._update_config(config_data)
            config.save()

            changed_keys = self._find_changed_keys(old_config, config.config)
            needs_restart = bool(changed_keys & (_RESTART_REQUIRED_KEYS | frontend_restart_keys))

            # Surface any critical post-save recommendations to the dashboard
            # in real time via SSE, instead of leaving them to be discovered
            # only when someone happens to visit /recommendations.
            recs = get_structured_recommendations(config.config)
            critical = [r for r in recs if r["severity"] == "critical"]
            dash = self.context.get("dashboard_module")
            if critical and dash:
                dash._push_sse({
                    "type": "config_warning",
                    "messages": [r["message"] for r in critical],
                })

            return jsonify(success=True, needs_restart=needs_restart, changed=list(changed_keys))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:
            return jsonify(error=str(e)), 500

    def _compare(self):
        try:
            data = request.get_json(force=True)
            new_data = data.get("config", {})
            config = self.context["config"]

            old_config = copy.deepcopy(config.config)
            temp = copy.deepcopy(old_config)
            config._update_config(new_data, temp)

            changed_keys = self._find_changed_keys(old_config, temp)
            needs_restart = bool(changed_keys & _RESTART_REQUIRED_KEYS)

            return jsonify(needs_restart=needs_restart, changed=list(changed_keys))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:
            return jsonify(error=str(e)), 500

    def _find_changed_keys(self, old: dict, new: dict, prefix="") -> set:
        changed = set()
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                changed |= self._find_changed_keys(old_val, new_val, full_key)
            elif old_val != new_val:
                changed.add(full_key)
        return changed