from flask import jsonify, request, Response
from iSpy.web.Backend.WebModule import WebModule

_HTML = """<!DOCTYPE html>
<html><head><title>iSpy Setup</title><style>
body{font-family:Arial;background:#111;color:#eee;padding:20px;max-width:600px;margin:auto}
label{display:block;margin-top:12px}
input,select{width:100%;padding:6px;margin-top:4px;background:#1a1a1a;color:#eee;border:1px solid #333}
button{padding:10px 20px;background:#2c7;color:#fff;border:none;border-radius:4px;cursor:pointer;margin-top:20px}
</style></head><body>
<h1>iSpy First-Boot Setup</h1>
<label>Camera name <input id="cam_name" value="default_cam"></label>
<label>Camera source (index or path) <input id="cam_source" value="0"></label>
<label>Subsystem <input id="cam_subsystem" value="field"></label>
<label>Unit <select id="unit"><option value="frc">FRC (meters)</option><option>meter</option><option>inch</option><option>foot</option></select></label>
<label>Use NetworkTables <select id="use_nt"><option value="false">No</option><option value="true">Yes</option></select></label>
<label>NetworkTables IP <input id="nt_ip" value="10.0.0.2"></label>
<button onclick="go()">Finish Setup</button><div id="msg"></div>
<script>
async function go(){
  const payload = {
    unit: unit.value, use_network_tables: use_nt.value==="true", network_tables_ip: nt_ip.value,
    camera_configs: {[cam_name.value]: {
      name: cam_name.value, source: isNaN(cam_source.value)?cam_source.value:Number(cam_source.value),
      subsystem: cam_subsystem.value,
      pipeline: {name: "object_detection", settings: {}},
      yaw:0, pitch:0, height:1.0, x:0, y:0,
      calibration:{distance:0, game_piece_size:0, size:0, fov:0}
    }}
  };
  const r = await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j = await r.json();
  msg.textContent = r.ok ? 'Saved. Restart iSpy to apply.' : ('Error: '+j.error);
  msg.style.color = r.ok ? '#2c7' : '#f44';
}
</script></body></html>"""


class SetupWizardModule(WebModule):
    plugin_name = "setup_wizard"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/setup", "setup_page", lambda: Response(_HTML, mimetype="text/html"))
        flask_app.add_url_rule("/api/setup", "api_setup", self._save, methods=["POST"])

    def _save(self):
        try:
            data = request.get_json(force=True) or {}
            if not data.get("camera_configs"):
                return jsonify(error="At least one camera_configs entry required"), 400
            config = self.context["config"]
            if "unit" in data:
                config.set("unit", data["unit"])
            # legacy wizard payloads had use_network_tables / network_tables_ip as
            # top-level keys. NetworkTables is now an add-on - enabling it
            # just means the network_table_handler utility is in the config
            if "use_network_tables" in data or "network_tables_ip" in data:
                if data.get("use_network_tables"):
                    config.update_addon_settings(
                        "utilities", "network_table_handler",
                        {"network_tables_ip": data.get("network_tables_ip", "10.0.0.2")},
                        save=False,
                    )
                    config.enable_addon("utilities", "network_table_handler", save=False)
                else:
                    config.disable_addon("utilities", "network_table_handler", save=False)
            config.set("camera_configs", data["camera_configs"])
            # normalize entries (nested pipeline layout, default vision_model
            # for model-backed pipelines) so the first run is valid
            from iSpy.config.iSpyConfig import ensure_camera_entries_ready
            ensure_camera_entries_ready(config.get("camera_configs", {}))
            config.save()
            return jsonify(success=True)
        except Exception as e:
            return jsonify(error=str(e)), 500