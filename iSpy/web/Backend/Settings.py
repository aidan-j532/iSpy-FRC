import json
from flask import jsonify, request, Response
from iSpy.web.Backend.WebModule import WebModule

_HTML = """<!DOCTYPE html>
<html><head><title>Settings</title><style>
body{font-family:Arial;background:#111;color:#eee;padding:20px}
textarea{width:100%;height:70vh;background:#1a1a1a;color:#0f0;font-family:monospace;font-size:13px;border:1px solid #333;padding:10px}
button{padding:10px 20px;background:#2c7;color:#fff;border:none;border-radius:4px;cursor:pointer;margin-top:10px}
</style></head><body>
<h1>Settings (raw config.json)</h1>
<textarea id="cfg"></textarea><br/><button onclick="save()">Save</button><span id="msg"></span>
<script>
async function load(){document.getElementById('cfg').value = JSON.stringify(await (await fetch('/api/settings')).json(), null, 2);}
async function save(){
  const msg = document.getElementById('msg');
  try{
    const data = JSON.parse(document.getElementById('cfg').value);
    const r = await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j = await r.json();
    msg.textContent = r.ok ? 'Saved. Restart iSpy to apply.' : ('Error: '+j.error);
    msg.style.color = r.ok ? '#2c7' : '#f44';
  }catch(e){ msg.textContent='Invalid JSON: '+e.message; msg.style.color='#f44'; }
}
load();
</script></body></html>"""


class SettingsModule(WebModule):
    plugin_name = "settings"

    def register_routes(self, flask_app):
        flask_app.add_url_rule("/settings", "settings_page", lambda: Response(_HTML, mimetype="text/html"))
        flask_app.add_url_rule("/api/settings", "api_settings_get", self._get, methods=["GET"])
        flask_app.add_url_rule("/api/settings", "api_settings_post", self._post, methods=["POST"])

    def _get(self):
        return jsonify(self.context["config"].config)

    def _post(self):
        try:
            data = request.get_json(force=True)
            required = ("unit", "vision_model", "camera_configs")
            missing = [k for k in required if k not in data]
            if missing:
                return jsonify(error=f"Missing required key(s): {missing}"), 400
            self.context["config"]._update_config(data)
            self.context["config"].save()
            return jsonify(success=True)
        except Exception as e:
            return jsonify(error=str(e)), 500