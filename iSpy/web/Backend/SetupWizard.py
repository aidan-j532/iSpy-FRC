from flask import jsonify, request, Response
from iSpy.web.Backend.WebModule import WebModule

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#0d1117">
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
  <title>iSpy Setup</title>
  <link rel="stylesheet" href="/static/css/design.css">
  <style>
    body { display: flex; flex-direction: column; align-items: center; padding: 48px 20px; }
    .setup-wrap { width: 100%; max-width: 640px; }
    .setup-header {
      display: flex; align-items: center; gap: 12px;
      font-weight: 700; font-size: 1.35rem; letter-spacing: 0.3px;
      color: var(--text); margin-bottom: 4px;
    }
    .setup-header svg { width: 26px; height: 26px; color: var(--accent); flex-shrink: 0; }
    .setup-subtitle { color: var(--text-dim); font-size: 0.9rem; margin: 0 0 24px; }
    .form-row select.form-input, .form-row input.form-input { flex: 1; min-width: 0; }
    .setup-alert {
      display: none; margin-top: 16px; padding: 10px 14px;
      border-radius: var(--radius); border: 1px solid transparent;
      font-size: 0.88rem; font-weight: 500;
    }
    .setup-alert.ok { display: block; background: var(--ok-dim); color: var(--ok); border-color: var(--ok); }
    .setup-alert.err { display: block; background: var(--bad-dim); color: var(--bad); border-color: var(--bad); }
    .setup-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 4px; }
  </style>
</head>
<body>
<div class="setup-wrap">
  <div class="setup-header">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
    <span>iSpy</span>
  </div>
  <p class="setup-subtitle">Board setup &mdash; configure your first camera to get started.</p>

  <div class="card">
    <div class="settings-section">
      <h3>Camera</h3>
      <div class="form-row">
        <label class="form-label" for="cam_name">Camera name</label>
        <input id="cam_name" class="form-input" value="default_cam">
      </div>
      <div class="form-row">
        <label class="form-label" for="cam_source">Camera source (index or path)</label>
        <input id="cam_source" class="form-input" value="0">
      </div>
      <div class="form-row">
        <label class="form-label" for="cam_subsystem">Subsystem</label>
        <input id="cam_subsystem" class="form-input" value="field">
      </div>
    </div>

    <div class="settings-section">
      <h3>Units</h3>
      <div class="form-row">
        <label class="form-label" for="unit">Unit</label>
        <select id="unit" class="form-input">
          <option value="frc">FRC (meters)</option>
          <option>meter</option>
          <option>inch</option>
          <option>foot</option>
        </select>
      </div>
    </div>

    <div class="settings-section">
      <h3>NetworkTables</h3>
      <div class="form-row">
        <label class="form-label" for="use_nt">Use NetworkTables</label>
        <select id="use_nt" class="form-input">
          <option value="false">No</option>
          <option value="true">Yes</option>
        </select>
      </div>
      <div class="form-row" id="nt-ip-row">
        <label class="form-label" for="nt_ip">NetworkTables IP</label>
        <input id="nt_ip" class="form-input" value="10.0.0.2">
      </div>
    </div>

    <div class="setup-actions">
      <button class="btn-ok" id="finish-btn" onclick="go()">Finish Setup</button>
    </div>
    <div id="msg" class="setup-alert" role="status"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

function syncNtRow() {
  $('nt-ip-row').style.display = $('use_nt').value === 'true' ? '' : 'none';
}
$('use_nt').addEventListener('change', syncNtRow);
syncNtRow();

async function go() {
  const btn = $('finish-btn');
  const msg = $('msg');
  btn.disabled = true;
  btn.textContent = 'Saving\u2026';
  msg.className = 'setup-alert';
  const payload = {
    unit: $('unit').value,
    use_network_tables: $('use_nt').value === 'true',
    network_tables_ip: $('nt_ip').value,
    camera_configs: {
      [$('cam_name').value]: {
        name: $('cam_name').value,
        source: isNaN($('cam_source').value) ? $('cam_source').value : Number($('cam_source').value),
        subsystem: $('cam_subsystem').value,
        pipeline: {name: 'object_detection', settings: {}},
        yaw: 0, pitch: 0, height: 1.0, x: 0, y: 0,
        calibration: {distance: 0, game_piece_size: 0, size: 0, fov: 0}
      }
    }
  };
  try {
    const r = await fetch('/api/setup', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const j = await r.json();
    if (r.ok) {
      msg.textContent = 'Saved. Restart iSpy to apply.';
      msg.classList.add('ok');
      btn.textContent = 'Saved';
    } else {
      msg.textContent = 'Error: ' + j.error;
      msg.classList.add('err');
      btn.disabled = false;
      btn.textContent = 'Finish Setup';
    }
  } catch (e) {
    msg.textContent = 'Error: could not reach the iSpy server.';
    msg.classList.add('err');
    btn.disabled = false;
    btn.textContent = 'Finish Setup';
  }
}
</script>
</body>
</html>"""


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