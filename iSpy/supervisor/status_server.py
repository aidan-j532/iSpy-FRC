"""Threaded stdlib HTTP server for the supervisor's always-on :5000 pages."""

from __future__ import annotations

import errno
import html
import json
import mimetypes
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from iSpy.core.request_auth import remote_denied_message, restart_allowed
from iSpy.supervisor.config import allow_remote_restart, read_runtime_config
from iSpy.supervisor.messages import load_messages
from iSpy.supervisor.state import snapshot

_PAGE_CSS = """
:root{
  --bg:#0d1117;--bg-elevated:#161b22;--bg-surface:#1c2128;
  --border:#30363d;--border-light:#3d444d;
  --text:#e6edf3;--text-muted:#8b949e;--text-dim:#6e7681;
  --accent:#2f81f7;--accent-hover:#4a95f8;--accent-dim:rgba(47,129,247,.15);
  --ok:#3fb950;--bad:#f85149;--warn:#d29922;--radius:10px;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{
  margin:0;min-height:100vh;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;
}
body.restart-page{
  display:flex;align-items:center;justify-content:center;padding:24px;
  background:
    radial-gradient(circle at 50% 0%,rgba(47,129,247,.16),transparent 55%),
    var(--bg);
}
.shell{width:100%;max-width:760px;margin:0 auto;padding:48px 24px}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:36px;color:var(--text-muted);
  font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:.78rem}
.brand-mark{width:34px;height:34px;border-radius:9px;background:var(--accent-dim);
  border:1px solid rgba(47,129,247,.4);display:flex;align-items:center;justify-content:center;
  color:var(--accent);font-weight:700;font-size:1rem;letter-spacing:0}
.card{background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius);
  padding:28px;box-shadow:0 10px 30px rgba(0,0,0,.28)}
h1{margin:0 0 10px;font-size:clamp(1.5rem,4vw,2.15rem);line-height:1.18;letter-spacing:-.02em}
.sub{margin:0;color:var(--text-muted);font-size:1rem;max-width:62ch}
.phase{margin-top:18px;display:inline-flex;align-items:center;gap:10px;border:1px solid var(--border);
  background:var(--bg-surface);border-radius:999px;padding:7px 13px;color:var(--text-muted);
  font-size:.88rem;max-width:100%}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px var(--accent-dim);
  animation:pulse 1.6s ease-in-out infinite;flex:none}
@keyframes pulse{50%{opacity:.35;transform:scale(.8)}}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}
button{font:inherit;border-radius:8px;border:1px solid var(--border-light);background:var(--bg-surface);
  color:var(--text);padding:9px 16px;cursor:pointer;transition:background .15s,border-color .15s}
button:hover{background:#252b33;border-color:#59616b}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.primary:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
button:disabled{opacity:.55;cursor:wait}
.note{margin-top:20px;padding:14px 16px;border:1px solid var(--border);border-radius:8px;
  background:var(--bg-surface);color:var(--text-muted);font-size:.9rem}
.note.warn{border-color:rgba(210,153,34,.45);color:#e3b341}
.note.bad{border-color:rgba(248,81,73,.45);color:#ff7b72}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0 0}
.meta div{background:var(--bg-surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.meta dt{color:var(--text-dim);font-size:.72rem;text-transform:uppercase;letter-spacing:.07em}
.meta dd{margin:2px 0 0;font-size:.95rem;overflow-wrap:anywhere}
pre.log{margin:18px 0 0;padding:16px;background:#090c10;border:1px solid var(--border);
  border-radius:8px;color:#c9d1d9;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
  max-height:320px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
.error{margin-top:16px;color:#ff7b72;font-size:.9rem;min-height:1.2em}
.restart-card{width:100%;max-width:680px;text-align:center}
.restart-card img{display:block;max-width:min(100%,420px);max-height:44vh;width:auto;height:auto;
  margin:0 auto 26px;border-radius:12px;object-fit:contain}
.restart-card .wiggle{color:var(--accent);font-weight:600;min-height:1.5em;display:block;margin-top:12px}
.muted{color:var(--text-muted)}
.small{font-size:.86rem}
@media (max-width:520px){.shell{padding:28px 16px}.card{padding:22px}.meta{grid-template-columns:1fr}}
"""


def _esc(value) -> str:
    return html.escape(str(value or ""))


def _script_json(value) -> str:
    return json.dumps(value).replace("</", "<\\/")


def _shell(title: str, body: str, *, restart: bool = False, extra_js: str = "") -> bytes:
    body_class = "restart-page" if restart else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0d1117">
<title>{_esc(title)} - iSpy</title>
<style>{_PAGE_CSS}</style>
</head>
<body class="{body_class}">
{body}
<script>{extra_js}</script>
</body>
</html>"""
    return page.encode("utf-8")


def _brand() -> str:
    return (
        '<div class="brand"><span class="brand-mark">i</span>'
        "<span>iSpy FRC</span></div>"
    )


def _log_block(state: dict, *, element_id: str | None = None) -> str:
    lines = state.get("log_tail") or []
    text = (
        "\n".join(lines)
        if lines
        else "No log output yet - Outputs/log.txt is not readable."
    )
    id_attr = f' id="{_esc(element_id)}"' if element_id else ""
    return f'<pre class="log"{id_attr}>{_esc(text)}</pre>'


def _poll_js(state_key: str, web_enabled: bool | None = None) -> str:
    expected_web = "null" if web_enabled is None else json.dumps(bool(web_enabled))
    return f"""
(function () {{
  const expectedState = {_script_json(state_key)};
  const expectedWeb = {expected_web};
  let failures = 0;
  async function poll() {{
    try {{
      const r = await fetch("/api/supervisor/state", {{cache: "no-store"}});
      if (!r.ok) throw new Error("http " + r.status);
      const j = await r.json();
      failures = 0;
      const phaseEl = document.getElementById("boot-phase");
      if (phaseEl && j.phase) phaseEl.textContent = j.phase;
      const logEl = document.getElementById("boot-log");
      if (logEl && Array.isArray(j.log_tail)) {{
        logEl.textContent = j.log_tail.length
          ? j.log_tail.join("\\n")
          : "No log output yet - Outputs/log.txt is not readable.";
      }}
      const childTookOver = j.served_by && j.served_by !== "supervisor";
      if (childTookOver || j.state !== expectedState || (expectedWeb !== null && !!j.web_enabled !== expectedWeb)) {{
        window.location.reload();
      }}
    }} catch (e) {{
      // During `ispy setup` the server intentionally disappears when setup
      // finishes; reloading there would just replace the page with a browser
      // error. Other pages keep polling and the next successful request reloads.
      failures++;
    }}
  }}
  setInterval(poll, 2000);
}})();
"""


def _action_js() -> str:
    return """
window.ispyAction = async function (path, opts) {
  opts = opts || {};
  if (opts.confirm && !window.confirm(opts.confirm)) return;
  const err = document.getElementById("action-error");
  const buttons = document.querySelectorAll("button[data-action]");
  buttons.forEach(function (b) { b.disabled = true; });
  if (err) err.textContent = "";
  try {
    const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
    let j = {};
    try { j = await r.json(); } catch (e) {}
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    if (err) err.textContent = opts.wait || "Working...";
  } catch (e) {
    if (err) err.textContent = e.message || String(e);
    buttons.forEach(function (b) { b.disabled = false; });
  }
};
"""


def _restarting_body(state: dict, messages: dict) -> str:
    title = "iSpy is restarting" if state.get("state") == "restarting" else "iSpy is stopping"
    subtitle = (
        "You will be reconnected once it boots up."
        if state.get("state") == "restarting"
        else "The web UI will return when iSpy is ready."
    )
    choices = messages["restarting"]
    image = messages.get("image")
    image_html = (
        '<img src="/restarting-image" alt="iSpy restarting">' if image else ""
    )
    return f"""
<main class="card restart-card">
  {image_html}
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(subtitle)}</p>
  <span class="wiggle" id="rotating-message">{_esc(random.choice(choices))}</span>
  <p class="small muted">This page reconnects automatically.</p>
</main>
<script>
(function () {{
  const messages = {_script_json(choices)};
  let i = 0;
  const el = document.getElementById("rotating-message");
  setInterval(function () {{
    i = (i + 1) % messages.length;
    el.textContent = messages[i];
  }}, 8000);
}})();
</script>
"""


def _idle_body() -> str:
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>iSpy isn't running</h1>
    <p class="sub">The vision process is stopped. Start it to bring the robot pipeline and web UI back.</p>
    <div class="actions">
      <button class="primary" data-action onclick="ispyAction('/api/supervisor/start', {{wait:'Starting iSpy...'}})">Start iSpy</button>
    </div>
    <div class="error" id="action-error"></div>
    <div class="note small">The status page stays available on port 5000 while iSpy starts.</div>
  </main>
</div>
"""


def _fresh_boot_body(messages: dict, state: dict) -> str:
    choices = messages["fresh_boot"]
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>iSpy is doing a fresh boot</h1>
    <p class="sub">First-time setup can take a while. This page will disappear as soon as setup exits.</p>
    <div class="phase"><span class="dot"></span><span id="rotating-message">{_esc(random.choice(choices))}</span></div>
    <div class="note">Fresh boot recreates Config/, Outputs/ and YoloModels/ and downloads the default models.</div>
    {_log_block(state, element_id="boot-log")}
  </main>
</div>
<script>
(function () {{
  const messages = {_script_json(choices)};
  let i = 0;
  const el = document.getElementById("rotating-message");
  setInterval(function () {{ i = (i + 1) % messages.length; el.textContent = messages[i]; }}, 8000);
}})();
</script>
"""


def _booting_body(state: dict) -> str:
    phase = state.get("phase") or "Starting the vision process"
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>iSpy is booting</h1>
    <p class="sub">The vision process is starting. The full UI will load automatically.</p>
    <div class="phase"><span class="dot"></span><span id="boot-phase">{_esc(phase)}</span></div>
    <div class="note small">This can take a little longer while models and camera pipelines initialize.</div>
    {_log_block(state, element_id="boot-log")}
  </main>
</div>
"""


def _web_off_body(state: dict) -> str:
    reason = state.get("web_off_reason") or "the web UI is disabled for this run"
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>The iSpy web UI is off for this run</h1>
    <p class="sub">Vision is still running. {_esc(reason)}. The full dashboard is not being served.</p>
    <div class="note warn">Detection, trackers, NetworkTables and the robot connection are untouched.</div>
    <div class="actions">
      <button class="primary" data-action onclick="ispyAction('/api/supervisor/restart-normal', {{wait:'Restarting with normal settings...'}})">Restart with normal settings</button>
    </div>
    <div class="error" id="action-error"></div>
    <div class="note small">This sets performance_level back to 4 (Full Features) and restarts iSpy. It is the same as running <code>ispy performance 4</code> and restarting.</div>
  </main>
</div>
"""


def _running_unavailable_body(state: dict) -> str:
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>iSpy is running, but its web UI did not take over the port</h1>
    <p class="sub">Vision is alive; port 5000 is still held by the supervisor. Restart iSpy to try again.</p>
    {_log_block(state)}
    <div class="actions">
      <button class="primary" data-action onclick="ispyAction('/api/supervisor/restart', {{confirm:'This stops vision and the robot connection for a few seconds. Do not do this mid-match. Continue?', wait:'Restarting iSpy...'}})">Restart iSpy</button>
    </div>
    <div class="error" id="action-error"></div>
  </main>
</div>
"""


def _crashed_body(state: dict) -> str:
    stopped = state.get("state") == "crash_loop_stopped"
    attempt = state.get("attempt_label") or "0/5"
    if stopped:
        title = "iSpy stopped after repeated crashes"
        sub = (
            f"It crashed {attempt} times, so the automatic restart loop stopped. "
            "Check the log below, fix the cause, then restart."
        )
        action = "Restart iSpy"
    else:
        title = "iSpy crashed"
        sub = (
            f"Restart attempt {attempt}. The supervisor will try again automatically. "
            "You can also restart now."
        )
        action = "Restart now"
    error = state.get("last_error") or ""
    return f"""
<div class="shell">
  {_brand()}
  <main class="card">
    <h1>{_esc(title)}</h1>
    <p class="sub">{_esc(sub)}</p>
    <dl class="meta">
      <div><dt>Attempt</dt><dd>{_esc(attempt)}</dd></div>
      <div><dt>State</dt><dd>{_esc(state.get("state"))}</dd></div>
      <div><dt>Last error</dt><dd>{_esc(error or "process exited")}</dd></div>
    </dl>
    {_log_block(state)}
    <div class="actions">
      <button class="primary" data-action onclick="ispyAction('/api/supervisor/restart', {{confirm:'This stops vision and the robot connection for a few seconds. Do not do this mid-match. Continue?', wait:'Restarting iSpy...'}})">{_esc(action)}</button>
    </div>
    <div class="error" id="action-error"></div>
  </main>
</div>
"""


def render_page(state: dict) -> bytes:
    messages = load_messages()
    current = state.get("state")
    action_js = _action_js()
    if current in ("restarting", "stopping"):
        return _shell(
            "Restarting" if current == "restarting" else "Stopping",
            _restarting_body(state, messages),
            restart=True,
            extra_js=action_js + _poll_js(current),
        )
    if current == "fresh_boot":
        return _shell(
            "Fresh boot",
            _fresh_boot_body(messages, state),
            extra_js=action_js + _poll_js("fresh_boot"),
        )
    if current == "booting":
        return _shell(
            "Booting",
            _booting_body(state),
            extra_js=action_js + _poll_js("booting"),
        )
    if current == "crashed":
        return _shell(
            "Crashed",
            _crashed_body(state),
            extra_js=action_js + _poll_js("crashed"),
        )
    if current == "crash_loop_stopped":
        return _shell(
            "Crash loop stopped",
            _crashed_body(state),
            extra_js=action_js + _poll_js("crash_loop_stopped"),
        )
    if current == "running":
        if not state.get("web_enabled", True):
            return _shell(
                "Web UI off",
                _web_off_body(state),
                extra_js=action_js + _poll_js("running", web_enabled=False),
            )
        return _shell(
            "Web UI unavailable",
            _running_unavailable_body(state),
            extra_js=action_js + _poll_js("running", web_enabled=True),
        )
    return _shell(
        "Not running",
        _idle_body(),
        extra_js=action_js + _poll_js("idle"),
    )


class _StatusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


class StatusServer:
    """Own :5000 while the child boots, hand it off, and take it back."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        action_handler=None,
        snapshot_func=None,
    ):
        self.host = host
        self.port = port
        self.action_handler = action_handler
        self.snapshot_func = snapshot_func or snapshot
        self._server: _StatusHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def serving(self) -> bool:
        with self._lock:
            return self._server is not None

    def start(self, retry_s: float = 10.0) -> bool:
        with self._lock:
            if self._server is not None:
                return True
            deadline = time.monotonic() + retry_s
            last_error = None
            while True:
                try:
                    self._server = _StatusHTTPServer(
                        (self.host, self.port), _StatusRequestHandler
                    )
                    break
                except OSError as exc:
                    last_error = exc
                    if exc.errno not in (errno.EADDRINUSE, errno.EACCES, 10048):
                        raise
                    if time.monotonic() >= deadline:
                        raise last_error
                    time.sleep(0.2)
            self._server.status_server = self
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="ispy-status-http",
                daemon=True,
            )
            self._thread.start()
            return True

    def release(self) -> None:
        with self._lock:
            server, self._server = self._server, None
            thread, self._thread = self._thread, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)

    stop = release

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)


class _StatusRequestHandler(BaseHTTPRequestHandler):
    server_version = "iSpySupervisor"
    sys_version = ""

    @property
    def supervisor(self) -> StatusServer:
        return self.server.status_server  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        # Keep the supervisor quiet; the vision child owns the useful logs.
        return

    def _send(self, code: int, body: bytes = b"", content_type: str = "text/html"):
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD" and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, payload, code: int = 200):
        self._send(
            code,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )

    def _state(self) -> dict:
        try:
            return self.supervisor.snapshot_func()
        except Exception:
            return snapshot()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/supervisor/state":
            self._json(self._state())
            return
        if path in ("/restarting-image", "/restarting.png"):
            self._serve_image()
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path.startswith("/api/"):
            self._json({"error": "not found"}, 404)
            return
        if path not in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send(200, render_page(self._state()))

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/supervisor/"):
            self._json({"ok": False, "error": "not found"}, 404)
            return

        # iSpy does not sit behind a trusted reverse proxy: ignore
        # X-Forwarded-For entirely and authenticate on the socket peer.
        remote_addr = self.client_address[0]
        cfg = read_runtime_config()
        if not restart_allowed(
            remote_addr,
            self.headers,
            allow_remote=allow_remote_restart(cfg),
        ):
            self._json(
                {
                    "ok": False,
                    "error": remote_denied_message(allow_remote_restart(cfg)),
                },
                403,
            )
            return

        # drain tiny request bodies so keep-alive clients are not surprised
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if 0 < length <= 1024 * 1024:
                self.rfile.read(length)
        except Exception:
            pass

        action = {
            "/api/supervisor/start": "start",
            "/api/supervisor/stop": "stop",
            "/api/supervisor/restart": "restart",
            "/api/supervisor/restart-normal": "restart-normal",
        }.get(path)
        if action is None:
            self._json({"ok": False, "error": "not found"}, 404)
            return
        handler = self.supervisor.action_handler
        if handler is None:
            self._json(
                {"ok": False, "error": "the supervisor is not accepting actions right now"},
                503,
            )
            return
        try:
            result = handler(action)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        self._json(result, 200 if result.get("ok") else 503)

    def _serve_image(self):
        try:
            image = (load_messages().get("image") or "").strip()
        except Exception:
            image = ""
        if not image:
            self._json({"error": "no restarting image configured"}, 404)
            return
        path = Path(image)
        try:
            if not path.is_file():
                self._json({"error": "restarting image not found"}, 404)
                return
            data = path.read_bytes()
        except OSError:
            self._json({"error": "restarting image not readable"}, 404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, data, content_type)
