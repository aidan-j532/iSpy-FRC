# 09 — Boot Sequence and Service Management

> How iSpy starts up, installs as a service, handles first-boot provisioning, and manages the running vision process.

---

## Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `ispy-boot` | `iSpy.boot.boot:main` | First-time setup, directory creation, service install |
| `ispy-run` | `iSpy.core.game_loop:main` | Start the vision pipeline |
| `iSpy/core/ispy.py` | `main()` | Convenience: boot then run in one command |

The `ispy-boot` command is the primary entry point for initial setup. It creates
the directory structure, copies bundled models, validates the system, optionally
installs a system service, and starts the UDP announcer so other tools can find
the board on the network. The `ispy-run` command skips all of that and goes
straight to running the vision pipeline. The combined `ispy.py` calls `on_boot()`
first, then `game_loop.main()`.

---

## Boot Sequence (`iSpy/boot/boot.py`, 342 lines)

### File-Level Constants and Setup (lines 1–39)

```python
_BOOT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = Path.cwd().resolve()
_ASSETS_DIR = _PACKAGE_ROOT.parent / "assets"

_READINESS_POLL_S = 2.0
_READINESS_WAIT_TIMEOUT_S = 1200
```

- `_PACKAGE_ROOT` and `_BOOT_DIR` both resolve to `iSpy/boot/`.
- `_PROJECT_ROOT` is the current working directory (the repo root).
- `_ASSETS_DIR` points to `iSpy/assets/` where bundled `.pt` models live.
- `_READINESS_POLL_S = 2.0` — seconds between readiness poll iterations.
- `_READINESS_WAIT_TIMEOUT_S = 1200` — 20-minute timeout for pipeline readiness.

Lines 36–39 capture the real stdout/stderr file descriptors before anything
swaps `sys.stdout`. This is critical because later code (the optimizer, OpenCV
fix) may redirect stdout, and the logging system must still write to the actual
terminal:

```python
_REAL_STDOUT_FD = os.dup(1)
_REAL_STDERR_FD = os.dup(2)
_REAL_STDOUT = os.fdopen(_REAL_STDOUT_FD, "w", buffering=1, closefd=False)
_REAL_STDERR = os.fdopen(_REAL_STDERR_FD, "w", buffering=1, closefd=False)
```

### main() (line 323)

The CLI entry point. Parses three flags:

| Flag | Long | Purpose |
|------|------|---------|
| `-s` | `--service` | Install as a system service after setup |
| `-f` | `--fresh` | Delete all generated dirs and start from scratch |
| `-w` | `--wait` | Wait for pipeline readiness after setup |

**Jetson CSI auto-fix (lines 324–327):** Before parsing any arguments, if the
system is a Jetson and the config has any CSI camera enabled, the boot process
calls `ensure_csi_capable_opencv(auto_fix=True)`. If the fix succeeds (OpenCV
was rebuilt with GStreamer), it calls `os.execv()` to re-execute `boot.py`
itself with the same arguments so the new OpenCV is loaded. This is necessary
because the venv's cv2 might lack GStreamer support, and CSI cameras on Jetson
require the GStreamer backend.

```python
def main():
    if has_jetson() and _any_camera_uses_csi():
        if ensure_csi_capable_opencv(auto_fix=True):
            logger.info("OpenCV fixed - re-executing boot.py to pick it up...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser.add_argument("-s", "--service", action="store_true",
                         help="Install and start the watchdog service")
    parser.add_argument("-f", "--fresh", action="store_true",
                         help="Forcefully wipe generated state ...")
    parser.add_argument("-w", "--wait", action="store_true",
                         help="Wait for all pipelines to be ready ...")
    args = parser.parse_args()
    on_boot(install_service=args.service, fresh=args.fresh, wait=args.wait)
```

### on_boot() (line 259)

The main boot orchestrator. Executes the following steps in order:

1. **Configure quiet logging** — calls `_configure_quiet_logging()` to set up
   filtered handlers that only emit `iSpy.*` log records.

2. **Setup files** — calls `setup_files(fresh=...)` to create the directory
   structure and copy bundled models.

3. **Validate system** — calls `validate_system()` which checks that required
   Python packages are installed, model files exist, and config is valid.

4. **Get pipeline classes** — calls `get_pipeline_classes()` which discovers
   all registered vision pipeline backends (object_detection, april_tag, etc.).

5. **Optionally wait for pipeline readiness** — if `-w` flag was passed,
   instantiates each camera's pipeline and polls `is_ready()` with a 1200-second
   timeout.

6. **Save config** — calls `config.save(quiet=True)` to persist any changes.

7. **Start UDP announcer** — imports and calls `start_announcer(daemon=True)` to
   broadcast discovery packets on the local subnet.

8. **Optionally install service** — if `-s` flag was passed, runs
   `install.py` as a subprocess.

```python
def on_boot(install_service: bool = False, fresh: bool = False, wait: bool = False):
    _configure_quiet_logging()

    if fresh:
        logger.info("boot -f: forcefully fresh installation state")
        setup_files(fresh=True)
        config_path = str(_PROJECT_ROOT / "Config" / "config.json")
        config = iSpyConfig(config_path, create=True)
        _bootstrap_default_camera(config)
    else:
        setup_files(fresh=False)
        config_path = search_for_config()
        if not config_path:
            raise RuntimeError(
                "No configuration found. First-run initialization is only "
                "performed by 'boot -f' (forcefully fresh) - run that first."
            )
        logger.info("Using existing config: %s", config_path)
        config = iSpyConfig(config_path, create=False)

    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")

    pipeline_classes = get_pipeline_classes()
    if wait:
        _wait_for_pipeline_ready(config, pipeline_classes)
    config.save(quiet=True)
    logger.info("Boot sequence complete.")
    ...
```

The `fresh` vs non-`fresh` paths differ significantly:
- **Fresh**: Creates config from defaults, runs `_bootstrap_default_camera()`.
- **Non-fresh**: Searches for an existing config, raises if none found.

### _configure_quiet_logging() (line 77)

Sets up a logging filter (`_iSpyLogFilter` at line 94) that only passes records
where the logger name is `"root"` or starts with `"iSpy"`. All third-party
library loggers (ultralytics, flask, werkzeug, etc.) are silenced by setting
their level to `WARNING`. Two handlers are attached:

1. **StreamHandler** bound to `_REAL_stdout` (the pre-captured real file
   descriptor) — prints to the terminal.
2. **FileHandler** writing to `Outputs/log.txt` — appends to the log file.

Both use the same format: `"%(asctime)s [%(name)s] %(levelname)s: %(message)s"`.

### _close_logging_handlers() (line 76)

Helper that removes and closes all existing root logger handlers before
re-configuring. Prevents duplicate log output if `on_boot()` is called
multiple times.

### search_for_config() (line 122)

Searches `Config/` for JSON files. Returns the first non-`config.json` file if
any exist (these are named configs created by the web UI), otherwise returns
`config.json` itself. Returns `None` if no config directory or no JSON files
exist.

### setup_files() (line 135)

Creates the full directory structure:

```
YoloModels/
  pytorch/        # bundled .pt models
  onnx/           # converted ONNX models
  tflite/         # converted TFLite models
  rknn/           # converted RKNN models
  openvino/       # converted OpenVINO models
  coreml/         # converted CoreML models
  engine/         # converted TensorRT engines
Config/
  config.json     # main config (created from defaults if missing)
Outputs/
  log.txt         # runtime log
QuantizeDataset/
  images/         # calibration images for quantization
  images_frc/     # FRC-specific calibration images
```

**Fresh boot behavior (lines 141–145):** When `fresh=True`, all four top-level
directories are deleted via `_remove_path_for_cleanup()` (which handles
read-only files on Windows by `chmod`-ing before removal), then recreated from
scratch.

**Bundled model copy (lines 162–167):** Iterates over all `.pt` files in
`iSpy/assets/` and copies them to `YoloModels/pytorch/`. Only copies if the
target doesn't exist or it's a fresh boot.

**Metadata generation (lines 169–186):** For every `.pt` file in
`YoloModels/pytorch/`, generates or reads a metadata sidecar (`.yaml` file
alongside the `.pt`). If `keywords.json` exists in assets, applies keyword
overrides to the metadata's `calibration_keywords` field. Metadata is generated
by `metadata_from_pt()` which loads the model and extracts class names, input
size, and task type.

```python
for pt_file in pytorch_dir.glob("*.pt"):
    meta_path = metadata_path_for(pt_file)
    try:
        if meta_path.exists():
            meta = read_metadata(meta_path)
        else:
            logger.info("Generating metadata for %s", pt_file.name)
            meta = metadata_from_pt(pt_file)

        if pt_file.stem in default_keywords:
            meta["calibration_keywords"] = default_keywords[pt_file.stem]
        write_metadata(meta_path, meta)
    except Exception as e:
        logger.warning("Could not generate metadata for %s: %s", pt_file.name, e)
```

### _bootstrap_default_camera() (line 52)

Auto-probes USB cameras on first boot when no camera source is configured:

1. Checks if `camera_configs` already has a camera with a non-empty `source`.
   If so, returns immediately.
2. Imports `CamerasModule` and calls `_probe_devices()` to scan for available
   cameras (USB on Linux, DirectShow/registry on Windows).
3. If no devices are found, logs a warning and returns.
4. Creates a camera config entry named `"camera_1"` using the default camera
   template from `config.default_config`, sets `source` to the first detected
   device path, and saves.

```python
def _bootstrap_default_camera(config: iSpyConfig):
    cams = config.get("camera_configs", {})
    if cams and any(c.get("source") not in (None, "") for c in cams.values()):
        return  # already has something real
    from iSpy.web.modules.cameras import CamerasModule
    devices = CamerasModule({})._probe_devices()
    if not devices:
        logger.warning("No cameras detected at first boot ...")
        return
    dev = devices[0]
    name = "camera_1"
    default_cam = config.default_config["camera_configs"]["default_cam"]
    cam_cfg = json.loads(json.dumps(default_cam))
    cam_cfg["name"] = name
    cam_cfg["source"] = dev["path"]
    config.set("camera_configs", {name: cam_cfg})
    config.save()
```

### _wait_for_pipeline_ready() (line 189)

Instantiates each camera's pipeline and polls `is_ready()` until all report
ready or the timeout expires.

**Parameters:**
- `config` — the `iSpyConfig` instance
- `pipeline_classes` — dict mapping pipeline name strings to their classes

**Algorithm:**
1. Extract all camera configs from `config.config["camera_configs"]`.
2. Raise `RuntimeError` if no cameras are configured.
3. For each camera, determine its pipeline name and look up the class. Raise if
   the pipeline name is unknown.
4. Instantiate the pipeline: `inst = cls(iSpyCameraConfig(cam_cfg), config, None)`.
5. Enter a polling loop with a 1200-second deadline:
   - For each pending camera, call `inst.is_ready()` which returns
     `(ready: bool, status: str)`.
   - Log status changes.
   - If status contains `"error:"` and not ready, raise immediately
     (unrecoverable error).
   - If ready, remove from pending set.
   - Sleep 2 seconds between iterations.
6. If any cameras are still pending after the deadline, raise with details.

```python
deadline = _time.monotonic() + _READINESS_WAIT_TIMEOUT_S
pending = set(instances)
last_status: dict[str, str] = {n: "" for n in instances}
while pending and _time.monotonic() < deadline:
    for name in tuple(pending):
        inst = instances[name]
        try:
            ready, status = inst.is_ready()
        except Exception as e:
            ready, status = False, f"error: {e}"
        if status != last_status[name]:
            logger.info("  camera %-16s -> %s", name, status)
            last_status[name] = status
        if "error:" in status and not ready:
            raise RuntimeError(
                f"Camera '{name}' pipeline entered an unrecoverable error ..."
            )
        if ready:
            pending.discard(name)
    if pending:
        _time.sleep(_READINESS_POLL_S)
```

### _any_camera_uses_csi() (line 310)

Reads the config file (if it exists) and checks whether any camera has
`"csi": True`. Used by `main()` to decide whether to run the OpenCV GStreamer
fix.

### _remove_path_for_cleanup() (line 42)

Wrapper around `shutil.rmtree` that handles Windows read-only files by
catching `OSError` on each file and `chmod`-ing it to `0o777` before retrying.

---

## First Boot (`iSpy/boot/first_boot.py`, 36 lines)

A systemd oneshot service entry point. Designed to run on every boot but
idempotent — it only does work if `Config/config.json` doesn't exist.

```python
def main() -> None:
    if _CONFIG_PATH.exists():
        logger.info("first_boot: Config/config.json already exists, nothing to do.")
        return

    logger.info("first_boot: Config/config.json not found — running fresh install.")
    on_boot(install_service=False, fresh=True, wait=False)
    logger.info("first_boot: fresh install complete.")
```

**Key details:**
- `_PROJECT_ROOT = Path.cwd().resolve()` — uses CWD as the project root.
- `_CONFIG_PATH` = `_PROJECT_ROOT / "Config" / "config.json"`.
- If the config exists, returns immediately (exit code 0).
- If not, calls `on_boot(fresh=True)` which wipes everything and recreates.
- The `__main__` block wraps `main()` in a try/except that logs the exception
  and calls `sys.exit(1)` on failure, so systemd reports the error.

This is run by `ispy-first-boot.service` (Type=oneshot, RemainAfterExit=yes).

---

## Service Management

### setup_service.py (`iSpy/boot/setup_service.py`, 350 lines)

Cross-platform service installer that detects the OS and creates the
appropriate service mechanism.

#### Platform Detection (line 111)

```python
def get_platform():
    if platform.system() == "Windows":
        return "windows"
    if platform.system() == "Darwin":
        return "macos"
    result = run(["pidof", "systemd"], check=False)
    if result.returncode == 0:
        return "linux_systemd"
    return "linux_other"
```

Returns one of: `"windows"`, `"macos"`, `"linux_systemd"`, `"linux_other"`.

#### Linux — systemd (lines 122–200)

Two services are created:

**1. ispy-first-boot.service (oneshot):**
```ini
[Unit]
Description=iSpy first-boot setup
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={python} -m iSpy.boot.first_boot
WorkingDirectory={workdir}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**2. iSpy.service (main):**
```ini
[Unit]
Description={SERVICE_NAME}
After=network-online.target ispy-first-boot.service
Requires=ispy-first-boot.service

[Service]
ExecStart={python} -m iSpy.boot.boot
Restart=on-failure
RestartSec=5
User={user}
WorkingDirectory={workdir}

[Install]
WantedBy=multi-user.target
```

The main service depends on first-boot completing successfully. Both are
written via `sudo tee` to handle the privilege escalation. After writing,
`daemon-reload`, `enable`, and `start` are called.

**mDNS setup (line 13):** `setup_mdns()` installs and configures avahi-daemon
so the board is reachable at `http://<hostname>.local:5000`. The hostname
defaults to a unique per-board name from `default_mdns_hostname()` —
`ispy-<6 hex>` derived from the machine-id (or first physical MAC, or OS
hostname hash). This prevents two coprocessors on the same field network from
colliding over `ispy.local`. Set `ISPY_MDNS_HOSTNAME` (or pass `hostname=`) to
force a specific name. Updates `/etc/hosts`, enables avahi.

**DHCP hostname hints (line 47):** `_configure_dhcp_hostname()` is a best-effort
fallback for Windows machines without Bonjour (and a parallel discovery path on
Linux). It hints the same unique per-board hostname derived by
`default_mdns_hostname()`. It tries three approaches:
1. **dhcpcd** (Raspberry Pi OS / Armbian) — appends `hostname ispy` to
   `/etc/dhcpcd.conf`.
2. **NetworkManager** — uses `nmcli connection modify` to set DHCP hostname.
3. **systemd-networkd** — appends `[DHCP]\nHostname=ispy` to `.network` files.

#### Windows — schtasks (lines 253–299)

Creates a Windows scheduled task that runs at logon:

```python
cmd = [
    "schtasks", "/create", "/tn", SERVICE_NAME,
    "/tr", f"{python} {script_path}",
    "/sc", "onlogon",
    "/rl", "highest",
    "/f"  # overwrite if exists
]
```

**UAC elevation (line 211):** If `schtasks` fails and the user isn't admin,
`_relaunch_as_admin_windows()` writes the command to a `.bat` file in
`C:\Users\Public\`, then uses PowerShell's `Start-Process -Verb RunAs` to
elevate. It captures the exit code from a temp file.

**Fallback (line 278):** If elevation is declined or still fails, creates a
per-user startup `.bat` in `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`.
This only runs at user login, not at system boot.

#### macOS — LaunchAgent (lines 302–331)

Creates a plist at `~/Library/LaunchAgents/com.iSpy.plist`:

```xml
<dict>
    <key>Label</key>
    <string>com.iSpy</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
```

`KeepAlive=true` means launchd will restart the process if it exits. Loads it
immediately via `launchctl load`.

#### Main Dispatcher (line 334)

```python
def setup(script_path: str, project_root: str | None = None):
    detected = get_platform()
    if detected == "linux_systemd":
        setup_first_boot_service(project_root)
        setup_systemd(script_path, project_root)
        setup_mdns()
    elif detected == "windows":
        setup_windows(script_path)
    elif detected == "macos":
        setup_macos(script_path)
    else:
        print("Unsupported platform ...")
```

---

### service_daemon.py (`iSpy/boot/service_daemon.py`, 149 lines)

A Flask-based service manager running on port 5050. Provides a REST API to
start, stop, restart, pause, and resume the vision pipeline subprocess.

#### VisionSupervisor class (line 15)

Manages a subprocess running `iSpy/core/game_loop.py`. Communicates via stdin
pipe with text commands.

**State machine:** `"stopped"` → `"running"` → `"paused"` → `"running"` →
`"stopped"`. Also has `"error"` and `"stopping"` transient states.

**Key methods:**

| Method | Lines | Behavior |
|--------|-------|----------|
| `start()` | 23–36 | Spawns subprocess, starts `_watch` thread, returns pid |
| `_watch()` | 38–48 | Background thread that waits for process exit, updates status |
| `_send(cmd)` | 50–59 | Writes command string to subprocess stdin |
| `pause()` | 61–66 | Sends `"PAUSE"` command, sets status to `"paused"` |
| `resume()` | 68–73 | Sends `"RESUME"` command, sets status to `"running"` |
| `stop(timeout)` | 75–94 | Sends `"SHUTDOWN"`, waits 10s, then terminate, then kill |
| `restart()` | 96–99 | Calls stop() then start() with 0.5s delay |
| `_save_state()` | 101–107 | Writes JSON state to `Outputs/service_state.json` |
| `get_status()` | 109–112 | Returns status dict with status, pid, last_error |

The `_watch()` thread (line 38) runs as a daemon thread and monitors the
subprocess. When it exits:
- If status was `"stopping"`, sets to `"stopped"` (clean shutdown).
- If exit code is 0 or None, sets to `"stopped"`.
- Otherwise sets to `"error"` with the exit code.

The `stop()` method (line 75) has a three-stage shutdown:
1. Send `"SHUTDOWN"` via stdin and wait up to `timeout` (default 10s).
2. If timeout expires, call `proc.terminate()` (SIGTERM) and wait 5s.
3. If still alive, call `proc.kill()` (SIGKILL).

#### Flask Routes (line 115)

```python
def create_service_app(entry_point: str) -> Flask:
    app = Flask(__name__)
    sup = VisionSupervisor(entry_point)

    @app.route("/service/status")
    def status(): return jsonify(sup.get_status())

    @app.route("/service/start", methods=["POST"])
    def start(): return jsonify(sup.start())

    @app.route("/service/stop", methods=["POST"])
    def stop(): return jsonify(sup.stop())

    @app.route("/service/restart", methods=["POST"])
    def restart(): return jsonify(sup.restart())

    @app.route("/service/pause", methods=["POST"])
    def pause(): return jsonify(sup.pause())

    @app.route("/service/resume", methods=["POST"])
    def resume(): return jsonify(sup.resume())

    return app
```

All endpoints return JSON with `{"ok": bool, ...}` and additional fields like
`"pid"`, `"status"`, `"last_error"`, `"error"`.

---

## Watchdog (`iSpy/boot/watchdog.py`, 23 lines)

A simple restart loop that wraps any Python script:

```python
if len(sys.argv) < 2:
    print("Usage: python watchdog.py <script.py>")
    sys.exit(1)

script = sys.argv[1]

while True:
    logger.info(f"Starting {script}...")
    result = subprocess.run([sys.executable, script])

    if result.returncode == 0:
        logger.info("Script exited cleanly, stopping watchdog.")
        break

    logger.warning(f"Script crashed (code {result.code}), restarting in 5s...")
    time.sleep(5)
```

**Behavior:**
- Runs the given script as a subprocess.
- If exit code is 0 → clean exit, stop watchdog.
- If exit code is non-zero → crash detected, wait 5 seconds, restart.
- The 5-second delay prevents rapid restart loops from consuming resources.

Typical invocation: `python watchdog.py iSpy/boot/service_daemon.py`

---

## UDP Announcer (`iSpy/boot/announce.py`, 117 lines)

Broadcasts board discovery packets on the local subnet so
`tools/find_ispy.py` (and similar clients) can locate iSpy boards even when
mDNS and DHCP hostname resolution both fail.

### Protocol Details

| Field | Value |
|-------|-------|
| Protocol | UDP broadcast |
| Port | 37429 |
| Interval | Every 5.0 seconds |
| Payload prefix | `ISPY_DISCOVER:` |
| Payload format | JSON: `{"hostname": str, "ip": str, "port": int}` |

### Functions

**`_local_ip()` (line 29):** Returns the preferred outbound IP without sending
traffic. Creates a UDP socket connected to `10.255.255.255:1` and reads back
the local socket address. Falls back to `"127.0.0.1"`.

**`_hostname()` (line 41):** Returns `socket.gethostname()`.

**`_broadcast_addr(ip)` (line 45):** Derives the broadcast address for a /24
subnet by replacing the last octet with 255. Falls back to `255.255.255.255`
if parsing fails.

**`build_payload()` (line 54):** Builds the full announce payload:
```python
MAGIC + json.dumps({"hostname": _hostname(), "ip": _local_ip(), "port": WEB_PORT}).encode("utf-8")
```

**`send_announcement(sock, payload)` (line 64):** Sends one UDP datagram to
the broadcast address on port 37429. Logs failures at DEBUG level.

**`announce_loop(stop_event)` (line 74):** Blocking loop that:
1. Creates a UDP socket with `SO_BROADCAST` enabled.
2. Builds the payload once (it doesn't change).
3. Sends an announcement every 5 seconds.
4. Uses `stop_event.wait(5.0)` for interruptible sleeping.
5. Closes the socket in a `finally` block.

**`start_announcer(daemon=True)` (line 92):** Starts `announce_loop` in a
background daemon thread. Returns a `threading.Event` that can be set to stop
the announcer.

**`main()` (line 103):** Standalone entry point for running as a systemd
service. Calls `announce_loop()` directly (blocking).

---

## OpenCV Fix (`iSpy/boot/opencv_fix.py`, 123 lines)

Auto-fixes missing GStreamer support needed for CSI cameras on Jetson boards.

### cv2_has_gstreamer() (line 13)

Checks the current OpenCV build info for GStreamer support:
```python
def cv2_has_gstreamer() -> bool:
    try:
        import cv2
        return bool(re.search(r"GStreamer:\s*YES", cv2.getBuildInformation()))
    except Exception:
        return False
```

### _apt_install_python3_opencv() (line 27)

Runs `sudo apt-get install -y python3-opencv` with a 600-second timeout. If
the first attempt fails, runs `apt-get update` first, then retries. Sets
`DEBIAN_FRONTEND=noninteractive` to avoid interactive prompts.

### _find_system_cv2_path() (line 43)

Finds the cv2 module installed by the system Python (not the venv):
1. Runs `{system_python} -c "import cv2, os; print(os.path.dirname(...))"`
2. If the path is a directory with `__init__.py`, returns it (package-style).
3. Otherwise looks for `cv2*.so` or `cv2*.pyd` files.

### _current_cv2_targets() (line 61)

Finds all existing cv2 artifacts in the venv's site-packages directories. Returns
a tuple of `(target_dir, existing_artifacts)` where:
- `target_dir` is the purelib path (where new cv2 will be placed).
- `existing_artifacts` is a list of `Path` objects for cv2 dirs, `.so` files,
  `.pyd` files, and `opencv_python*` dist-info directories.

### ensure_csi_capable_opencv(auto_fix=True) (line 81)

The main entry point. Algorithm:

1. **Platform check:** Returns `False` if not Linux.
2. **Already fixed:** Returns `False` if GStreamer is already present.
3. **Auto-fix disabled:** Returns `False` if `auto_fix=False`.
4. **apt install:** Calls `_apt_install_python3_opencv()`.
5. **Find system cv2:** Calls `_find_system_cv2_path()`.
6. **Verify GStreamer:** Runs the system python's cv2 build info and checks for
   `GStreamer: YES`.
7. **Remove old cv2:** Deletes all existing cv2 artifacts from the venv.
8. **Copy new cv2:** If the system cv2 is a directory, uses `shutil.copytree`.
   If it's a single `.so`/`.pyd` file, copies it directly.
9. **Returns `True`** to signal that the fix was applied and the caller should
   re-exec.

```python
def ensure_csi_capable_opencv(auto_fix: bool = True) -> bool:
    if sys.platform != "linux":
        return False
    if cv2_has_gstreamer():
        return False
    ...
    for old in existing:
        try:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        except Exception as e:
            logger.warning("Could not remove old cv2 artifact %s: %s", old, e)

    if system_cv2.is_dir():
        shutil.copytree(system_cv2, target_dir / "cv2")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(system_cv2, target_dir / system_cv2.name)

    logger.info("Vendored GStreamer-enabled cv2 (from apt) into %s", target_dir)
    return True
```

---

## CLI Entry Points (`iSpy/core/`)

### game_loop.py (62 lines)

The standalone vision pipeline runner:

```python
def main():
    config_path = Path.cwd() / "Config" / "config.json"
    config = iSpyConfig(str(config_path))
    vision = iSpy(config)
    vision.run()
```

Sets `OPENCV_LOG_LEVEL=ERROR` at module load time (line 6) to suppress OpenCV
noise. Configures quiet logging (same pattern as boot.py) before importing
`iSpy` to ensure all log output is filtered. Creates an `iSpyConfig` from the
default config path, instantiates the main `iSpy` vision controller, and calls
`run()` which enters the main loop.

### ispy.py (19 lines)

Combines boot + run in one command:

```python
def main():
    parser = argparse.ArgumentParser(description="iSpy boot sequence")
    parser.add_argument("-s", "--service", ...)
    parser.add_argument("-f", "--fresh", ...)
    parser.add_argument("-w", "--wait", ...)
    args = parser.parse_args()
    boot_run(install_service=args.service, fresh=args.fresh, wait=args.wait)
    game_loop_main()
```

Parses the same flags as `boot.py`, runs `on_boot()` first, then immediately
starts the vision pipeline.

---

## Boot Sequence Flow Diagram

```
main()
  │
  ├─ has_jetson() && _any_camera_uses_csi()?
  │    └─ YES: ensure_csi_capable_opencv(auto_fix=True)
  │         └─ succeeds? → os.execv() (re-exec)
  │
  ├─ argparse: -s, -f, -w
  │
  └─ on_boot(install_service, fresh, wait)
       │
       ├─ _configure_quiet_logging()
       │
       ├─ fresh?
       │    ├─ YES: setup_files(fresh=True) → iSpyConfig(create=True) → _bootstrap_default_camera()
       │    └─ NO:  setup_files(fresh=False) → search_for_config() → iSpyConfig(create=False)
       │
       ├─ validate_system()
       │
       ├─ get_pipeline_classes()
       │
       ├─ wait?
       │    └─ YES: _wait_for_pipeline_ready() [1200s timeout]
       │
       ├─ config.save(quiet=True)
       │
       ├─ start_announcer(daemon=True) [UDP broadcast]
       │
       └─ install_service?
            └─ YES: subprocess.run([python, install.py])
```
