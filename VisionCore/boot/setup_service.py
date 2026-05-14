import subprocess
import sys
import os
import platform

# Mainly vibe coded but supercool ngl

SERVICE_NAME = "visioncore"

def run(cmd, check=True):
    return subprocess.run(cmd, check=check, text=True, capture_output=True)

def get_platform():
    if platform.system() == "Windows":
        return "windows"
    if platform.system() == "Darwin":
        return "macos"
    # Linux, check if systemd is running
    result = run(["pidof", "systemd"], check=False)
    if result.returncode == 0:
        return "linux_systemd"
    return "linux_other"

def setup_systemd(script_path):
    user = os.environ.get("USER", "pi")
    python = sys.executable
    workdir = os.path.dirname(os.path.abspath(script_path))

    service = f"""[Unit]
Description={SERVICE_NAME}
After=network.target

[Service]
ExecStart={python} {os.path.abspath(script_path)}
Restart=always
RestartSec=5
User={user}
WorkingDirectory={workdir}

[Install]
WantedBy=multi-user.target
"""
    service_file = f"/etc/systemd/system/{SERVICE_NAME}.service"
    
    # Write via tee so we can use sudo
    proc = subprocess.run(
        ["sudo", "tee", service_file],
        input=service,
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        print(f"Failed to write service file: {proc.stderr}")
        sys.exit(1)

    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", SERVICE_NAME])
    run(["sudo", "systemctl", "start", SERVICE_NAME])
    print(f"Service '{SERVICE_NAME}' installed and started.")
    print(f"  Logs:    journalctl -u {SERVICE_NAME} -f")
    print(f"  Stop:    sudo systemctl stop {SERVICE_NAME}")
    print(f"  Disable: sudo systemctl disable {SERVICE_NAME}")


def _is_admin_windows():
    """Return True if the current process has administrator privileges."""
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _relaunch_as_admin_windows(cmd):
    """Run a command elevated via UAC using PowerShell Start-Process -Verb RunAs.

    Passes the exact schtasks command directly so there is no path/argv ambiguity.
    Returns True if the elevation request was accepted.
    """
    try:
        # Write the exact schtasks command to a .bat file so there are no
        # quoting/escaping issues passing arguments through Start-Process.
        public = os.environ.get("PUBLIC", "C:\\Users\\Public")
        bat_file = os.path.join(public, "vc_elevate.bat")
        out_file = os.path.join(public, "vc_schtasks_result.txt")

        # Quote args that contain spaces
        def quote(s):
            return f'"{s}"' if " " in s else s
        cmd_line = " ".join(quote(a) for a in cmd)

        with open(bat_file, "w") as f:
            f.write("@echo off\r\n")
            f.write(f"{cmd_line}\r\n")
            f.write(f'echo %ERRORLEVEL% > "{out_file}"\r\n')

        # Run the bat file elevated
        ps_cmd = f"Start-Process -FilePath '{bat_file}' -Verb RunAs -Wait"
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            check=False
        )

        try:
            exit_code = int(open(out_file).read().strip())
        except Exception:
            exit_code = -1
        finally:
            for f in [bat_file, out_file]:
                try: os.remove(f)
                except: pass

        if exit_code != 0:
            print(f"Elevated schtasks failed with exit code {exit_code}")
            return False
        return True
    except Exception as e:
        print(f"UAC elevation failed: {e}")
        return False


def setup_windows(script_path):
    python = sys.executable
    script_path = os.path.abspath(script_path)

    # Register as a scheduled task that runs at startup
    cmd = [
        "schtasks", "/create", "/tn", SERVICE_NAME,
        "/tr", f"{python} {script_path}",
        "/sc", "onlogon",
        "/rl", "highest",
        "/f"  # overwrite if exists
    ]
    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"Failed to create task: {result.stderr.strip()}")

        # If we're not already admin, re-run just the schtasks command elevated.
        if not _is_admin_windows():
            print("Requesting administrator privileges via UAC...")
            if _relaunch_as_admin_windows(cmd):
                print(f"Scheduled task '{SERVICE_NAME}' created successfully (elevated).")
                return
            else:
                print("UAC elevation was declined or the task creation failed.")

        # Either we ARE admin and schtasks still failed, or elevation was
        # declined — fall back to a per-user startup entry.
        try:
            appdata = os.environ.get("APPDATA")
            if not appdata:
                raise RuntimeError("APPDATA environment variable not found")
            startup_dir = os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
            os.makedirs(startup_dir, exist_ok=True)
            bat_path = os.path.join(startup_dir, f"{SERVICE_NAME}_startup.bat")
            cmdline = f'"{python}" "{script_path}"'
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write("@echo off\n")
                f.write(cmdline + "\n")
            print(f"Created per-user startup fallback: {bat_path}")
            print("Note: this will run at user login (not at system boot). To register a system task, run this installer as Administrator.")
            return
        except Exception as e:
            print(f"Fallback failed: {e}")
            print("Please rerun this script in an elevated Administrator PowerShell to install as a system task.")
            return

    print(f"Scheduled task '{SERVICE_NAME}' created.")
    print(f"  Start:  schtasks /run /tn {SERVICE_NAME}")
    print(f"  Stop:   schtasks /end /tn {SERVICE_NAME}")
    print(f"  Remove: schtasks /delete /tn {SERVICE_NAME}")


def setup_macos(script_path):
    python = sys.executable
    script_path = os.path.abspath(script_path)
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/com.{SERVICE_NAME}.plist")

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{SERVICE_NAME}</string>
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
</plist>
"""
    with open(plist_path, "w") as f:
        f.write(plist)

    run(["launchctl", "load", plist_path])
    print(f"LaunchAgent '{SERVICE_NAME}' installed and started.")
    print(f"  Stop:    launchctl unload {plist_path}")
    print(f"  Remove:  rm {plist_path}")


def setup(script_path: str):
    """Public entrypoint expected by install.py — choose platform-specific installer.

    script_path may be a single script path or a command string (it will be used verbatim
    when composing the platform-specific service/task definition)."""
    detected = get_platform()
    print(f"Detected platform: {detected}")
    if detected == "linux_systemd":
        setup_systemd(script_path)
    elif detected == "windows":
        setup_windows(script_path)
    elif detected == "macos":
        setup_macos(script_path)
    else:
        print("Unsupported platform (no systemd detected). Set up a cron job manually:")
        print(f"  @reboot {sys.executable} {os.path.abspath(script_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_service.py <script.py>")
        sys.exit(1)

    script = sys.argv[1]
    if not os.path.isfile(script):
        print(f"File not found: {script}")
        sys.exit(1)

    detected = get_platform()
    print(f"Detected platform: {detected}")

    if detected == "linux_systemd":
        setup_systemd(script)
    elif detected == "windows":
        setup_windows(script)
    elif detected == "macos":
        setup_macos(script)
    else:
        print("Unsupported platform (no systemd detected). Set up a cron job manually:")
        print(f"  @reboot {sys.executable} {os.path.abspath(script)}")