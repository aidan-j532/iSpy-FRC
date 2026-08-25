import subprocess
import sys
import os
import platform
from pathlib import Path

SERVICE_NAME = "iSpy"
FIRST_BOOT_SERVICE_NAME = "ispy-first-boot"

MDNS_HOSTNAME = "ispy"

def setup_mdns(hostname: str = MDNS_HOSTNAME) -> None:
    """Make the board reachable at http://<hostname>.local"""
    platform_kind = get_platform()

    if platform_kind == "macos":
        _setup_mdns_macos(hostname)
        return

    if platform_kind not in ("linux_systemd", "linux_other"):
        print("mDNS setup skipped (unsupported platform).")
        return

    check = run(["systemctl", "list-unit-files", "avahi-daemon.service"], check=False)
    if "avahi-daemon.service" not in check.stdout:
        print("Installing avahi-daemon...")
        install = run(["sudo", "apt-get", "install", "-y", "avahi-daemon"], check=False)
        if install.returncode != 0:
            print(f"Failed to install avahi-daemon: {install.stderr.strip()}")
            print(f"Install it manually, then re-run setup_mdns() to get {hostname}.local")
            return

    current = run(["hostname"], check=False).stdout.strip()
    if current != hostname:
        result = run(["sudo", "hostnamectl", "set-hostname", hostname], check=False)
        if result.returncode != 0:
            print(f"Failed to set hostname: {result.stderr.strip()}")
            return
        run(["sudo", "sed", "-i", f"s/{current}/{hostname}/g", "/etc/hosts"], check=False)

    run(["sudo", "systemctl", "enable", "avahi-daemon"], check=False)
    run(["sudo", "systemctl", "restart", "avahi-daemon"], check=False)
    print(f"mDNS ready - board will be reachable at http://{hostname}.local:5000")
    print("(requires a reboot if the hostname just changed)")

    _configure_dhcp_hostname(hostname)


def _setup_mdns_macos(hostname: str = MDNS_HOSTNAME) -> None:
    """macOS ships Bonjour natively - just set the Bonjour hostname via scutil.

    No daemon install needed. scutil --set HostName sets the "real" hostname;
    ComputerName/LocalHostName control what shows up in Finder/Bonjour. We set
    all three so <hostname>.local resolves consistently and the Sharing prefpane
    shows something sane.
    """
    current = run(["scutil", "--get", "LocalHostName"], check=False).stdout.strip()
    if current == hostname:
        print(f"mDNS already set - reachable at http://{hostname}.local:5000")
        return

    for cmd in (
        ["sudo", "scutil", "--set", "ComputerName", hostname],
        ["sudo", "scutil", "--set", "LocalHostName", hostname],
        ["sudo", "scutil", "--set", "HostName", hostname],
    ):
        result = run(cmd, check=False)
        if result.returncode != 0:
            print(f"Failed to set {cmd[-2]}: {result.stderr.strip()}")
            print("Set it manually in System Settings > General > Sharing, "
                  "or re-run setup_mdns().")
            return

    # bounce mDNSResponder so the new LocalHostName takes effect immediately
    run(["sudo", "killall", "-HUP", "mDNSResponder"], check=False)
    print(f"mDNS ready - board will be reachable at http://{hostname}.local:5000")

def _configure_dhcp_hostname(hostname: str) -> None:
    """Best-effort: configure DHCP client to advertise hostname.

    Routers that publish DHCP client hostnames into local DNS will then
    resolve ``<hostname>`` (without .local) — useful for Windows clients
    that lack Bonjour/mDNS.  Failures are non-fatal.
    """
    # Try dhcpcd first (common on Raspberry Pi OS / Armbian).
    dhcpcd_conf = "/etc/dhcpcd.conf"
    if os.path.exists(dhcpcd_conf):
        try:
            with open(dhcpcd_conf) as f:
                existing = f.read()
            marker = f"# iSpy hostname hint"
            if marker not in existing:
                with open(dhcpcd_conf, "a") as f:
                    f.write(f"\n{marker}\nhostname {hostname}\n")
                run(["sudo", "systemctl", "restart", "dhcpcd"], check=False)
                print(f"dhcpcd: configured hostname hint '{hostname}' for DHCP.")
        except Exception as exc:
            print(f"dhcpcd: could not set hostname hint (non-fatal): {exc}")

    # Try NetworkManager (used on some Armbian builds).
    nm_conns = run(
        ["sudo", "nmcli", "-t", "-f", "NAME,DEVICE", "connection", "show", "--active"],
        check=False,
    )
    if nm_conns.returncode == 0 and nm_conns.stdout.strip():
        try:
            # Get the primary connection name
            primary = run(["sudo", "nmcli", "-t", "-f", "NAME", "general", "status"], check=False)
            conn_name = primary.stdout.strip().split("\n")[0] if primary.stdout.strip() else ""
            if conn_name:
                run(
                    ["sudo", "nmcli", "connection", "modify", conn_name,
                     "ipv4.dhcp-send-hostname", "yes",
                     "ipv4.dhcp-hostname", hostname],
                    check=False,
                )
                run(["sudo", "nmcli", "connection", "up", conn_name], check=False)
                print(f"NetworkManager: configured DHCP hostname '{hostname}'.")
        except Exception as exc:
            print(f"NetworkManager: could not set DHCP hostname (non-fatal): {exc}")

    # Try systemd-networkd (fallback for minimal images).
    networkd_dir = "/etc/systemd/network"
    if os.path.isdir(networkd_dir):
        try:
            for conffile in os.listdir(networkd_dir):
                if conffile.endswith(".network"):
                    path = os.path.join(networkd_dir, conffile)
                    with open(path) as f:
                        existing = f.read()
                    if "Hostname" not in existing:
                        with open(path, "a") as f:
                            f.write(f"\n[DHCP]\nHostname={hostname}\n")
                        print(f"systemd-networkd: configured DHCP hostname in {conffile}.")
        except Exception as exc:
            print(f"systemd-networkd: could not set DHCP hostname (non-fatal): {exc}")


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

def setup_first_boot_service(project_root: str | None = None) -> None:
    """Write and enable the ispy-first-boot.service oneshot unit."""
    python = sys.executable
    workdir = project_root or os.getcwd()

    unit = f"""[Unit]
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
"""
    unit_file = f"/etc/systemd/system/{FIRST_BOOT_SERVICE_NAME}.service"

    proc = subprocess.run(
        ["sudo", "tee", unit_file],
        input=unit,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"Failed to write {FIRST_BOOT_SERVICE_NAME}.service: {proc.stderr}")
        sys.exit(1)

    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", FIRST_BOOT_SERVICE_NAME])
    run(["sudo", "systemctl", "start", FIRST_BOOT_SERVICE_NAME])
    print(f"Service '{FIRST_BOOT_SERVICE_NAME}' installed, enabled, and started.")


def setup_systemd(script_path, project_root: str | None = None):
    user = "pi"
    python = sys.executable
    workdir = project_root or os.getcwd()

    service = f"""[Unit]
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
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _relaunch_as_admin_windows(cmd):
    try:
        # write the cmd to a .bat so we avoid quoting/escaping issues through Start-Process
        public = "C:\\Users\\Public"
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

        # either we're admin and schtasks still failed, or elevation got declined - fall back to a per-user startup entry
        try:
            appdata = str(Path.home() / "AppData" / "Roaming")
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


def setup(script_path: str, project_root: str | None = None):
    detected = get_platform()
    print(f"Detected platform: {detected}")
    if detected == "linux_systemd":
        setup_first_boot_service(project_root)
        setup_systemd(script_path, project_root)
        setup_mdns()
    elif detected == "windows":
        setup_windows(script_path)
    elif detected == "macos":
        setup_macos(script_path)
    else:
        print("Unsupported platform (no systemd detected). Set up a cron job manually:")
        print(f"  @reboot {sys.executable} {os.path.abspath(script_path)}")

if __name__ == "__main__":
    setup("watchdog.py iSpy/boot/service_daemon.py")