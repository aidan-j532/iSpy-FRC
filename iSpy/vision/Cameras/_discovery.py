"""Device discovery for OpenCV camera sources.

Extracted from the old web module so that every camera source can enumerate
what it can see. Discovery is platform-aware:

- Linux: /dev/video* glob grouped by physical device (sysfs identity), with
  v4l2 QUERYCAP capture-bit filtering and codec/radio/m2m node rejection.
- Windows: the UVC device-class registry (same ordering MSMF uses for its
  index assignments), with generic "USB Camera" names walked up to the parent
  USB node to find the real device name.
- Anything else (macOS ...): best-effort /dev/video glob + index probing.
"""

import glob
import os
import platform
import re
import subprocess

import cv2

# names windows hands out when it has nothing better - treat as unknown and keep digging
_GENERIC_WINDOWS_NAMES = {
    "", "usb video device", "usb camera", "camera", "uvc camera",
    "video capture device", "usb2.0 camera",
}

# v4l2 capability bits (videodev2.h) - caps u32 lives at offset 84 in QUERYCAP
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_VIDEO_M2M = 0x00004000
_VIDIOC_QUERYCAP = 0x80685600


def _v4l2_caps(video_path):
    try:
        import fcntl
        import struct
    except ImportError:
        return 0
    for mode in (os.O_RDWR, os.O_RDONLY):
        try:
            fd = os.open(video_path, mode | getattr(os, "O_NONBLOCK", 0))
        except OSError:
            continue
        try:
            buf = fcntl.ioctl(fd, _VIDIOC_QUERYCAP, b"\0" * 104)
            return struct.unpack_from("<I", buf, 84)[0]
        except OSError:
            continue
        finally:
            os.close(fd)
    return 0


_LINUX_NON_CAMERA_KEYWORDS = {"codec", "encode", "decode", "radio", "loopback"}


def _linux_is_known_non_camera(video_path):
    name = _linux_sysfs_name(video_path)
    if name:
        lower = name.lower()
        if any(kw in lower for kw in _LINUX_NON_CAMERA_KEYWORDS):
            return True
    # e.g. /dev/video-dec0, /dev/video-enc0 — the part after "video-" starts
    # with a non-digit which real camera nodes never do
    basename = os.path.basename(video_path).lower()
    if basename.startswith("video-") and len(basename) > 6 and not basename[6].isdigit():
        return True
    return False


def _linux_is_capture_node(video_path):
    caps = _v4l2_caps(video_path)
    if caps == 0:
        return None
    if caps & _V4L2_CAP_VIDEO_M2M:
        # m2m codecs (bcm2835-codec-decode) claim the capture bit but aint cameras
        return False
    return bool(caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))


def _linux_device_key(video_path):
    m = re.search(r"/video(?P<num>\d+)$", video_path)
    if not m:
        return video_path
    sysfs = f"/sys/class/video4linux/video{m.group('num')}/device"
    try:
        real = os.path.realpath(sysfs)
    except OSError:
        return None
    suffix = f"video4linux/video{m.group('num')}"
    if real.endswith(suffix):
        real = real[: -len(suffix)]
    return real


def _linux_sysfs_name(video_path):
    m = re.search(r"/video(\d+)$", video_path)
    if not m:
        return None
    try:
        with open(
            f"/sys/class/video4linux/video{m.group(1)}/name",
            encoding="utf-8", errors="replace",
        ) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_device_groups():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        result = None
    if result is not None and result.returncode == 0:
        groups = {}
        current_name = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                current_name = None
                continue
            if line.endswith(":") and not line.lower().startswith("/dev/"):
                current_name = line.rstrip(":")
                continue
            if line.startswith("/dev/video") and current_name:
                groups.setdefault(current_name, []).append(line)
        return [(name, nodes) for name, nodes in groups.items()]

    groups = {}
    for path in sorted(glob.glob("/dev/video*")):
        key = _linux_device_key(path) or path
        groups.setdefault(key, []).append(path)
    return [(None, nodes) for nodes in groups.values()]


def _windows_cameras_from_registry():
    devices = []
    try:
        import winreg
    except ImportError:
        return devices

    class_key = r"SYSTEM\CurrentControlSet\Control\DeviceClasses\{e5323777-f976-4f5b-9b55-b94699c46e44}"
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_key)
    except OSError:
        return devices

    try:
        subkey_pos = 0
        camera_index = 0
        while True:
            try:
                iface = winreg.EnumKey(key, subkey_pos)
            except OSError:
                break
            subkey_pos += 1
            if "#{" not in iface:
                continue
            raw_hw = iface[:iface.find("#{")]
            # USB cameras expose two UVC interfaces (MI_00 video, MI_01 metadata)
            # - strip &MI_XX so they collapse to the same physical device
            dedup_key = re.sub(r"&MI_\w+", "", raw_hw, flags=re.IGNORECASE)
            devices.append({
                "index": camera_index,
                "name": _windows_camera_name(iface) or f"Camera {camera_index}",
                "hw_id": raw_hw,
                "dedup_key": dedup_key,
            })
            camera_index += 1
    finally:
        winreg.CloseKey(key)
    return devices


def _windows_camera_name(iface):
    try:
        import winreg
    except ImportError:
        return None

    end = iface.find("#{")
    if end == -1:
        return None
    body = iface[:end]
    for prefix in ("##?#", "#?#"):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break

    def _read_name(enum_path):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, enum_path)
        except OSError:
            return None
        try:
            for value in ("FriendlyName", "DriverDesc", "DeviceDesc"):
                try:
                    name, _ = winreg.QueryValueEx(key, value)
                except OSError:
                    continue
                if not isinstance(name, str):
                    continue
                if name.startswith("@"):
                    # "inf_path,%desc%;Friendly Name" -> drop the locator, keep the friendly part
                    name = name.rsplit(";", 1)[-1] if ";" in name else ""
                name = name.strip()
                if name:
                    return name
        finally:
            winreg.CloseKey(key)
        return None

    # Each # in the interface path maps to \ in the registry Enum path:
    #   USB#VID_xxx&PID_yyy&MI_00#instance_id -> Enum\USB\VID_xxx&PID_yyy&MI_00\instance_id
    parts = body.split("#")
    if len(parts) < 3:
        return None
    enum_path = "SYSTEM\\CurrentControlSet\\Enum\\" + "\\".join(parts)
    hw_id = "\\".join(parts[:2])

    name = _read_name(enum_path)
    if name and name.lower() not in _GENERIC_WINDOWS_NAMES:
        return name

    # generic iface name -> real one usually lives on the parent usb node, not the uvc iface
    m = re.match(r"^USB\\(VID_\w+&PID_\w+)(?:&MI_\w+)?$", hw_id, re.IGNORECASE)
    if m:
        parent_hw = f"USB\\{m.group(1)}"
        parent_root = f"SYSTEM\\CurrentControlSet\\Enum\\{parent_hw}"
        try:
            parent_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, parent_root)
        except OSError:
            parent_key = None
        if parent_key is not None:
            try:
                try:
                    sub, _ = winreg.EnumKey(parent_key, 0)
                except OSError:
                    sub = None
                if sub:
                    parent_name = _read_name(f"{parent_root}\\{sub}")
                    if parent_name:
                        return parent_name
            finally:
                winreg.CloseKey(parent_key)
    return name or None


def _probe_index_devices(claimed):
    devices = []
    for i in range(0, 10):
        if str(i) in claimed:
            devices.append({"path": str(i), "name": f"Camera {i}", "device_id": None})
            continue
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                devices.append({
                    "path": str(i), "name": f"Camera {i}",
                    "resolution": f"{w}x{h}" if w and h else None,
                    "device_id": None,
                })
                cap.release()
        except Exception:
            continue
    return devices


def probe_opencv_devices(claimed_sources: set | None = None) -> list[dict]:
    """Enumerate the physical OpenCV camera sources currently connected.

    ``claimed_sources`` is the set of sources already bound to configured
    cameras; those are still returned (the UI marks them "active") but they are
    not re-probed by the index loop.
    """
    claimed = {str(s) for s in (claimed_sources or [])}
    devices = []
    system_name = platform.system()

    if system_name == "Linux":
        # one entry per physical device, not per node: uvc cams have a
        # metadata companion node (video0+video1) and codec/radio nodes
        # pollute the glob - QUERYCAP filters + sysfs identity dedupes
        seen_keys = set()
        for label, nodes in _linux_device_groups():
            capture = [n for n in nodes if _linux_is_capture_node(n)]
            if not capture:
                # nodes where caps couldn't be read - include only if sysfs
                # name doesn't scream "not a camera" (codec/encoder/decoder)
                capture = [
                    n for n in nodes
                    if _linux_is_capture_node(n) is None
                    and not _linux_is_known_non_camera(n)
                ]
            if not capture:
                # all nodes confirmed non-capture (encoder/codec/radio...) - skip the group
                continue
            node = sorted(capture)[0]
            key = _linux_device_key(node) or _linux_sysfs_name(node) or node
            if key in seen_keys:
                continue
            seen_keys.add(key)
            devices.append({
                "path": node,
                "name": label or _linux_sysfs_name(node) or node,
                "device_id": key,
            })

        # catch whatever grouping missed - still drop non-capture nodes + dupes
        for path in sorted(glob.glob("/dev/video*")):
            is_capture = _linux_is_capture_node(path)
            if is_capture is False:
                continue
            if is_capture is None and _linux_is_known_non_camera(path):
                continue
            key = _linux_device_key(path) or _linux_sysfs_name(path) or path
            if key in seen_keys:
                continue
            seen_keys.add(key)
            devices.append({
                "path": path,
                "name": _linux_sysfs_name(path) or path,
                "device_id": key,
            })

    elif system_name == "Windows":
        # no index probing - MSMF cant open by index, every attempt spams
        # "VIDEOIO(MSMF)..." to stderr, and discover reruns on refresh so
        # it'd spam forever. registry key order == CAP_MSMF index order
        seen_interfaces = set()
        for cam in _windows_cameras_from_registry():
            dedup = cam.get("dedup_key") or cam.get("hw_id")
            if dedup in seen_interfaces:
                continue
            seen_interfaces.add(dedup)
            index = str(cam["index"])
            if index in claimed:
                continue
            devices.append({
                "path": index,
                "name": cam["name"],
                "device_id": cam.get("hw_id"),
            })

    else:
        # macos / other: best-effort /dev/video glob + index probing
        for path in sorted(glob.glob("/dev/video*")):
            devices.append({
                "path": path, "name": path,
                "device_id": None,
            })

        if not devices:
            devices = _probe_index_devices(claimed)

    # dedupe
    seen = {}
    for dev in devices:
        key = dev.get("device_id") or dev.get("path")
        if key in seen:
            continue
        seen[key] = dev
    return list(seen.values())