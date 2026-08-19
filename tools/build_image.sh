#!/usr/bin/env bash
# build_image.sh — Build a zero-touch iSpy SD card image for multiple board families.
#
# Run on a Linux dev machine.  Downloads the appropriate base image for the
# target board, mounts it, chroots in, installs deps + iSpy, registers
# systemd units, sets hostname, then shrinks + compresses.
#
# Baked at image-build time:
#   - System packages (opencv deps, v4l-utils, avahi-daemon)
#   - iSpy pip package (+ all runtime dependencies)
#   - ispy-first-boot.service and ispy.service systemd units (enabled)
#   - Hostname set to "ispy" for http://ispy.local:5000
#
# Happens live at first boot (ispy-first-boot.service → first_boot.py):
#   - Config/config.json creation (if missing)
#   - Camera auto-detection and registration
#
# Usage:
#   sudo bash tools/build_image.sh <board>
#
# Supported boards:
#   orangepi5          Orange Pi 5 (RK3588) — Armbian
#   orangepi-zero3     Orange Pi Zero3 (RK3566) — Armbian
#   raspberrypi        Raspberry Pi (any) — Raspberry Pi OS Lite 64-bit
#
# Requirements on the dev machine:
#   - pishrink.sh (https://github.com/Drewsif/piShrink) in PATH
#   - qemu-user-static + binfmt-support (for ARM chroot on x86)
#   - xz-utils (for decompression of base images + final compression)
#   - curl or wget (for downloading base images)
#
# Output:  ispy-<board>-<version>.img.xz in the current directory.

set -euo pipefail

# ---------------------------------------------------------------------------
# Board definitions — extend this map as new boards are added.
# Keys: board slug used on the command line.
# Values: "family|base_image_url_or_source|extra_prepare_steps_function"
# ---------------------------------------------------------------------------
ISPY_VERSION="${ISPY_VERSION:-$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null || echo "dev")}"

declare -A BOARD_FAMILY BOARD_URL BOARD_ARCH BOARD_PREPARE

# --- Rockchip / Armbian boards ------------------------------------------------
BOARD_FAMILY[orangepi5]="rockchip"
BOARD_URL[orangepi5]="https://dl.armbian.com/orangepi5/Ubuntu_jammy_current_server_arm64.img.xz"
BOARD_ARCH[orangepi5]="arm64"

BOARD_FAMILY[orangepi-zero3]="rockchip"
BOARD_URL[orangepi-zero3]="https://dl.armbian.com/orangepizero3/Ubuntu_jammy_current_server_arm64.img.xz"
BOARD_ARCH[orangepi-zero3]="arm64"

# --- Raspberry Pi -------------------------------------------------------------
BOARD_FAMILY[raspberrypi]="rpi"
BOARD_URL[raspberrypi]="https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2024-11-19/2024-11-19-raspios-bookworm-arm64-lite.img.xz"
BOARD_ARCH[raspberrypi]="arm64"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: sudo bash $0 <board>"
    echo ""
    echo "Supported boards:"
    echo "  orangepi5          Orange Pi 5 (RK3588) — Armbian"
    echo "  orangepi-zero3     Orange Pi Zero3 (RK3566) — Armbian"
    echo "  raspberrypi        Raspberry Pi (any) — Raspberry Pi OS Lite"
    echo ""
    echo "Output: ispy-<board>-${ISPY_VERSION}.img.xz"
    exit 1
}

BOARD="${1:-}"
if [[ -z "$BOARD" ]] || [[ -z "${BOARD_FAMILY[$BOARD]+x}" ]]; then
    echo "ERROR: Unknown or missing board target '${BOARD}'." >&2
    usage
fi

FAMILY="${BOARD_FAMILY[$BOARD]}"
BASE_URL="${BOARD_URL[$BOARD]}"
ARCH="${BOARD_ARCH[$BOARD]}"
OUTPUT_NAME="ispy-${BOARD}-${ISPY_VERSION}"

WORK_DIR="$(mktemp -d /tmp/ispy-build-XXXXXX)"
MOUNT_POINT="${WORK_DIR}/mnt"
IMAGE_DIR="${WORK_DIR}/image"

cleanup() {
    echo "==> Cleaning up build workspace..."
    # unmount if still mounted
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        umount "$MOUNT_POINT" 2>/dev/null || true
    fi
    losetup -D 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)." >&2
    exit 1
fi

for cmd in curl xz chroot pishrink.sh; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: Required command '$cmd' not found in PATH." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 1. Download & extract base image
# ---------------------------------------------------------------------------
echo "==> Downloading base image for ${BOARD} (${FAMILY})..."
mkdir -p "$IMAGE_DIR"
BASE_IMG_XZ="${IMAGE_DIR}/base.img.xz"
curl -L -o "$BASE_IMG_XZ" "$BASE_URL"

echo "==> Extracting base image..."
xz -d "$BASE_IMG_XZ"
BASE_IMG="${IMAGE_DIR}/base.img"

# ---------------------------------------------------------------------------
# 2. Locate root partition and mount
# ---------------------------------------------------------------------------
echo "==> Locating root partition..."
# Use kpartx or losetup + fdisk to find the root partition offset.
LOOP_DEV=$(losetup -fP --show "$BASE_IMG")

# Find the Linux partition (type 83) — last partition is usually root on SBC images
ROOT_PART=""
for part in ${LOOP_DEV}p*; do
    if [[ -b "$part" ]]; then
        ROOT_PART="$part"
    fi
done
if [[ -z "$ROOT_PART" ]]; then
    echo "ERROR: Could not find a root partition in $BASE_IMG" >&2
    exit 1
fi

mkdir -p "$MOUNT_POINT"
mount "$ROOT_PART" "$MOUNT_POINT"

# If there's a boot partition, mount it too (RPi needs /boot/firmware)
BOOT_PART="${LOOP_DEV}p1"
if [[ -b "$BOOT_PART" ]] && [[ "$FAMILY" == "rpi" ]]; then
    mkdir -p "${MOUNT_POINT}/boot/firmware"
    mount "$BOOT_PART" "${MOUNT_POINT}/boot/firmware"
fi

IN_CHROOT=false

# ---------------------------------------------------------------------------
# Helper: run a command inside the target (chroot or direct)
# ---------------------------------------------------------------------------
run_in_target() {
    if $IN_CHROOT; then
        "$@"
    else
        chroot "$MOUNT_POINT" /usr/bin/env \
            PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
            "$@"
    fi
}

# ---------------------------------------------------------------------------
# 3. Board-specific preparation (runs inside chroot)
#    Only the base-image differences go here.  All shared steps follow.
# ---------------------------------------------------------------------------
echo "==> Running board-specific preparation for ${BOARD}..."
case "$FAMILY" in
    rockchip)
        # Armbian images are mostly ready — ensure kernel headers are available
        # for any out-of-tree modules iSpy might need at runtime.
        run_in_target bash -c '
            apt-get update
            apt-get install -y linux-headers-$(uname -r) 2>/dev/null || true
        ' || true
        ;;
    rpi)
        # Raspberry Pi OS Lite needs camera interface enabled and GPU memory set.
        if [[ -f "${MOUNT_POINT}/boot/firmware/config.txt" ]]; then
            sed -i '/^dtoverlay=vc4/d' "${MOUNT_POINT}/boot/firmware/config.txt"
            grep -q 'start_x=1' "${MOUNT_POINT}/boot/firmware/config.txt" || \
                echo 'start_x=1' >> "${MOUNT_POINT}/boot/firmware/config.txt"
            grep -q 'gpu_mem=128' "${MOUNT_POINT}/boot/firmware/config.txt" || \
                echo 'gpu_mem=128' >> "${MOUNT_POINT}/boot/firmware/config.txt"
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Shared install steps — common to all board families
# ---------------------------------------------------------------------------

ISPY_USER="pi"

# 4a. System packages (baked)
echo "==> Installing system packages..."
run_in_target bash -c '
    apt-get update
    apt-get install -y \
        python3 python3-pip python3-venv \
        libgl1-mesa-glx libglib2.0-0 \
        v4l-utils \
        avahi-daemon \
        libatlas-base-dev \
        libjpeg-dev libpng-dev \
        libhdf5-dev \
        wget curl git
'

# 4b. pip-install iSpy (baked)
echo "==> Installing iSpy..."
run_in_target bash -c "
    pip3 install --break-system-packages --no-cache-dir \
        'git+https://github.com/aidan-j532/iSpy-FRC.git'
"

# 4c. Systemd units (baked — enabled so they start on every boot)
echo "==> Registering systemd services..."
run_in_target bash -c "
    cd /home/${ISPY_USER} 2>/dev/null || cd /root
    python3 -c \"
import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages')
from iSpy.boot.setup_service import setup
setup('watchdog.py iSpy/boot/service_daemon.py')
\"
" || {
    # Fallback: write unit files directly if the pip import path differs.
    echo "  setup_service.py call failed — writing unit files directly..."
    PYTHON_BIN=$(run_in_target which python3)
    WORK_DIR_RT=$(run_in_target bash -c 'python3 -c "from pathlib import Path; print(Path(\"iSpy\").resolve().parent)"' 2>/dev/null || echo "/home/${ISPY_USER}/iSpy-FRC")

    cat > "${MOUNT_POINT}/etc/systemd/system/ispy-first-boot.service" <<UNIT
[Unit]
Description=iSpy first-boot setup
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=${PYTHON_BIN} -m iSpy.boot.first_boot
WorkingDirectory=${WORK_DIR_RT}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

    cat > "${MOUNT_POINT}/etc/systemd/system/ispy.service" <<UNIT
[Unit]
Description=iSpy
After=network-online.target ispy-first-boot.service
Requires=ispy-first-boot.service

[Service]
ExecStart=${PYTHON_BIN} -m iSpy.boot.boot
Restart=on-failure
RestartSec=5
User=${ISPY_USER}
WorkingDirectory=${WORK_DIR_RT}

[Install]
WantedBy=multi-user.target
UNIT

    run_in_target systemctl daemon-reload
    run_in_target systemctl enable ispy-first-boot.service
    run_in_target systemctl enable ispy.service
}

# 4d. Hostname / mDNS (baked)
echo "==> Setting hostname to 'ispy'..."
run_in_target bash -c '
    hostnamectl set-hostname ispy
    sed -i "s/127.0.1.1.*/127.0.1.1\tispy/" /etc/hosts
    systemctl enable avahi-daemon
    systemctl restart avahi-daemon
'

# 4e. Enable the UDP announce service for client discovery
echo "==> Enabling iSpy announce service..."
cat > "${MOUNT_POINT}/etc/systemd/system/ispy-announce.service" <<UNIT
[Unit]
Description=iSpy UDP announce service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${PYTHON_BIN:-$(run_in_target which python3)} -m iSpy.boot.announce
WorkingDirectory=${WORK_DIR_RT:-/home/${ISPY_USER}/iSpy-FRC}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
run_in_target systemctl enable ispy-announce.service

# 4f. Cleanup
echo "==> Cleaning up apt caches and temp files..."
run_in_target bash -c '
    apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    history -c
'

# ---------------------------------------------------------------------------
# 5. Unmount, shrink, compress
# ---------------------------------------------------------------------------
echo "==> Unmounting..."
sync
umount "$MOUNT_POINT" 2>/dev/null || true
[[ -d "${MOUNT_POINT}/boot/firmware" ]] && umount "${MOUNT_POINT}/boot/firmware" 2>/dev/null || true
losetup -D

echo "==> Shrinking image with pishrink.sh..."
pishrink.sh -a "$BASE_IMG" "${OUTPUT_NAME}.img"

echo "==> Compressing..."
xz -9 "${OUTPUT_NAME}.img"

echo ""
echo "============================================================"
echo "Image build complete: ${OUTPUT_NAME}.img.xz"
echo ""
echo "Board:   ${BOARD} (${FAMILY})"
echo "Arch:    ${ARCH}"
echo "Version: ${ISPY_VERSION}"
echo ""
echo "Baked in:"
echo "  - System packages (opencv deps, v4l-utils, avahi)"
echo "  - iSpy pip package + runtime dependencies"
echo "  - ispy-first-boot.service (enabled)"
echo "  - ispy.service (enabled, requires first-boot)"
echo "  - ispy-announce.service (enabled, UDP discovery beacon)"
echo "  - Hostname: ispy (http://ispy.local:5000)"
echo ""
echo "Flash with:  sudo dd if=${OUTPUT_NAME}.img.xz of=/dev/sdX bs=4M status=progress"
echo "  or use Raspberry Pi Imager / balenaEtcher"
echo "============================================================"
