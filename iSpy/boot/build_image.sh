#!/usr/bin/env bash
# iSpy/boot/build_image.sh
#
# Bakes iSpy into a base Raspberry Pi OS (or Armbian, for Orange Pi) image.
# Run this on a Linux host with qemu-user-static installed. Produces
# ispy-<board>-<date>.img.xz ready to flash with Raspberry Pi Imager / balenaEtcher.
#
# Usage:
#   ./build_image.sh rpi       # bakes onto Raspberry Pi OS Lite (64-bit)
#   ./build_image.sh orangepi  # bakes onto Armbian for Orange Pi 5
#
# Requires: qemu-user-static, binfmt-support, xz-utils, wget, parted, kpartx

set -euo pipefail

BOARD="${1:?usage: build_image.sh [rpi|orangepi]}"
WORK_DIR="$(mktemp -d)"
OUT_NAME="ispy-${BOARD}-$(date +%Y%m%d)"

trap 'sudo umount "$WORK_DIR"/rootfs/{dev,proc,sys} 2>/dev/null; sudo kpartx -d "$WORK_DIR"/base.img 2>/dev/null; rm -rf "$WORK_DIR"' EXIT

case "$BOARD" in
  rpi)
    BASE_URL="https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-latest/"
    ;;
  orangepi)
    BASE_URL="https://github.com/armbian/community/releases/latest/download/Armbian_community_orangepi5_bookworm_current.img.xz"
    ;;
  *)
    echo "Unknown board: $BOARD (expected rpi or orangepi)"; exit 1
    ;;
esac

echo "==> Downloading base image for $BOARD"
wget -q --show-progress -O "$WORK_DIR/base.img.xz" "$BASE_URL" \
  || { echo "Base image URL needs manual resolution for $BOARD - check the vendor's latest release page"; exit 1; }
unxz "$WORK_DIR/base.img.xz"

echo "==> Mounting image"
LOOPDEV=$(sudo losetup --show -fP "$WORK_DIR/base.img")
BOOT_PART="${LOOPDEV}p1"
ROOT_PART="${LOOPDEV}p2"

mkdir -p "$WORK_DIR/rootfs"
sudo mount "$ROOT_PART" "$WORK_DIR/rootfs"
sudo mount "$BOOT_PART" "$WORK_DIR/rootfs/boot" 2>/dev/null || true

echo "==> Enabling qemu-arm64 for chroot"
sudo cp /usr/bin/qemu-aarch64-static "$WORK_DIR/rootfs/usr/bin/"

echo "==> Bind-mounting kernel filesystems"
sudo mount --bind /dev "$WORK_DIR/rootfs/dev"
sudo mount --bind /proc "$WORK_DIR/rootfs/proc"
sudo mount --bind /sys "$WORK_DIR/rootfs/sys"

echo "==> Copying iSpy repo into image"
sudo mkdir -p "$WORK_DIR/rootfs/opt/iSpy-FRC"
sudo cp -r "$(git rev-parse --show-toplevel)/." "$WORK_DIR/rootfs/opt/iSpy-FRC/"

echo "==> Chroot: installing deps + running fresh boot + enabling services"
sudo chroot "$WORK_DIR/rootfs" /bin/bash -e <<'CHROOT_EOF'
set -e
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y python3 python3-pip python3-venv avahi-daemon v4l-utils

cd /opt/iSpy-FRC
pip3 install -e . --break-system-packages

# fresh install: this pulls hardware-appropriate deps via
# install_special_dependencies(auto_install=True) (see first_boot.py patch)
python3 -m iSpy.boot.boot -f

# bake in the systemd units + mDNS hostname so it's live on first real boot,
# not just this chroot session
python3 - <<'PYEOF'
from iSpy.boot.setup_service import setup_first_boot_service, setup_systemd, setup_mdns
setup_first_boot_service(project_root="/opt/iSpy-FRC")
setup_systemd("watchdog.py iSpy/boot/service_daemon.py", project_root="/opt/iSpy-FRC")
PYEOF

systemctl enable avahi-daemon
echo "ispy" > /etc/hostname
sed -i 's/127.0.1.1.*/127.0.1.1\tispy/' /etc/hosts

# expand-on-first-boot stays enabled (raspi-config / armbian-config default) -
# don't disable it, the shipped image is intentionally small
CHROOT_EOF

echo "==> Cleaning up chroot mounts"
sudo umount "$WORK_DIR/rootfs/dev" "$WORK_DIR/rootfs/proc" "$WORK_DIR/rootfs/sys"
sudo rm "$WORK_DIR/rootfs/usr/bin/qemu-aarch64-static"
sudo umount "$WORK_DIR/rootfs/boot" 2>/dev/null || true
sudo umount "$WORK_DIR/rootfs"
sudo losetup -d "$LOOPDEV"

echo "==> Compressing final image"
mv "$WORK_DIR/base.img" "./${OUT_NAME}.img"
xz -T0 -9 "./${OUT_NAME}.img"

echo "==> Done: ${OUT_NAME}.img.xz"
echo "Flash with Raspberry Pi Imager or balenaEtcher. Board boots as 'ispy.local'."