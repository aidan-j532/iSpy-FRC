#!/bin/bash
# build-image.sh - builds a custom Orange Pi image with VisionCore first-boot
# Run on an x86 Linux machine (Ubuntu/Debian recommended)
# Requirements: debootstrap, qemu-user-static, qemu-system-arm, parted, kpartx
set -e

IMAGE_NAME="orangepi.img"
IMAGE_SIZE="4G"
UBUNTU_RELEASE="jammy" # Ubuntu 22.04 - matches Orange Pi OS base
REPO_RAW="https://raw.githubusercontent.com/aidan-j532/VisionCore-Deploy/main/Image"

echo "=== VisionCore Image Builder ==="

apt-get install -y \
    debootstrap qemu-user-static qemu-system-arm \
    parted kpartx binfmt-support

echo "Creating ${IMAGE_SIZE} image..."
dd if=/dev/zero of="$IMAGE_NAME" bs=1M count=4096 status=progress
parted "$IMAGE_NAME" --script \
    mklabel msdos \
    mkpart primary ext4 1MiB 100%

LOOP=$(losetup -fP --show "$IMAGE_NAME")
mkfs.ext4 "${LOOP}p1"
MOUNT=$(mktemp -d)
mount "${LOOP}p1" "$MOUNT"

echo "Bootstrapping Ubuntu ${UBUNTU_RELEASE} ARM64..."
debootstrap \
    --arch=arm64 \
    --foreign \
    "$UBUNTU_RELEASE" \
    "$MOUNT" \
    http://ports.ubuntu.com/ubuntu-ports

cp /usr/bin/qemu-aarch64-static "$MOUNT/usr/bin/"
chroot "$MOUNT" /debootstrap/debootstrap --second-stage

echo "visioncore" > "$MOUNT/etc/hostname"

mkdir -p "$MOUNT/etc/systemd/system/getty@tty1.service.d"
cat > "$MOUNT/etc/systemd/system/getty@tty1.service.d/override.conf" <<EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear %I \$TERM
EOF

mkdir -p "$MOUNT/etc/systemd/network"
cat > "$MOUNT/etc/systemd/network/eth0.network" <<EOF
[Match]
Name=eth0

[Network]
DHCP=yes
EOF

chroot "$MOUNT" systemctl enable systemd-networkd
chroot "$MOUNT" systemctl enable systemd-resolved

echo "Injecting first-boot service..."

curl -fsSL "$REPO_RAW/first-boot.sh" \
    -o "$MOUNT/usr/local/bin/first-boot.sh"
chmod +x "$MOUNT/usr/local/bin/first-boot.sh"

# Copy the systemd unit
curl -fsSL "$REPO_RAW/first-boot.service" \
    -o "$MOUNT/etc/systemd/system/first-boot.service"

touch "$MOUNT/etc/visioncore-firstboot"

chroot "$MOUNT" systemctl enable first-boot.service

cat > "$MOUNT/etc/issue" <<EOF

  |-----------------------------------------------|
  |        VisionCore FRC - First Boot            |
  |                                               |
  |  Connect ethernet, then power on.             |
  |  Setup runs automatically.                    |
  |  Watch progress: journalctl -u first-boot -f  |
  |-----------------------------------------------|

EOF

umount "$MOUNT"
losetup -d "$LOOP"
rmdir "$MOUNT"

echo ""
echo "=== Done ==="
echo "Image: $IMAGE_NAME"
echo "Flash with: balena-etcher $IMAGE_NAME"
