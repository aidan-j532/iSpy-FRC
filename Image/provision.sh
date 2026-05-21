#!/bin/bash
# provision.sh - run once on a fresh Orange Pi to set up VisionCore
# This is the automated version of install-deploy.sh used for imaging.
set -e

REPO_URL="https://github.com/aidan-j532/VisionCore-Deploy"
INSTALL_DIR="/opt/visioncore"
SERVICE_USER="visioncore"
RKNN_WHEELS_URL="https://github.com/aidan-j532/VisionCore-Deploy/tree/main/RknnWheels"

echo "=== VisionCore Provisioner ==="
echo "Python: $(python3 --version)"

apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git \
    libopencv-dev \
    v4l-utils \
    libgl1 libglib2.0-0

if [ -d "$INSTALL_DIR" ]; then
    echo "Repo exists, pulling latest..."
    git -C "$INSTALL_DIR" pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

pip3 install "$INSTALL_DIR[deploy]" \
    --find-links "$RKNN_WHEELS_URL" \
    --break-system-packages

if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
fi
usermod -aG video "$SERVICE_USER"

mkdir -p /etc/visioncore
CONFIG_DEST="/etc/visioncore/config.json"
if [ ! -f "$CONFIG_DEST" ]; then
    cp "$INSTALL_DIR/VisionCore/core/config.json" "$CONFIG_DEST"
    echo "Config copied to $CONFIG_DEST - edit this to configure your cameras."
fi

cat > /etc/systemd/system/visioncore.service <<EOF
[Unit]
Description=VisionCore FRC Vision Pipeline
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 -m VisionCore.boot.boot
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=VISIONCORE_CONFIG=$CONFIG_DEST

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable visioncore
systemctl start visioncore

echo ""
echo "=== Done ==="
echo "Status:  journalctl -u visioncore -f"
echo "Config:  $CONFIG_DEST"
echo "Restart: systemctl restart visioncore"
