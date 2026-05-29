#!/bin/bash
# provision.sh - run once on a fresh Orange Pi to set up iSpy
# This is the automated version of install-deploy.sh used for imaging.
set -e

REPO_URL="https://github.com/aidan-j532/iSpy-FRC"
INSTALL_DIR="/opt/iSpy"
SERVICE_USER="iSpy"

echo "=== iSpy Provisioner ==="
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

pip3 install "$INSTALL_DIR" \
    --break-system-packages

if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
fi
usermod -aG video "$SERVICE_USER"

mkdir -p /etc/iSpy
CONFIG_DEST="/etc/iSpy/config.json"
if [ ! -f "$CONFIG_DEST" ]; then
    cp "$INSTALL_DIR/Config/config.json" "$CONFIG_DEST"
    echo "Config copied to $CONFIG_DEST - edit this to configure your cameras."
fi

cat > /etc/systemd/system/iSpy.service <<EOF
[Unit]
Description=iSpy FRC Vision Pipeline
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 -m iSpy.boot.boot
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=iSpy_CONFIG=$CONFIG_DEST

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable iSpy
systemctl start iSpy

echo ""
echo "=== Done ==="
echo "Status:  journalctl -u iSpy -f"
echo "Config:  $CONFIG_DEST"
echo "Restart: systemctl restart iSpy"
