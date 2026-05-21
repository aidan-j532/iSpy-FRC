#!/bin/bash
# install-deploy.sh - sets up VisionCore for deployment on Orange Pi / aarch64
# Run from the repo root: ./install-deploy.sh
set -e

RKNN_WHEELS_URL="https://github.com/yourname/visioncore/releases/expanded_assets/rknn-wheels"

echo "=== VisionCore Deploy Install ==="
echo "Python: $(python3 --version)"

# Verify they're (we're) on aarch64 - rknn-toolkit-lite2 is aarch64 only
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "ERROR: rknn-toolkit-lite2 only runs on aarch64 (Orange Pi / Rockchip)."
    echo "       You are on: $ARCH"
    echo "       For a dev machine, run install-dev.sh instead."
    exit 1
fi

pip install ".[deploy]" --find-links "$RKNN_WHEELS_URL" --break-system-packages

echo ""
echo "=== Done ==="
echo "Run with: visioncore-run"
echo "         visioncore-boot"