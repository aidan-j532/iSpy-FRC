#!/bin/bash
# install-deploy.sh - sets up VisionCore for deployment on Orange Pi / aarch64
# Run from the repo root: ./install-deploy.sh
set -e

echo "=== VisionCore Deploy Install ==="
echo "Python: $(python3 --version)"

ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "WARNING: This script is intended for aarch64 (Orange Pi / Rockchip)."
    echo "         You are on: $ARCH"
    echo "         Continue anyway? (y/n)"
    read -r reply
    if [[ "$reply" != "y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Install base package — boot.py handles all hardware-specific deps at runtime
pip install . --break-system-packages

echo ""
echo "=== Done ==="
echo "Run: visioncore-boot    (first-time setup + auto-optimization)"
echo "     visioncore-run     (main detection loop)"
