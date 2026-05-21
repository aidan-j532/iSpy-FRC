#!/bin/bash
# install-dev.sh - sets up a VisionCore dev environment
# Run from the repo root: ./install-dev.sh
set -e

RKNN_WHEELS_URL="https://github.com/yourname/visioncore/releases/expanded_assets/rknn-wheels"

echo "=== VisionCore Dev Install ==="
echo "Python: $(python3 --version)"

# Verify they're (we're) on x86, rknn-toolkit2 (full conversion toolkit) is x86 only
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    echo "ERROR: rknn-toolkit2 (full conversion toolkit) only runs on x86_64."
    echo "       You are on: $ARCH"
    echo "       For Orange Pi / aarch64, run install-deploy.sh instead."
    exit 1
fi

pip install ".[dev]" --find-links "$RKNN_WHEELS_URL"

echo ""
echo "=== Done ==="
echo "Run with: visioncore-run"
echo "         visioncore-boot"