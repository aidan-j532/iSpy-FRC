#!/bin/bash
# install-dev.sh - sets up a VisionCore dev environment (x86_64, training tools)
# Run from the repo root: ./install-dev.sh
set -e

echo "=== VisionCore Dev Install ==="
echo "Python: $(python3 --version)"

ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    echo "WARNING: This script is optimized for x86_64 (desktop)."
    echo "         You are on: $ARCH"
    echo "         Continue anyway? (y/n)"
    read -r reply
    if [[ "$reply" != "y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Install base + dev extras (torch, tensorflow, etc.)
# boot.py handles all hardware-specific deps at runtime
pip install ".[dev]" --break-system-packages

echo ""
echo "=== Done ==="
echo "Run: visioncore-boot    (first-time setup + auto-optimization)"
echo "     visioncore-run     (main detection loop)"
