#!/bin/bash
# install.sh - sets up iSpy on any supported platform
set -e

echo "=== iSpy Install ==="
echo "Python: $(python3 --version)"

ARCH=$(uname -m)

if [[ "$ARCH" == "aarch64" ]]; then
    echo "Detected aarch64 - deploy install (no dev extras)."
    pip install . --break-system-packages
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Detected x86_64 - dev install (includes training tools)."
    pip install ".[dev]" --break-system-packages
else
    echo "Detected $ARCH - base install."
    pip install . --break-system-packages
fi

echo ""
echo "=== Done ==="
echo "Run: iSpy-boot    (first-time setup + auto-optimization)"
echo "     iSpy-run     (main detection loop)"
