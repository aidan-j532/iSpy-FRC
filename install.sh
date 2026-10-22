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
echo "Run: ispy setup    (first-time setup + auto-optimization)"
echo "     ispy start    (supervised vision + web UI)"
echo ""
echo "Dashboard: http://localhost:5000"
echo "  The supervisor keeps this URL available while iSpy boots, restarts,"
echo "  crashes, or runs with the web UI disabled."
