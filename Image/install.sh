#!/bin/bash
# install.sh - sets up iSpy on any supported platform
set -e

echo "=== iSpy Install ==="
echo "Python: $(python3 --version)"

ARCH=$(uname -m)

# Runtime dependencies (including the CPU/CUDA-correct torch) are selected by
# the backend registry during `ispy setup`. Do not install [dev] here: its
# torchvision dependency can pull a CUDA torch from PyPI on Linux.
if [[ "${ISPY_DEV:-0}" == "1" ]]; then
    echo "ISPY_DEV=1 - installing developer extras (install torch first)."
    pip install -e ".[dev]" --break-system-packages
else
    echo "Detected $ARCH - base install (runtime backends are provisioned by ispy setup)."
    pip install . --break-system-packages
fi

echo ""
echo "=== Done ==="
echo "Run: ispy setup    (first-time setup + hardware-aware backend install)"
echo "     ispy start    (supervised vision + web UI)"
echo ""
echo "Dashboard: http://localhost:5000"
echo "  The supervisor keeps this URL available while iSpy boots, restarts,"
echo "  crashes, or runs with the web UI disabled."
