#!/usr/bin/env bash
# iSpy one-liner installer for macOS / Linux dev machines.
# Usage: curl -fsSL https://raw.githubusercontent.com/<org>/iSpy-FRC/main/iSpy/boot/install.sh | bash

set -euo pipefail

echo "iSpy Installer"
echo "=============="

INSTALL_DIR="${ISPY_INSTALL_DIR:-$HOME/iSpy-FRC}"

# 1. Check python3.10+
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c 'import sys; print(sys.version_info[:2])')
        major=$(echo "$ver" | tr -dc '0-9,' | cut -d, -f1)
        minor=$(echo "$ver" | tr -dc '0-9,' | cut -d, -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.10+ not found."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Install it with: brew install python@3.12"
        echo "(no Homebrew? https://brew.sh)"
    else
        echo "Install it with your distro's package manager, e.g.:"
        echo "  sudo apt-get install python3.12 python3.12-venv"
    fi
    exit 1
fi
echo "Found: $PYTHON_BIN ($($PYTHON_BIN --version))"

# 2. Clone or update
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Existing install found at $INSTALL_DIR - pulling latest..."
    git -C "$INSTALL_DIR" pull
else
    echo "Cloning iSpy-FRC to $INSTALL_DIR ..."
    git clone https://github.com/aidan-j532/iSpy-FRC.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# 3. Install
echo "Installing iSpy and dependencies..."
"$PYTHON_BIN" -m pip install -e . --break-system-packages 2>/dev/null \
    || "$PYTHON_BIN" -m pip install -e .

# 4. Run fresh setup (prefer the `ispy` CLI; fall back to `python -m` if the
# console script didn't register, e.g. on some non-editable installs)
echo "Running first-time setup..."
if command -v ispy >/dev/null 2>&1; then
    ispy setup
else
    "$PYTHON_BIN" -m iSpy.boot.boot -f
fi

echo ""
echo "Setup complete."
echo "Run 'ispy start' from $INSTALL_DIR to start iSpy"
echo "  (fallback: '$PYTHON_BIN -m iSpy.boot.boot')."
echo "Run 'ispy start -s' -- or '$PYTHON_BIN -m iSpy.boot.boot -s' -- to install iSpy as a background service."
echo "Dashboard: http://localhost:5000"