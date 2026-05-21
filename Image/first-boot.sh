#!/bin/bash
# first-boot.sh - baked into the image, runs once on first boot
# Pulls the repo and provisions the full VisionCore environment
set -e

LOG="/var/log/visioncore-firstboot.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " VisionCore First Boot - $(date)"
echo "============================================"

# Wait for a real internet connection before doing anything
echo "Waiting for internet..."
for i in $(seq 1 30); do
    if curl -sf --max-time 3 https://github.com > /dev/null; then
        echo "Internet OK."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: No internet after 30 attempts. Plug in ethernet and reboot."
        exit 1
    fi
    sleep 2
done

# Run the main provisioner
curl -fsSL https://raw.githubusercontent.com/aidan-j532/VisionCore-Deploy/main/Image/provision.sh | bash

# Remove the flag file so this service never runs again
rm -f /etc/visioncore-firstboot

echo "============================================"
echo " First boot complete - $(date)"
echo " VisionCore is running."
echo " Logs: journalctl -u visioncore -f"
echo "============================================"