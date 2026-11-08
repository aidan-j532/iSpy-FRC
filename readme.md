# iSpy

> FRC vision pipeline for object detection and field mapping. Runs on Orange Pi (RK3588 NPU) and supports RKNN, ONNX, OpenVINO, TFLite, TensorRT, and CoreML backends.

iSpy is a plug-and-play computer vision system for FRC robots. Point a camera at the field, it detects game pieces, converts pixel positions into field-relative coordinates, and publishes them to your robot over NetworkTables.

## Features

- YOLO object detection (any size/format)
- Camera calibration to real-world coordinates
- Object tracking (EMA smoothing, DBSCAN clustering)
- NetworkTables publishing
- Live annotated video over a local web server
- Auto-selects the fastest model format for your hardware
- Process supervisor for crash recovery
- Plugin system (trackers, utilities, frame processors)
- Unit packs and bundles for measurement/output conventions and presets
- Application-agnostic core (FRC specifics live in plugins)

## Quick Start

Flash the pre-built image (`orangepi.img` from Releases) with balenaEtcher, power on with ethernet. The board clones, installs, and starts automatically. Watch progress:

```bash
journalctl -u ispy-first-boot -f
```

Service runs on every boot:

```bash
journalctl -u iSpy -f
```

## Manual Install

```bash
git clone https://github.com/aidan-j532/iSpy-FRC
cd iSpy-FRC
python -m venv .venv && source .venv/bin/activate
pip install -e .
ispy-boot
```

See the [installation docs](iSpy-Docs/docs/installation.mdx) for platform details.

## Documentation

Full documentation lives in [`iSpy-Docs/`](iSpy-Docs/) and is published at the project site. Topics:

- [Getting started](iSpy-Docs/docs/getting-started.mdx)
- [Installation](iSpy-Docs/docs/installation.mdx)
- [Configuration](iSpy-Docs/docs/configuration.mdx)
- [Coordinates and units](iSpy-Docs/docs/coordinates-and-distance.mdx)
- [Camera calibration](iSpy-Docs/docs/calibration.mdx)
- [Vision pipelines](iSpy-Docs/docs/vision-pipelines.mdx)
- [Tracking and targeting](iSpy-Docs/docs/tracking-and-targeting.mdx)
- [Robot integration (NetworkTables)](iSpy-Docs/docs/robot-integration.mdx)
- [Web UI and networking](iSpy-Docs/docs/web-ui-and-networking.mdx)
- [Performance and hardware](iSpy-Docs/docs/performance-and-hardware.mdx)
- [CLI reference](iSpy-Docs/docs/cli-reference.mdx)
- [Troubleshooting](iSpy-Docs/docs/troubleshooting.mdx)
- [Development and advanced config](iSpy-Docs/docs/development-and-advanced-config.mdx)

## Hardware

Recommended: Orange Pi 5 / 5 Pro (RK3588 NPU). Also runs on Raspberry Pi, Jetson, x86 Linux, macOS (Apple Silicon), and Windows.

## License

PolyForm Noncommercial 1.0.0. See [LICENSE](LICENSE). Commercial licensing inquiries welcome.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Community plugins: [iSpy-FRC-Plugins](https://github.com/aidan-j532/iSpy-FRC-Plugins).
