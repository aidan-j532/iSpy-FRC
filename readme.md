VisionCore FRC - YOLO-Based Vision Pipeline
================================================

VisionCore is a comprehensive FRC (FIRST Robotics Competition) vision pipeline library built on YOLOv8 object detection. It provides multi-format YOLO model support, automatic hardware optimization, multi-camera handling, game piece tracking, and FRC NetworkTables integration.

Quick Start
-----------

1. Setup project structure:
   python setup.py

2. Install dependencies:
   pip install -e .

3. Configure your setup:
   - Edit config.json
   - Set camera source and model path

4. Run pipeline:
   python -m VisionCore run

For detailed setup instructions, see SETUP.md or read QUICKSTART.md for a 5-minute guide.


Features
--------

Core Vision
- Multi-format YOLO support (PyTorch, OpenVINO, RKNN, ONNX, TFLite)
- Automatic model optimization for your hardware
- Real-time object detection with confidence thresholds
- Multi-camera support with frame synchronization

Tracking & Analysis
- Game piece tracking with state estimation
- Autonomous path planning
- Clustering-based tracking (DBSCAN)
- Custom tracker framework for extensibility

Integration
- FRC NetworkTables support for robot communication
- Web-based dashboard for monitoring
- Video recording and analysis capabilities
- System health reporting and metrics

Deployment
- Cross-platform support (Linux, Windows, macOS)
- systemd service installation on Linux
- Scheduled task setup on Windows
- Edge device optimization (RKNN for RockChip)


Project Structure
-----------------

VisionCore-Deploy/
|
+-- setup.py                  Setup script - run this first
+-- config.json              Main configuration file (edit this)
+-- pyproject.toml           Package metadata and dependencies
|
+-- config/                  Configuration files directory
+-- YoloModels/              YOLO models organized by format
|   +-- pytorch/
|   +-- openvino/
|   +-- rknn/
|   +-- onnx/
+-- Outputs/                 Detection results and analysis
+-- VideoRecordings/         Recorded video files
+-- RknnWheels/              RKNN optimization wheels
|
+-- VisionCore/              Main library code
|   +-- boot/                Boot and startup logic
|   +-- config/              Configuration handling
|   +-- core/                Main game loop
|   +-- vision/              Detection modules
|   +-- trackers/            Tracking implementations
|   +-- utilities/           Helper utilities
|   +-- web/                 Web interface
|   +-- validations/         System validation
|   +-- examples/            Usage examples


Installation
-------------

From Fresh Clone
1. Run setup script to create directories and configs:
   python setup.py

2. Install with pip:
   pip install -e .

Or use the provided installation scripts:

For deployment (minimal dependencies):
   bash install-deploy.sh

For development (includes ML frameworks):
   bash install-dev.sh

Requirements:
- Python 3.10 or higher
- pip package manager
- numpy >= 1.26.0
- opencv-python >= 4.8.0
- ultralytics >= 8.3.0

Optional for advanced features:
- torch >= 2.0.0 (for PyTorch models)
- For OpenVINO: Intel OpenVINO toolkit
- For RKNN: RockChip RKNN toolkit


Configuration
--------------

Main Configuration File: config.json

Example configuration:

{
  "unit": "meter",
  "use_network_tables": false,
  "app_mode": true,
  "debug_mode": true,
  "record_mode": false,
  "auto_opt": true,
  "vision_model": {
    "file_path": "yolov8n.pt",
    "input_size": [640, 640],
    "min_conf": 1
  },
  "camera_configs": {
    "Webcam": {
      "name": "Webcam",
      "source": 0,
      "fps_cap": 30,
      "subsystem": "field",
      "grayscale": false,
      "x": 0, "y": 0, "height": 0, "pitch": 0, "yaw": 0,
      "calibration": {
        "size": 0, "distance": 0, "game_piece_size": 0, "fov": 70
      },
      "pipeline": "object_detection"
    }
  }
}

Key Settings:
- unit: Distance unit (meter, inch, foot)
- auto_opt: Enable automatic model format selection
- vision_model.file_path: Path to YOLO model
- camera_configs: List of connected cameras
- debug_mode: Enable verbose logging
- record_mode: Enable video recording
- log_level: Optional. One of DEBUG, INFO, WARNING, ERROR. Controls console and file logging.
- log_file: Optional. Path to write logs (relative to repo root). If omitted, logs go to console only.

For complete configuration options, see config/README.md or VisionCore/examples/example_config.json


YOLO Models
-----------

Supported Formats:

Format      Extension   Use Case                    Performance
--------    ---------   --------                    -----------
PyTorch     .pt         Development, high accuracy  Highest (GPU)
OpenVINO    .xml/.bin   Intel CPUs, edge devices    Good
RKNN        .rknn       RockChip edge devices       Good
ONNX        .onnx       Cross-platform              Medium
TFLite      .tflite     Mobile/ARM edge             Lower

Model Organization:

Place models in YoloModels/ organized by format:

YoloModels/
+-- pytorch/nano/yolov8n.pt
+-- openvino/nano/
|   +-- yolov8n.xml
|   +-- yolov8n.bin
+-- rknn/nano/yolov8n.rknn
+-- onnx/nano/yolov8n.onnx

Auto-Optimization:

With "auto_opt": true in config.json, VisionCore will automatically:
1. Detect your hardware capabilities
2. Find the best compatible model format
3. Load that model for optimal performance

Priority order: RKNN > OpenVINO > ONNX > PyTorch > TFLite

Model Conversion Behaviour:

When a PyTorch (.pt) model is provided and `auto_opt` is enabled, VisionCore will
attempt to convert the model to the best format for the host (for example OpenVINO
on Intel CPUs). Converted artifacts are placed under the organized `YoloModels/`
tree following the pattern: `YoloModels/<format>/<size>/`. If an exported file already
exists it will be reused (cached). If conversion fails, VisionCore will fall back to
using the original .pt model.


Usage
-----

Command Line Interface

python -m VisionCore boot      Run full boot sequence
python -m VisionCore run       Run vision pipeline
python -m VisionCore setup     Setup project structure
python -m VisionCore config    Show current configuration
python -m VisionCore models    List available models
python -m VisionCore help      Show help message

Entry Points (if installed):
visioncore-boot                Boot entry point
visioncore-run                 Run entry point


Python API

from VisionCore.core.game_loop import main
from VisionCore.boot.boot import on_boot
from VisionCore.config.VisionCoreConfig import VisionCoreConfig

# Boot sequence with validation
on_boot()

# Run main pipeline
main()

# Load configuration
config = VisionCoreConfig("config.json")
model_path = config.get("vision_model")["file_path"]


Web Interface

When running with app_mode enabled, access the web interface:

URL: http://localhost:5000

Features:
- Live camera view with detections
- Detection metrics and performance graphs
- System health monitoring
- Configuration view


Multi-Camera Support

Configure multiple cameras in config.json:

{
  "camera_configs": {
    "FrontCam": {
      "source": 0,
      "subsystem": "field",
      "fps_cap": 30
    },
    "RearCam": {
      "source": 1,
      "subsystem": "intake",
      "fps_cap": 30
    },
    "IPCamera": {
      "source": "http://192.168.1.100:8080/video",
      "subsystem": "turret",
      "fps_cap": 15
    }
  }
}

All cameras are synchronized and processed together.


FRC Integration
---------------

NetworkTables Configuration

Enable robot communication:

{
  "use_network_tables": true,
  "network_tables_ip": "10.22.7.2"
}

NetworkTableHandler:

from VisionCore.utilities.NetworkTableHandler import NetworkTableHandler

handler = NetworkTableHandler("10.22.7.2")
handler.send_vision_data(detections)
handler.send_metrics(fps, latency)


Custom Development
------------------

Create Custom Tracker

from VisionCore.trackers import CustomDBScan

class MyTracker(CustomDBScan):
    def __init__(self):
        super().__init__()
    
    def process(self, detections, frame):
        # Your tracking logic
        return tracked_objects

Register in pyproject.toml:

[project.entry-points.visioncore_trackers]
my_tracker = "mymodule:MyTracker"


Create Custom Vision Module

from VisionCore.vision.Camera import Camera

class CustomVision(Camera):
    def __init__(self, config):
        super().__init__(config)
    
    def process(self, frame):
        # Your processing logic
        return processed_frame, results


Troubleshooting
---------------

Setup Issues

Problem: FileNotFoundError: config directory not found
Solution: Run python setup.py to create required directories

Problem: No YOLO models found
Solution: Place models in YoloModels/ and run python setup.py

Problem: Camera not detected
Solution: Check camera source ID (usually 0 for default)
         On Linux: ls -la /dev/video*
         Test with: python -c "import cv2; cap = cv2.VideoCapture(0)"


Runtime Issues

Problem: Low detection performance
Solution: Use optimized model format (OpenVINO > ONNX)
         Enable auto_opt in config
         Reduce input_size to [320, 320]

Problem: Module import errors
Solution: Verify installation: pip install -e .
         Check Python version: 3.10 or higher
         Run setup.py: python setup.py

Problem: Web interface not loading
Solution: Check Flask installation
         Verify port 5000 is available
         Check firewall settings


Documentation
--------------

QUICKSTART.md       5-minute quick start guide
SETUP.md            Complete setup and installation guide
LIBRARY_GUIDE.md    Comprehensive library reference
QUICK_REFERENCE.md  Command and configuration reference
IMPROVEMENTS.md     Summary of improvements and features
INDEX.md            Documentation navigation and index
config/README.md    Detailed configuration options


Examples
--------

Basic Usage:

from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera

config = VisionCoreConfig("config.json")
camera = ObjectDetectionCamera("MainCamera", config)

while True:
    frame, detections = camera.get_frame_and_detections()
    
    for detection in detections:
        x, y, w, h = detection.bbox
        confidence = detection.conf
        class_name = detection.class_name
        print(f"Detected {class_name} at ({x}, {y}) with confidence {confidence}")


Multi-Camera Processing:

from VisionCore.utilities.MultipleCameraHandler import MultipleCameraHandler

handler = MultipleCameraHandler(config)

for frames in handler.get_frames():
    for camera_name, frame in frames.items():
        # Process each camera's frame
        detections = process(frame)
        print(f"Processed {camera_name}")


See VisionCore/examples/ for more complete examples.


Performance Tips
----------------

- Use optimized model formats (OpenVINO, RKNN) instead of PyTorch
- Enable auto_opt to automatically select best format
- Reduce input_size from [640, 640] to [320, 320] for speed
- Use grayscale: true for grayscale cameras
- Lower fps_cap if system is overwhelmed
- Disable record_mode when not needed
- Single camera faster than multiple cameras


Dependencies
------------

Core Dependencies:
- numpy >= 1.26.0
- opencv-python >= 4.8.0
- protobuf >= 4.25.0
- ultralytics >= 8.3.0
- requests >= 2.31.0
- psutil >= 7.2.0
- Pillow >= 12.0.0
- flask >= 3.0
- scikit-learn >= 1.3.0
- pyntcore >= 2024.0.0
- robotpy >= 2024.0.0

Optional (Development):
- torch >= 2.0.0
- torchvision >= 0.15.0
- tensorflow >= 2.13.0
- onnx >= 1.16.0
- onnxruntime >= 1.16.0
- matplotlib >= 3.8.0
- jupyter
- tensorboard

To install with optional dependencies:
pip install -e ".[dev]"


License
-------

VisionCore is licensed under the GPL-3.0 license.
See LICENSE file for details.


Contributing
------------

To contribute to VisionCore:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For major changes, please open an issue first to discuss proposed changes.


Support
-------

For issues, questions, or suggestions:
- Check existing documentation in SETUP.md or LIBRARY_GUIDE.md
- Review examples in VisionCore/examples/
- Check troubleshooting section above
- Open an issue on GitHub


Getting Help
------------

1. Read QUICKSTART.md for basic setup (5 minutes)
2. Read SETUP.md for complete guide (15 minutes)
3. Check QUICK_REFERENCE.md for commands
4. See VisionCore/examples/ for code samples
5. Review LIBRARY_GUIDE.md for architecture details
6. Check troubleshooting section in SETUP.md


Getting Started
---------------

1. Clone the repository
2. Run: python setup.py
3. Edit: config.json with your setup
4. Add: YOLO models to YoloModels/
5. Run: python -m VisionCore run

For more details, see QUICKSTART.md


Latest Changes
--------------

Recent improvements include:
- Automatic project setup script
- Configuration file creation and validation
- YOLO model organization and validation
- Comprehensive documentation
- CLI command interface
- Model format auto-detection
- Multi-format model support


System Requirements
-------------------

Operating Systems:
- Linux (recommended for deployment)
- macOS
- Windows

Hardware:
- CPU: Modern multi-core processor (Intel i5+, AMD Ryzen 5+)
- RAM: 4GB minimum, 8GB+ recommended
- GPU: Optional but recommended for PyTorch models

Python:
- Python 3.10 or higher
- pip package manager


Version Information
-------------------

Project: VisionCore FRC
Version: 1.3.7.2
Python: >= 3.10
License: GPL-3.0


Contact & Resources
-------------------

- GitHub: https://github.com/aidan-j532/VisionCore-Deploy
- Documentation: See included .md files
- Examples: VisionCore/examples/
- Questions: See troubleshooting and FAQ sections


Quick Commands Reference
------------------------

Setup and Installation:
python setup.py                  -- Initialize project
pip install -e .                 -- Install locally
pip install -e ".[dev]"          -- Install with dev tools

Running:
python -m VisionCore run         -- Start pipeline
python -m VisionCore boot        -- Boot with service setup
visioncore-run                   -- Run (if installed)

Information:
python -m VisionCore config      -- Show configuration
python -m VisionCore models      -- List models
python -m VisionCore help        -- Show help

Testing:
python -c "import VisionCore; print('OK')" -- Test import


Next Steps
----------

After installation:

1. Review config.json and customize for your setup
2. Add YOLO models to YoloModels/ directory
3. Test with: python -m VisionCore run
4. Access web interface: http://localhost:5000
5. Review examples: VisionCore/examples/
6. Implement custom trackers if needed
7. Deploy to robot or edge device

For detailed instructions, refer to SETUP.md or QUICKSTART.md
