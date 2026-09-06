# iSpy-FRC Product Strategy

---

## 1. Competitive Landscape

| | **Limelight 4** | **PhotonVision** | **iSpy-FRC** |
|---|---|---|---|
| **Price** | $350+ hardware + camera | Free (software only) | Free (software only) |
| **Hardware** | All-in-one (Hailo-8 NPU + camera) | Runs on Orange Pi 5 / RPi / Jetson | Runs on Orange Pi 5 / RPi / Jetson / x86 / Mac / Windows |
| **Setup** | Plug-and-profit (pre-flashed) | Manual: flash OS, install, configure | Flash image OR manual install; web UI config |
| **AI Backend** | Hailo-8 HEF (fixed) | Ultralytics YOLO / OpenCV | RKNN, ONNX, OpenVINO, TFLite, CoreML, TensorRT, Hailo, Coral TPU |
| **Model Support** | Limelight-trained only (Hailo format) | Any YOLO .pt | Any YOLO .pt + auto-conversion to 8+ formats |
| **Field Coordinates** | Yes (MegaTag2, multi-cam) | Yes (single-cam triangulation) | Yes (ground-plane ray + multi-cam triangulation) |
| **Pose Estimation** | No (detect only) | Limited | Yes (PnP with keypoints, rigid/flexible modes) |
| **Web Dashboard** | Basic (crosshair, tuning) | Moderate (camera view, tuning) | Rich (12+ modules: cameras, models, datasets, 3D viewer, metrics, health, logs, recommendations, onboarding, setup wizard, settings) |
| **AprilTag Tracking** | Yes (MegaTag2) | Yes | Yes |
| **Multi-Camera** | Yes (hardware sync) | Limited | Yes (threaded, ray-triangulation merge) |
| **Depth Estimation** | No | No | Yes (Depth Anything V2) |
| **Open-Source** | No | Yes (GPLv3) | Source-available (PolyForm Noncommercial 1.0.0) |
| **Model Training** | No | No | YOLO-World (zero-shot), dataset tools, Roboflow integration |

### Where iSpy Wins
- **Hardware agnostic**: runs on literally anything (Rockchip NPU, NVIDIA, Intel, Apple Silicon, Coral TPU, Hailo, Google TPU, x86 CPU)
- **Deepest model pipeline**: auto-detect hardware, auto-convert, auto-benchmark, background optimization
- **Most feature-rich web dashboard**: 12+ modules vs 1-2 for competitors
- **Multi-camera triangulation**: first-class, ray-based, not just "use two cameras"
- **Pose estimation + PnP**: no competitor does this
- **Plugin architecture**: extensible without touching core
- **Depth Anything V2**: monocular depth, unique capability
- **Dataset management + training integration**: built into the web UI

### Where iSpy Loses
- **Not plug-and-play**: Limelight is "buy box, plug in, done." iSpy requires setup even with the flash image
- **Community size**: PhotonVision has more FRC teams using it = more docs, YouTube videos, forum posts
- **Brand recognition**: "Use a Limelight" is a verb in FRC. "Use iSpy" doesn't exist yet
- **Hardware bundle**: no iSpy-branded camera+board combo you can buy on AndyMark
- **Documentation depth**: competitors have more tutorials, wiring guides, code examples

---

## 2. Strategic Position: "The Power Tool"

**iSpy should NOT try to be Limelight.** Limelight's moat is simplicity + hardware bundle. You cannot out-plug-and-play a $350 pre-flashed box.

**iSpy should NOT try to be PhotonVision.** PhotonVision's moat is community momentum + "free alternative to Limelight" positioning.

**iSpy should be the Blender of FRC vision.** Blender lost the "easy" fight to Maya/3ds Max. It won by being so powerful that professionals adopted it, then education followed. iSpy should target:
1. **The 5% of teams that push vision hard** (multi-cam, pose, custom models, advanced tuning)
2. **The teams that outgrow Limelight/PhotonVision** and need more
3. **Non-FRC robotics** (FIRST Tech Challenge, VEX, research, agriculture, industrial)

---

## 3. Honest Weaknesses (Things That Hurt Today)

1. **First-boot experience is hostile**: even with flash image, the user stares at a black terminal and has to SSH in. No visual feedback.
2. **Config is JSON by hand**: the web UI exists but config.json is still the source of truth. Confusing when UI and file disagree.
3. **Calibration is multi-step wizard hell**: focal length, ChArUco board, distance measurement, game piece size... competitors just work.
4. **No standardized robot-side library**: Limelight has `LimelightHelpers.java`. iSpy publishes raw `FuelStruct[]` and the user figures it out.
5. **Model selection is confusing**: users don't know what "RKNN", "ONNX", "TFLite" mean or why they'd pick one.
6. **No field testing tools**: no way to visualize "how accurate are my coordinates in real field conditions?"
7. **Thread safety issues**: some modules access shared state without locks (health reporter, some web endpoints).
8. **No automated regression tests for vision accuracy**: if triangulation math breaks, nobody knows until competition.

---

## 4. Feature Ideas (25 Features, Scored)

Scoring: **Impact** (1-5, how much it moves the needle for adoption/retention) x **Effort** (1-5, 1=trivial, 5=months). Score = Impact/Effort.

| # | Feature | Impact | Effort | Score | Category |
|---|---------|--------|--------|-------|----------|
| 1 | **One-Command Robot-Side Library** (Java/C++ `iSpyClient` with auto-discovery, typed structs, example commands) | 5 | 3 | 1.67 | DX |
| 2 | **Zero-Config First Boot** (LED ring status, web-based setup wizard on first power-on, no SSH needed) | 5 | 4 | 1.25 | Onboarding |
| 3 | **Camera Auto-Discovery** (plug in USB camera, it appears in UI automatically with recommended settings) | 4 | 2 | 2.00 | Onboarding |
| 4 | **One-Click Model Train** (capture images in web UI -> label with YOLO-World -> train YOLOv8 nano -> deploy, all in-browser) | 5 | 5 | 1.00 | Model Lifecycle |
| 5 | **Accuracy Dashboard** (field-test mode: drive robot to known positions, compare vision output vs ground truth, visualize error heatmap) | 4 | 3 | 1.33 | Debugging |
| 6 | **Automatic Calibration** (hold ChArUco board in front of camera, auto-capture, auto-solve, auto-save intrinsics + extrinsics) | 4 | 3 | 1.33 | Calibration |
| 7 | **Performance Budgets** (set FPS/latency targets per camera, alert when exceeded, auto-downscale resolution to maintain budget) | 3 | 2 | 1.50 | Observability |
| 8 | **Live Comparison Mode** (side-by-side: iSpy vs Limelight vs raw camera, same field, same moment, for benchmarking) | 3 | 2 | 1.50 | Debugging |
| 9 | **Plugin Marketplace** (browse/install community plugins from web UI, versioned, with reviews) | 3 | 4 | 0.75 | Ecosystem |
| 10 | **SWAT/AdvantageKit Integration** (publish structured logging to AdvantageKit, auto-generate log waypoints) | 4 | 2 | 2.00 | Integration |
| 11 | **NetworkTables v5 Support** (future-proof for when FRC migrates from NT4) | 2 | 3 | 0.67 | Compatibility |
| 12 | **Dynamic Model Switching** (hot-swap models mid-match without restarting pipeline, e.g., auto-switch to "close-range" model when near speaker) | 4 | 4 | 1.00 | Vision |
| 13 | **Detection Confidence Visualization** (color-coded bounding boxes, confidence heatmaps overlaid on field) | 2 | 1 | 2.00 | Debugging |
| 14 | **Multi-Model Ensembling** (run 2 models simultaneously, merge detections for higher accuracy at cost of FPS) | 3 | 3 | 1.00 | Vision |
| 15 | **Roboflow Dataset Sync** (two-way: upload captures for training, pull trained models, all from web UI) | 3 | 3 | 1.00 | Model Lifecycle |
| 16 | **FRC Match Replay** (record full match video + detections, replay with timeline scrubbing, overlay detection events) | 3 | 3 | 1.00 | Debugging |
| 17 | **Auto-Exposure / Auto-White-Balance Tuning** (optimize camera ISP settings from web UI for field lighting) | 3 | 2 | 1.50 | Calibration |
| 18 | **GStreamer Pipeline Input** (accept RTSP, UDP multicast, USB3, CSI camera streams via GStreamer) | 3 | 3 | 1.00 | Compatibility |
| 19 | **Scheduled Model Updates** (OTA model push: coach uploads new model, all cameras update on next boot) | 3 | 4 | 0.75 | DX |
| 20 | **Coordinate System Validator** (visual tool to verify camera mount positions by detecting known field landmarks) | 3 | 2 | 1.50 | Calibration |
| 21 | **Simulated Field Mode** (feed simulated FRC field images into pipeline for off-season development and testing) | 2 | 4 | 0.50 | DX |
| 22 | **Telemetry API** (REST/GraphQL API exposing all vision data, detections, health metrics for custom dashboards) | 3 | 2 | 1.50 | DX |
| 23 | **Alert System** (configurable alerts: "camera offline", "FPS dropped below 15", "no detections for 30s" -> push to phone via webhook/NT) | 3 | 2 | 1.50 | Observability |
| 24 | **Pose Model PnP Wizard** (step-by-step UI for keypoint annotation + object_points setup, currently manual JSON) | 4 | 2 | 2.00 | Calibration |
| 25 | **Automatic Aspect-Ratio Filtering** (ML-based: learn which detections are valid game pieces vs noise from field images) | 2 | 3 | 0.67 | Vision |

---

## 5. Top 10 Features (Ranked by Score)

| Rank | Feature | Score | Why |
|------|---------|-------|-----|
| 1 | **Camera Auto-Discovery** | 2.00 | Eliminates the #1 friction point: "I plugged in a camera and nothing happened" |
| 2 | **SWAT/AdvantageKit Integration** | 2.00 | Instant credibility with top FRC teams who already use AdvantageKit |
| 3 | **Pose Model PnP Wizard** | 2.00 | Unlocks the killer feature (pose estimation) for teams who currently can't figure out JSON keypoints |
| 4 | **Detection Confidence Visualization** | 2.00 | Trivial to build, huge UX improvement for tuning |
| 5 | **One-Command Robot-Side Library** | 1.67 | Makes iSpy usable from Java/C++ without reading source code |
| 6 | **Performance Budgets** | 1.50 | Teams need to know "will this work at competition?" before they get there |
| 7 | **Auto-Exposure / White-Balance Tuning** | 1.50 | Lighting varies wildly between venues; this is a silent killer of detection accuracy |
| 8 | **Coordinate System Validator** | 1.50 | If your camera mount is 2 inches off, every detection is wrong. Visual verification catches this. |
| 9 | **Alert System** | 1.50 | "Camera died mid-match" should never be discovered during eliminations |
| 10 | **Telemetry API** | 1.50 | Enables custom dashboards, data logging, and integration with tools that don't use NetworkTables |

---

## 6. Five 10x Features

These are features that, if built well, make iSpy 10x better than the competition in that specific dimension:

### 10x #1: **The FRC Vision Operating System**
Turn iSpy from "vision pipeline" into "platform." Camera management, model lifecycle, calibration, testing, deployment, monitoring -- all from one web UI. No competitor has this. Limelight is "camera + feed." PhotonVision is "camera + feed + tuning." iSpy should be "the entire vision stack."

**Components**: Auto-discovery + setup wizard + model marketplace + calibration + accuracy dashboard + OTA updates + telemetry API.

### 10x #2: **Zero-Shot Field Adaptation**
Use YOLO-World + Depth Anything V2 to let teams point their camera at any field element and say "detect that" without training a model. At competition, field elements change (bumpers, tape, lighting). A model trained on your shop won't work perfectly. iSpy could re-adapt in real-time.

**Components**: YOLO-World zero-shot detection + few-shot fine-tuning from web UI + automatic confidence calibration per-venue.

### 10x #3: **Multi-Camera Perception System**
Not just "run 2 cameras." True sensor fusion: stereo depth, object persistence across cameras, occlusion handling, unified field map. iSpy already has the `MultipleCameraHandler` with ray triangulation -- this is the seed.

**Components**: Auto-extrinsics calibration between cameras + stereo depth as alternative to monocular + occlusion-aware tracking + per-object identity persistence across cameras.

### 10x #4: **Match Intelligence**
Record every detection in every match. After the match, visualize: "here's where every game piece was, when your robot saw it, what it did." Post-match analysis tooling. No FRC vision tool does this.

**Components**: Match recording (detections + robot pose + timestamps) + replay viewer + "detection coverage" heatmap + "time to detect" metrics.

### 10x #5: **Cross-Platform Model Benchmarking**
One click: "benchmark this model on my hardware." Shows FPS, latency, accuracy, memory. Compare across formats (ONNX vs RKNN vs TensorRT). Auto-recommend the best format. iSpy already has `AutoOpt.py` -- extend it into a full benchmarking suite accessible from the web UI.

**Components**: Web UI benchmark runner + results history + format comparison chart + hardware-specific recommendations + "competition mode" (lock to proven config).

---

## 7. Features to AVOID (Bad Ideas)

| Feature | Why Not |
|---------|---------|
| **Build a Limelight killer (hardware)** | You're a software project. Hardware margins are terrible. $350 Limelight includes manufacturing, distribution, support. Don't compete there. |
| **Real-time object tracking with Kalman filters / UKF** | Overkill for FRC. Game pieces don't have predictable trajectories (human pickup, bouncing, collision). EMA + DBSCAN is already good enough. The complexity isn't worth it. |
| **Custom neural network architecture** | Don't reinvent YOLO. Stay on Ultralytics YOLOv8/v11/v26 and let them do the architecture research. Your value is the pipeline around the model, not the model. |
| **GPT/CV integration for "describe what you see"** | Gimmick. Adds latency, cost, and distraction. FRC teams need (x, y) coordinates, not "I see a coral." |
| **Blockchain-based model licensing** | Obviously. But also: don't build any DRM or model protection. Open models = community trust. |
| **VR/AR field visualization** | Cool demo, zero practical value for FRC. A 3D viewer is enough (you already have `viewer3d.py`). |
| **Robot path planning** | That's the robot code team's job. iSpy provides (x, y) of objects. The robot code uses them. Stay in your lane. |
| **Automatic robot driving** | See above. Way out of scope, massive liability, not what vision pipelines do. |
| **Video streaming to scouting apps** | Nice-to-have but not core. Scouting apps (Statbotics, TBA) don't consume video. They consume match data. Don't build a streaming server. |
| **On-device training** | Orange Pi 5 has 8GB RAM and an RK3588 NPU. Training YOLO on-device will be painfully slow. Offload training to cloud/colab, deploy to device. |

---

## 8. Feature Classification

### FRC-Specific (Only Useful in FRC)
- NetworkTables publishing (FuelStruct)
- FRC coordinate system (field-relative x/y)
- roboRIO auto-discovery
- Match recording and replay
- FRC game piece-specific calibration (known object size)
- AdvantageKit integration
- FIRST-provided field calibration targets

### Robotics (Useful Beyond FRC)
- Multi-camera triangulation
- AprilTag / fiducial tracking
- Pose estimation (PnP)
- Ground-plane intersection
- Plugin architecture
- Auto-backend selection (RKNN, TensorRT, etc.)
- Web-based camera management
- Model lifecycle (train -> convert -> deploy -> monitor)

### Core Computer Vision
- YOLO inference across 8+ backends
- Depth estimation (Depth Anything V2)
- Optical flow
- Camera calibration (ChArUco, auto layout detection)
- DBSCAN clustering
- EMA smoothing
- YOLO-World zero-shot detection

### General (Not Even Vision-Specific)
- Web dashboard with modules
- Health monitoring and alerting
- Telemetry API
- Plugin marketplace
- OTA updates
- Performance benchmarking
- Dataset management

---

## 9. Phased Roadmap

### Phase 1: "Fix the Basics" (Months 1-2)
*Goal: Make iSpy usable by a 16-year-old who's never touched Linux*

| Feature | Priority | Effort |
|---------|----------|--------|
| Camera Auto-Discovery | P0 | 2 weeks |
| Zero-Config First Boot (LED ring status + web wizard) | P0 | 3 weeks |
| One-Command Robot-Side Library (Java `iSpyClient`) | P0 | 2 weeks |
| Detection Confidence Visualization | P1 | 3 days |
| Alert System (camera offline, FPS drop) | P1 | 1 week |
| Fix thread safety issues in health/web modules | P0 | 3 days |

**Exit criteria**: A new team can go from "box in hand" to "seeing detections on robot dashboard" in under 30 minutes, with zero SSH.

### Phase 2: "Power Features" (Months 3-4)
*Goal: Give advanced teams reasons to switch from Limelight*

| Feature | Priority | Effort |
|---------|----------|--------|
| Pose Model PnP Wizard | P0 | 2 weeks |
| SWAT/AdvantageKit Integration | P1 | 2 weeks |
| Auto-Exposure / White-Balance Tuning | P1 | 1 week |
| Coordinate System Validator | P1 | 1 week |
| Telemetry API (REST) | P1 | 1 week |
| Performance Budgets + auto-downscale | P2 | 2 weeks |

**Exit criteria**: A team doing pose estimation can set it up without reading source code. AdvantageKit teams get structured vision logs for free.

### Phase 3: "Intelligence" (Months 5-6)
*Goal: Make iSpy the most capable vision platform in FRC*

| Feature | Priority | Effort |
|---------|----------|--------|
| Zero-Shot Field Adaptation (YOLO-World + venue calibration) | P1 | 3 weeks |
| Multi-Camera Perception (auto-extrinsics, stereo depth) | P2 | 4 weeks |
| Dynamic Model Switching (hot-swap mid-match) | P2 | 2 weeks |
| Match Intelligence (record, replay, analysis) | P2 | 3 weeks |

**Exit criteria**: iSpy can adapt to a new venue's lighting and field layout in minutes, not hours. Multi-camera setups auto-calibrate.

### Phase 4: "Platform" (Months 7-9)
*Goal: Make iSpy a self-sustaining ecosystem*

| Feature | Priority | Effort |
|---------|----------|--------|
| Model Marketplace (browse, install, rate) | P2 | 4 weeks |
| OTA Model Updates (coach pushes, all cameras pull) | P2 | 2 weeks |
| Roboflow Two-Way Sync | P2 | 2 weeks |
| Cross-Platform Benchmarking Suite | P1 | 3 weeks |
| Simulated Field Mode | P3 | 4 weeks |

**Exit criteria**: iSpy has a community-contributed model library. Teams can benchmark any model on any hardware from the web UI.

---

## 10. Single Strongest Strategic Direction

**Build "The FRC Vision Operating System."**

Not a camera driver. Not a model runner. Not a calibration tool. The **entire stack**, from "I bought a camera" to "I have field-tested, battle-hardened vision at competition."

The execution path:

1. **Phase 1** eliminates the setup tax (the reason most teams don't try iSpy)
2. **Phase 2** unlocks the power features (the reason advanced teams switch)
3. **Phase 3** creates capabilities no competitor has (the reason people talk about iSpy)
4. **Phase 4** builds the ecosystem (the reason iSpy becomes self-sustaining)

The key insight: **Limelight and PhotonVision are camera products. iSpy should be a vision platform.** A camera captures frames. A vision platform tells you where things are, how accurate that information is, whether it's working, and what to do when it breaks.

Your unfair advantage is the plugin architecture + web dashboard + hardware abstraction. No competitor has all three. If you build the "OS" layer on top of what you already have, iSpy becomes the default choice for any team that takes vision seriously.

The metric that matters: **"How many FRC teams are running iSpy at their first event?"** Everything in this roadmap serves that number.
