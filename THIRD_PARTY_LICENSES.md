# Third-Party Licenses

iSpy is licensed under PolyForm Noncommercial 1.0.0 (see LICENSE). The
following third-party components are distributed under their own terms:

## RKNN-Toolkit2 wheels
Mirrored from https://github.com/airockchip/rknn-toolkit2 for install
convenience (see iSpy/vision/optimizer.py, _RKNN_FULL_WHEELS). These wheels
are distributed under Rockchip's RKNN SDK License, not BSD-3-Clause. That
license grants redistribution of modifications or derivative works solely for
design, development, and testing of applications compatible with Rockchip
products, and it contains separate third-party notices, export-control, and
termination conditions. Review the complete terms before redistributing the
wheels or using them outside that scope.
© Rockchip Electronics Co., Ltd.
Full text: https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE

## Ultralytics YOLO pretrained weights (optional download)
_default_detect.pt and _default_pose.pt, when downloaded, are stock
Ultralytics pretrained checkpoints obtained from
https://github.com/ultralytics/assets/releases. They are NOT bundled in
this repository and are licensed separately under AGPL-3.0 by Ultralytics,
independent of iSpy's own PolyForm Noncommercial license. See
https://github.com/ultralytics/ultralytics/blob/main/LICENSE.
iSpy never imports Ultralytics code at runtime (see the subprocess
isolation architecture in iSpy/boot/_convert_worker.py); Ultralytics
weights and the optional `[optimizer]` build-time dependency are the only
AGPL-touching pieces, and neither ships inside the PolyForm-licensed
codebase. When the weights are downloaded on demand, a THIRD_PARTY_NOTICE.txt
is written next to them in YoloModels/pytorch/.

## iSpy fuel-detect default model (optional download)
_default_v26_detect_for_fuel.pt is a checkpoint trained by the iSpy project
owner using Ultralytics code and training. It is therefore an
Ultralytics-derived AGPL-3.0 model, not an independently relicensable
PolyForm asset. The project's own training contribution is released under
AGPL-3.0 with the checkpoint. It is downloaded on demand from the project's
own GitHub release
(https://github.com/aidan-j532/iSpy-FRC/releases/download/Fuel_Detect_Model/fuel_detection_v26.pt)
and, like the Ultralytics checkpoints, is not bundled inside the
PolyForm-licensed codebase.

## Depth Anything V2 Small (optional download)
The `depth-anything/Depth-Anything-V2-Small-hf` checkpoint (24.8M params,
DPT/DINOv2 architecture, Lihe Yang et al.) is downloaded on demand via the
transformers `from_pretrained` API into YoloModels/huggingface/ (see
iSpy/vision/pipelines/depth_anything.py). It is NOT bundled in this
repository. The model card declares this checkpoint under the Apache-2.0
license, unlike Depth Anything V2 Base/Large which are CC-BY-NC-4.0, so V2
Small carries no noncommercial restriction. See
https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf.

## Bundled frontend libraries
Vendored under iSpy/web/static/vendor/ and served to the web dashboard.
The text below reflects only what is discoverable in the file headers
(version strings / license banners); entries without a visible license
header are flagged for manual verification.

- **three.module.js** - Three.js REVISION r164
  - LICENSE CONFIRMED: MIT - the file header carries
    `SPDX-License-Identifier: MIT` and it is byte-identical to the
    official `three@0.164.0/build/three.module.js` npm artifact
    (modulo CRLF line endings).
- **OrbitControls.js** - three.js r164 orbit controls example addon
  - LICENSE CONFIRMED: MIT - byte-identical to the official
    `three@0.164.0/examples/jsm/controls/OrbitControls.js` npm
    artifact (modulo CRLF line endings).
- **chart.umd.min.js** - Chart.js 4.4.0, no license banner in the minified
  file, source of origin previously unknown
  - LICENSE CONFIRMED: MIT - replaced with the official
    `chart.js@4.4.0/dist/chart.umd.min.js` npm artifact (sha256
    0e2326c686...aac6abff0) on 2026-09-06. Chart.js is MIT licensed
    (https://github.com/chartjs/Chart.js/blob/v4.4.0/LICENSE.md).
- **lines/** (LineSegments2.js, LineMaterial.js, LineSegmentsGeometry.js) -
  three.js r164 line-segment addon used for OBB track overlays in viewer3d
  - LICENSE CONFIRMED: MIT - body text byte-identical to the official
    `three@0.164.0/examples/jsm/lines/*` npm artifacts (modulo CRLF line
    endings); each file is annotated with the same SPDX MIT banner as
    three.module.js (sha256 of the annotated vendored copies:
    LineSegments2.js 95e71f7ca10c959a, LineMaterial.js 09a3697b4bba6f50,
    LineSegmentsGeometry.js f1bbc6deb767bff2).