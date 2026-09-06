# Third-Party Licenses

iSpy is licensed under PolyForm Noncommercial 1.0.0 (see LICENSE). The
following third-party components are distributed under their own terms:

## RKNN-Toolkit2 wheels
Mirrored from https://github.com/airockchip/rknn-toolkit2 for install
convenience (see iSpy/vision/optimizer.py, _RKNN_FULL_WHEELS).
© Rockchip Electronics Co., Ltd. Licensed under BSD-3-Clause.
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

## Bundled frontend libraries
Vendored under iSpy/web/static/vendor/ and served to the web dashboard.
The text below reflects only what is discoverable in the file headers
(version strings / license banners); entries without a visible license
header are flagged for manual verification.

- **three.module.js** - Three.js REVISION r164 (version string in file)
  - LICENSE CONFIRMED: MIT - the file header carries
    `// SPDX-License-Identifier: MIT` and
    `Copyright 2010-2024 Three.js Authors`.
- **chart.umd.min.js** - Chart.js 4.4.0 (version string in file:
  `version="4.4.0"`)
  - LICENSE UNKNOWN - VERIFY MANUALLY - no license banner in the
    minified file.
- **OrbitControls.js** - three.js orbit controls example addon (no
  version string in file)
  - LICENSE UNKNOWN - VERIFY MANUALLY - no license banner in the file.