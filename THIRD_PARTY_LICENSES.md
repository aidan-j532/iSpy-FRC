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