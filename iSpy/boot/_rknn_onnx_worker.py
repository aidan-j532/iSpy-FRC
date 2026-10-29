"""Standalone ONNX -> RKNN builder for the isolated converter environment.

Runs as ``python -m iSpy.boot._rknn_onnx_worker <args.json>`` under the RKNN
converter environment's Python (see ``iSpy.boot._rknn_converter_env``).  The
full ``rknn-toolkit2`` stack is never imported into the main runtime
environment, so this worker deliberately imports *no* iSpy modules - it only
needs stdlib + ``rknn.api``.

Args (JSON):

- ``onnx``: path to the intermediate ONNX model.
- ``output``: destination ``.rknn`` path.
- ``quantize``: build with int8 quantization.
- ``target_platform``: Rockchip SoC string for ``rknn.config``.
- ``mean`` / ``std``: 3-element input normalization applied by the runtime.
- ``input_size_list``: optional NCHW input shape list, e.g. ``[[1, 3, 518, 518]]``.
- ``calibration``: optional ``dataset.txt`` path for quantization.

On success writes ``<args>.result.json`` with ``{"result": <path>}``; on
failure writes a structured ``{"error", "error_type", "traceback"}`` payload
and exits non-zero so the parent can surface the *real* cause.
"""

import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m iSpy.boot._rknn_onnx_worker <args.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(sys.argv[1]) as f:
        args = json.load(f)

    # The converter environment is provisioned specifically so this import
    # works - a failure here is a broken converter env, not a missing
    # dependency in the main runtime.
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        rknn.config(
            mean_values=[args.get("mean") or [0, 0, 0]],
            std_values=[args.get("std") or [255, 255, 255]],
            target_platform=args.get("target_platform") or "rk3588",
        )
        ret = rknn.load_onnx(
            model=args["onnx"], input_size_list=args.get("input_size_list")
        )
        if ret != 0:
            raise RuntimeError(f"rknn load_onnx failed with code {ret}")
        ret = rknn.build(
            do_quantization=bool(args.get("quantize")),
            dataset=args.get("calibration"),
        )
        if ret != 0:
            raise RuntimeError(f"rknn build failed with code {ret}")
        ret = rknn.export_rknn(args["output"])
        if ret != 0:
            raise RuntimeError(f"rknn export_rknn failed with code {ret}")
    finally:
        rknn.release()

    out_path = Path(sys.argv[1] + ".result.json")
    out_path.write_text(
        json.dumps({"result": args["output"], "python": sys.executable})
    )


def run() -> None:
    """Run main() and report failures as structured errors, never silent."""
    try:
        main()
    except Exception as exc:
        out_path = Path(sys.argv[1] + ".result.json") if len(sys.argv) == 2 else None
        error_payload = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        if out_path is not None:
            try:
                out_path.write_text(json.dumps(error_payload))
            except OSError:
                pass
        print(
            f"RKNN ONNX builder failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()