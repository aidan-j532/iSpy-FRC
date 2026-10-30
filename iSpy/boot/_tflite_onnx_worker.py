"""Standalone ONNX -> TFLite builder for the isolated converter environment.

Runs as ``python -m iSpy.boot._tflite_onnx_worker <args.json>`` under the TFLite
converter environment's Python (see :mod:`iSpy.boot.converter_envs`).  The
TensorFlow/onnx2tf build stack is never imported into the main runtime
environment, so this worker deliberately imports no iSpy modules - it only
needs stdlib plus ``onnx2tf``/TensorFlow from the converter env.

Args (JSON):

- ``onnx``: path to the intermediate ONNX model.
- ``output``: destination ``.tflite`` path.

``onnx2tf`` writes a sibling named ``<onnx-stem>_float32.tflite`` (the exact
suffix can vary between versions), so the worker locates that file and moves it
to ``output``.  On success it writes ``<args>.result.json`` with
``{"result": <path>, "python": <interpreter>}``; on failure it writes a
structured ``{"error", "error_type", "traceback"}`` payload and exits non-zero
so the parent can surface the real cause.
"""

import json
import sys
import traceback
from pathlib import Path


def _generated_tflite(output: Path, onnx: Path) -> Path:
    preferred = output.parent / f"{onnx.stem}_float32.tflite"
    if preferred.is_file():
        return preferred

    # Older/newer onnx2tf releases can use a different suffix.  Restrict the
    # fallback to this model's stem so we never pick up another model that is
    # being converted into the same YoloModels/tflite directory concurrently.
    candidates = sorted(
        (
            path
            for path in output.parent.glob(f"{onnx.stem}*.tflite")
            if path.is_file() and path != output
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise RuntimeError(
        f"onnx2tf produced no .tflite file for {onnx.name} in {output.parent}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m iSpy.boot._tflite_onnx_worker <args.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(sys.argv[1]) as f:
        args = json.load(f)

    onnx_path = Path(args["onnx"])
    output_path = Path(args["output"])
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX source not found: {onnx_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The TFLite converter environment is provisioned specifically so this
    # import works.  Never import onnx2tf/TensorFlow in the runtime process.
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_path.parent),
        non_verbose=True,
    )

    generated = _generated_tflite(output_path, onnx_path)
    if generated != output_path:
        generated.replace(output_path)
    if not output_path.is_file():
        raise RuntimeError(
            f"TFLite build reported success but produced no artifact at "
            f"{output_path}"
        )

    out_path = Path(sys.argv[1] + ".result.json")
    out_path.write_text(
        json.dumps({"result": str(output_path), "python": sys.executable})
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
            f"TFLite ONNX builder failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
