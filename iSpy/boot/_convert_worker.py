"""Subprocess worker that runs model conversion in an isolated environment.

Launched as ``python -m iSpy.boot._convert_worker <args.json>`` by
``iSpy.vision.optimizer._convert_model_subprocess``.  The interpreter used is
the RKNN converter environment's Python (see ``iSpy.boot._rknn_converter_env``)
so that the full ``rknn-toolkit2`` stack and its incompatible NumPy/SciPy/
setuptools pins never touch the main runtime environment.

The worker must locate the iSpy source/package regardless of whether iSpy is
a source checkout, an editable installation, or a regular installed
deployment, and regardless of the caller's working directory.
"""

import json
import sys
import traceback
from pathlib import Path


def _ensure_ispy_importable() -> None:
    """Make sure ``import iSpy`` resolves to *this* checkout's source.

    When the worker runs under the converter environment's Python, the iSpy
    package is not installed there.  ``_convert_model_subprocess`` sets
    ``PYTHONPATH`` to the source root, but we also resolve it here as a
    fallback so the worker is robust when launched directly.
    """
    try:
        import iSpy  # noqa: F401

        return
    except ImportError:
        pass

    # <repo>/iSpy/boot/_convert_worker.py -> <repo>
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "iSpy" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        return

    # Regular installed deployment: the iSpy package lives in site-packages.
    # Walking up from this file reaches it only if the package layout is
    # preserved (it is - installed packages mirror the source layout).
    for parent in Path(__file__).resolve().parents:
        if (parent / "iSpy" / "__init__.py").exists():
            sys.path.insert(0, str(parent))
            return


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m iSpy.boot._convert_worker <args.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1]) as f:
        args = json.load(f)

    # import torch first (also repairs the logging tables rknn tools clobber)
    from iSpy.vision._safe_imports import ensure_torch_imported

    ensure_torch_imported()

    # the parent's offline state arrives as ISPY_OFFLINE=1; importing the
    # policy arms the socket guard in this process too, so a conversion that
    # tries to fetch something fails fast instead of hanging
    from iSpy.core import network_policy  # noqa: F401

    # lazy import so a bad args file fails fast w/o paying boot.py's heavy import cost
    from iSpy.vision.optimizer import convert_model

    result = convert_model(
        args["model_file"],
        args["target_format"],
        args["input_size"],
        quantize=args.get("quantize"),
        force=args.get("force", False),
        dataset_path=args.get("dataset_path"),
    )

    out_path = Path(sys.argv[1] + ".result.json")
    out_path.write_text(
        json.dumps(
            {
                "result": result,
                # Interpreter evidence: the parent can confirm the worker ran
                # under the converter environment's Python, not the runtime one.
                "python": sys.executable,
            }
        )
    )


def run() -> None:
    """Run main() and report failures as structured errors, never a silent exit."""
    try:
        _ensure_ispy_importable()
        main()
    except Exception as exc:
        # Write a structured error so the parent can surface the *real*
        # failure instead of just a non-zero exit code.  The traceback is
        # preserved so nested import errors (NumPy/SciPy/pkg_resources) are
        # visible, not misreported as a missing toolkit.
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
            f"Conversion worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
