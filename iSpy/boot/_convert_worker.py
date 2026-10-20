import json
import sys
import traceback
from pathlib import Path


def _ensure_ispy_importable() -> None:
    try:
        import iSpy  # noqa: F401

        return
    except ImportError:
        pass

    # source checkout
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "iSpy" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        return

    # installed package
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

    # torch first; rknn tools mess with the logging tables
    from iSpy.vision._safe_imports import ensure_torch_imported

    ensure_torch_imported()

    # offline guard from the parent
    from iSpy.core import network_policy  # noqa: F401

    # lazy import so bad args fail fast
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
                # parent uses this to confirm the converter env ran it
                "python": sys.executable,
            }
        )
    )


def run() -> None:
    try:
        _ensure_ispy_importable()
        main()
    except Exception as exc:
        # keep the real traceback for the parent
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
