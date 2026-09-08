
import sys
import json
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python -m iSpy.boot._yoloworld_reparam_worker <args.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(sys.argv[1], encoding="utf-8") as f:
        args = json.load(f)

    weights = args["weights"]
    classes = args["classes"]
    output_path = Path(args["output_path"])

    out_path = Path(sys.argv[1] + ".result.json")
    try:
        # lazy import so a bad args file fails fast without paying the
        # (possibly absent-in-this-env) ultralytics import cost
        from ultralytics import YOLOWorld, YOLO  # optional build-time tool (AGPL)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLOWorld(weights, verbose=False)
        model.set_classes(classes)
        model.save(str(output_path))

        # reload+resave exactly as the previous in-process path did: this
        # normalizes the saved checkpoint format into a plain fixed-vocab
        # detector before the on-device loader reads it.
        try:
            model = YOLO(str(output_path), task="detect", verbose=False, weights_only=True)
        except TypeError:
            model = YOLO(str(output_path), task="detect", verbose=False)

        out_path.write_text(json.dumps({"result": str(output_path)}), encoding="utf-8")
    except Exception as exc:  # clear error on failure, never a silent 0-exit
        try:
            out_path.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        except OSError:
            pass
        print(f"YOLO World reparameterization failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
