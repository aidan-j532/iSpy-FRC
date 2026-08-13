import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m iSpy.boot._convert_worker <args.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1]) as f:
        args = json.load(f)

    # lazy import so a bad args file fails fast w/o paying boot.py's heavy import cost
    from iSpy.vision.optimizer import convert_model

    result = convert_model(
        args["model_file"],
        args["target_format"],
        args["input_size"],
        quantize=args.get("quantize"),
        force=args.get("force", False),
        kw=args.get("kw"),
        dataset_path=args.get("dataset_path"),
    )

    out_path = Path(sys.argv[1] + ".result.json")
    out_path.write_text(json.dumps({"result": result}))

if __name__ == "__main__":
    main()