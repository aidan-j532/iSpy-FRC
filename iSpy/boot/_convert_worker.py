import sys
import json


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m iSpy.boot._convert_worker <args.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1]) as f:
        args = json.load(f)

    # Imported here, not at module level, so a bad args file fails fast
    # without paying boot.py's heavy import cost first.
    from iSpy.boot.boot import convert_model

    result = convert_model(
        args["model_file"],
        args["target_format"],
        args["input_size"],
        quantize=args.get("quantize"),
        force=args.get("force", False),
        kw=args.get("kw"),
    )

    # print(f"ISPY_RESULT:{result
if __name__ == "__main__":
    main()