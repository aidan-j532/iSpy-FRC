"""Test script for quantization dataset image download by keyword."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iSpy.dataset.dataset import prepare_quantization_dataset, validate_quantization_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images for quantization dataset")
    parser.add_argument("-c", "--count", type=int, default=20, help="Number of images to download (default: 20)")
    parser.add_argument("-k", "--keywords", type=str, nargs="+", default=["car"], help="Search keywords (default: car)")
    parser.add_argument("-p", "--path", type=str, default="QuantizeDataset", help="Dataset path (default: QuantizeDataset)")
    parser.add_argument("--validate", action="store_true", help="Only validate existing dataset, don't download")
    args = parser.parse_args()

    if args.validate:
        result = validate_quantization_dataset(args.path)
        print(f"\nValidation {'PASSED' if result['valid'] else 'FAILED'} for {result['dataset_path']}")
        print(f"  Images: {result['image_count']}")
        print(f"  RKNN ready: {result['rknn_ready']}")
        print(f"  Ultralytics ready: {result['ultralytics_ready']}")
        if result["issues"]:
            print("  Issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")
    else:
        keywords = args.keywords
        print(f"Downloading {args.count} images for: {keywords}")
        print(f"Output path: {args.path}")
        ds = prepare_quantization_dataset(args.path, keywords=keywords, count=args.count)
        img_dir = Path(ds) / "images"
        images = sorted(img_dir.glob("*"))
        print(f"\nDone. {len(images)} images in {img_dir}:")
        for p in images:
            print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

        print("\nValidating dataset...")
        result = validate_quantization_dataset(args.path)
        print(f"  Valid: {result['valid']}")
        print(f"  RKNN ready: {result['rknn_ready']}")
        if result["issues"]:
            print("  Issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")
