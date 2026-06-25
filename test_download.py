"""Test script for quantization dataset image download by keyword."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iSpy.dataset.dataset import prepare_quantization_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images for quantization dataset")
    parser.add_argument("-c", "--count", type=int, default=20, help="Number of images to download (default: 20)")
    args = parser.parse_args()

    keywords = ["car"]
    print(f"Downloading {args.count} images for: {keywords}")
    ds = prepare_quantization_dataset("QuantizeDataset", keywords=keywords, count=args.count)
    img_dir = Path(ds) / "images"
    images = list(img_dir.glob("*"))
    print(f"\nDone. {len(images)} images in {img_dir}:")
    for p in sorted(images):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
