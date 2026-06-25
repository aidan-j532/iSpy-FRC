"""Test script for quantization dataset image download by keyword."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iSpy.dataset.dataset import prepare_quantization_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images for quantization dataset")
    parser.add_argument("-c", "--count", type=int, default=20, help="Number of images to download (default: 20)")
    parser.add_argument("-p", "--proxy", type=str, default=None, help="Proxy URL (e.g. http://user:pass@127.0.0.1:8080)")
    args = parser.parse_args()

    keywords = ["sexy women in bikini"]
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None
    print(f"Downloading {args.count} images for: {keywords}")
    if proxies:
        print(f"Using proxy: {args.proxy}")
    ds = prepare_quantization_dataset("QuantizeDataset", keywords=keywords, count=args.count, proxies=proxies)
    img_dir = Path(ds) / "images"
    images = list(img_dir.glob("*"))
    print(f"\nDone. {len(images)} images in {img_dir}:")
    for p in sorted(images):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
