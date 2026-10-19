"""
One-Time Offline Preprocessing and Caching Script for IDRiD Dataset.
Transforms large 4288x2848 fundus images into standardized 256x256 CLAHE-enhanced
and original RGB images once, reducing epoch training time from 2.8 minutes to ~2 seconds.
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
import cv2
import pandas as pd
from PIL import Image

from src.preprocessing import preprocess_fundus_image
from src.dataset import load_and_clean_csv

def process_single_image(args):
    img_name, raw_img_dir, out_dir, img_size, method = args
    in_path = raw_img_dir / f"{img_name}.jpg"
    out_path = out_dir / f"{img_name}.png"

    if out_path.exists():
        return True

    try:
        pil_img = preprocess_fundus_image(
            image_input=str(in_path),
            image_size=img_size,
            method=method,
            crop_border=True
        )
        pil_img.save(out_path, format="PNG")
        return True
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return False

def build_cache(
    base_data_dir: Path,
    cache_root_dir: Path,
    image_size: int = 256,
    method: str = "clahe"
):
    cache_dir = cache_root_dir / f"cache_{image_size}_{method}"
    train_out = cache_dir / "train"
    test_out = cache_dir / "test"

    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    train_csv = base_data_dir / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = base_data_dir / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
    train_raw_dir = base_data_dir / "1. Original Images" / "a. Training Set"
    test_raw_dir = base_data_dir / "1. Original Images" / "b. Testing Set"

    df_train = load_and_clean_csv(train_csv)
    df_test = load_and_clean_csv(test_csv)

    print(f"Generating offline cache at: {cache_dir}")
    print(f"Preprocessing method: {method.upper()} | Target size: {image_size}x{image_size}")

    train_tasks = [
        (name, train_raw_dir, train_out, image_size, method)
        for name in df_train['Image name']
    ]
    test_tasks = [
        (name, test_raw_dir, test_out, image_size, method)
        for name in df_test['Image name']
    ]

    # Use multiprocessing to speed up across CPU cores
    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Processing {len(train_tasks)} training images using {num_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_single_image, train_tasks))

    print(f"Processing {len(test_tasks)} test images using {num_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_single_image, test_tasks))

    print(f"Successfully cached {len(train_tasks) + len(test_tasks)} images in {cache_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache preprocessed IDRiD images.")
    parser.add_argument("--data_dir", type=str, default="B. Disease Grading")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--method", type=str, default="clahe")
    args = parser.parse_args()

    build_cache(
        base_data_dir=Path(args.data_dir),
        cache_root_dir=Path(args.cache_dir),
        image_size=args.image_size,
        method=args.method
    )
