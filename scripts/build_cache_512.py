"""
Generate 512x512 CLAHE Preprocessed Image Cache for IDRiD.
Preprocesses:
- All 413 training images -> data/cache/cache_512_clahe/train
- All 103 testing images  -> data/cache/cache_512_clahe/test
Matches the exact preprocessing pipeline used for 256x256 cache.
"""

import sys
from pathlib import Path
import time
from PIL import Image
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_fundus_image
from src.dataset import load_and_clean_csv

def build_cache_512():
    train_csv = ROOT / "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = ROOT / "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    train_img_dir = ROOT / "B. Disease Grading/1. Original Images/a. Training Set"
    test_img_dir = ROOT / "B. Disease Grading/1. Original Images/b. Testing Set"

    out_train = ROOT / "data/cache/cache_512_clahe/train"
    out_test = ROOT / "data/cache/cache_512_clahe/test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    df_train = load_and_clean_csv(train_csv)
    df_test = load_and_clean_csv(test_csv)

    print(f"Building 512x512 CLAHE Cache...")
    print(f"  Training targets: {len(df_train)} images -> {out_train}")
    print(f"  Testing targets : {len(df_test)} images -> {out_test}")

    # Process training set
    t0 = time.time()
    train_count = 0
    for name in df_train["Image name"]:
        dest = out_train / f"{name}.png"
        if not dest.exists():
            src_path = train_img_dir / f"{name}.jpg"
            img = preprocess_fundus_image(str(src_path), image_size=512, method="clahe", crop_border=True)
            img.save(dest, format="PNG")
            train_count += 1
    t_train = time.time() - t0
    print(f"  Training cache ready: {len(list(out_train.glob('*.png')))} images (newly generated: {train_count}) in {t_train:.1f}s")

    # Process test set
    t0 = time.time()
    test_count = 0
    for name in df_test["Image name"]:
        dest = out_test / f"{name}.png"
        if not dest.exists():
            src_path = test_img_dir / f"{name}.jpg"
            img = preprocess_fundus_image(str(src_path), image_size=512, method="clahe", crop_border=True)
            img.save(dest, format="PNG")
            test_count += 1
    t_test = time.time() - t0
    print(f"  Testing cache ready : {len(list(out_test.glob('*.png')))} images (newly generated: {test_count}) in {t_test:.1f}s")
    print(f"512x512 CLAHE Cache Build Complete!\n")

if __name__ == "__main__":
    build_cache_512()
