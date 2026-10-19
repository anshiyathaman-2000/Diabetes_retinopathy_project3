"""
Stratified and Leakage-Aware Cross-Validation Module for IDRiD Dataset (Step 5).
Provides:
1. 5-Fold Stratified Group K-Fold splitting on the official training set (N=413).
2. Guarantees that duplicate image pairs are never split across folds.
3. Completely excludes the official held-out test set (N=103).
4. DataLoader generation with fold-specific training augmentation, deterministic validation,
   and class weight computation isolated strictly to the training folds.
5. Out-of-Fold (OOF) cross-validation evaluation and metric aggregation.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    IDRiDDataset,
    DR_CLASSES_5,
    DR_CLASSES_BINARY
)
from src.preprocessing import (
    get_train_transforms,
    get_eval_transforms
)
from src.utils import (
    set_seed,
    get_device,
    calculate_metrics
)

TRAIN_CSV = Path("B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
TEST_CSV = Path("B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
TRAIN_IMG_DIR = Path("B. Disease Grading/1. Original Images/a. Training Set")
TEST_IMG_DIR = Path("B. Disease Grading/1. Original Images/b. Testing Set")
MANIFEST_JSON = Path("data/splits/stratified_kfold_manifest.json")
MANIFEST_CSV = Path("data/splits/stratified_kfold_manifest.csv")

# Verified intra-training duplicate clusters from cryptographic hash audits
INTRA_TRAIN_DUPLICATE_PAIRS = [
    ("IDRiD_375", "IDRiD_073"),  # Cluster 2 (both Grade 2)
    ("IDRiD_333", "IDRiD_045"),  # Cluster 3 (both Grade 2)
    ("IDRiD_028", "IDRiD_327"),  # Cluster 4 (Grade 4 & Grade 3)
    ("IDRiD_079", "IDRiD_279"),  # Cluster 5 (both Grade 1)
    ("IDRiD_021", "IDRiD_391"),  # Cluster 6 (Grade 1 & Grade 2)
]

def generate_stratified_kfold_splits(
    num_folds: int = 5,
    seed: int = 42,
    save_manifest: bool = True
) -> Dict[str, Any]:
    """
    Generates 5-fold stratified, leakage-aware splits on the IDRiD training set.
    Ensures duplicate images stay together in the same fold and test images are untouched.
    """
    df_train = load_and_clean_csv(TRAIN_CSV)
    df_test = load_and_clean_csv(TEST_CSV)

    # Note: IDRiD challenge labels both sets starting at IDRiD_001, but they reside in distinct folders:
    # Train: 'B. Disease Grading/1. Original Images/a. Training Set' (N=413)
    # Test : 'B. Disease Grading/1. Original Images/b. Testing Set'  (N=103)
    # The cross-validation splits are constructed exclusively from df_train.
    assert len(df_train) == 413, f"Expected 413 training samples, got {len(df_train)}"
    assert len(df_test) == 103, f"Expected 103 testing samples, got {len(df_test)}"

    # Build group mappings to prevent duplicate leakage across folds
    image_to_group = {}
    group_counter = 0

    for img1, img2 in INTRA_TRAIN_DUPLICATE_PAIRS:
        image_to_group[img1] = group_counter
        image_to_group[img2] = group_counter
        group_counter += 1

    groups = []
    for img_name in df_train["Image name"]:
        if img_name in image_to_group:
            groups.append(image_to_group[img_name])
        else:
            image_to_group[img_name] = group_counter
            groups.append(group_counter)
            group_counter += 1

    df_train["group_id"] = groups
    y_strat = df_train["Retinopathy grade"].values

    # Use StratifiedGroupKFold to honor both class stratification and group cohesion
    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(df_train), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df_train, y_strat, groups=df_train["group_id"])):
        fold_assignments[val_idx] = fold_idx

    df_train["fold"] = fold_assignments

    # Verification: check duplicate pairs never cross folds
    pair_violations = []
    for img1, img2 in INTRA_TRAIN_DUPLICATE_PAIRS:
        f1 = df_train.loc[df_train["Image name"] == img1, "fold"].values[0]
        f2 = df_train.loc[df_train["Image name"] == img2, "fold"].values[0]
        if f1 != f2:
            pair_violations.append((img1, img2, f1, f2))

    assert len(pair_violations) == 0, f"Duplicate leakage detected across folds: {pair_violations}"

    # Compile fold distribution statistics
    fold_stats = {}
    folds_dict = {f"fold_{i}": [] for i in range(num_folds)}

    for f_idx in range(num_folds):
        df_f = df_train[df_train["fold"] == f_idx]
        folds_dict[f"fold_{f_idx}"] = df_f["Image name"].tolist()
        class_counts = df_f["Retinopathy grade"].value_counts().sort_index().to_dict()
        fold_stats[f"fold_{f_idx}"] = {
            "total_samples": len(df_f),
            "class_distribution": {f"Grade_{k}": int(v) for k, v in class_counts.items()}
        }

    manifest = {
        "metadata": {
            "num_folds": num_folds,
            "random_seed": seed,
            "total_train_samples": len(df_train),
            "held_out_test_samples": len(df_test),
            "stratification_method": "StratifiedGroupKFold",
            "leakage_prevention": "Intra-training duplicate pairs assigned to identical folds",
            "test_set_isolation": "100% held-out; 0 test samples in cross-validation",
            "duplicate_pairs_verified": INTRA_TRAIN_DUPLICATE_PAIRS
        },
        "fold_statistics": fold_stats,
        "folds": folds_dict
    }

    if save_manifest:
        MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_JSON, "w") as f:
            json.dump(manifest, f, indent=2)

        df_train[["Image name", "Retinopathy grade", "fold", "group_id"]].to_csv(MANIFEST_CSV, index=False)
        print(f"Saved Stratified K-Fold manifest to:")
        print(f"  JSON: {MANIFEST_JSON}")
        print(f"  CSV : {MANIFEST_CSV}")

    return manifest

def get_kfold_dataloaders(
    config: Dict[str, Any],
    val_fold: int,
    manifest_path: Optional[Union[str, Path]] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Returns (train_loader, val_loader, class_weights) for a designated cross-validation fold.
    - val_fold is the validation fold.
    - remaining folds are merged for training.
    - Class weights are computed strictly from the training folds.
    """
    if manifest_path is None:
        manifest_path = MANIFEST_JSON

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        generate_stratified_kfold_splits(seed=seed)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    num_folds = manifest["metadata"]["num_folds"]
    if val_fold < 0 or val_fold >= num_folds:
        raise ValueError(f"val_fold must be between 0 and {num_folds-1}, got {val_fold}")

    df_full_train = load_and_clean_csv(TRAIN_CSV)

    val_images = set(manifest["folds"][f"fold_{val_fold}"])
    train_images = set()
    for f in range(num_folds):
        if f != val_fold:
            train_images.update(manifest["folds"][f"fold_{f}"])

    df_train = df_full_train[df_full_train["Image name"].isin(train_images)].reset_index(drop=True)
    df_val = df_full_train[df_full_train["Image name"].isin(val_images)].reset_index(drop=True)

    task = config["task"]["type"]
    num_classes = config["task"]["num_classes"]
    binary_threshold = config["task"].get("binary_threshold", 2)
    img_size = config["preprocessing"]["image_size"]
    prep_method = config["preprocessing"].get("method", "clahe")
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 2)

    # Class weights computed strictly from the fold's training images
    if task == "five_class":
        train_labels = df_train["Retinopathy grade"].tolist()
    else:
        train_labels = (df_train["Retinopathy grade"] >= binary_threshold).astype(int).tolist()

    weight_method = config.get("training", {}).get("class_weight_method", "inverse")
    weight_power = float(config.get("training", {}).get("class_weight_power", 1.0))
    class_weights = compute_class_weights(
        train_labels,
        num_classes,
        method=weight_method,
        power=weight_power
    )

    # Transforms: training uses augmentation, validation uses deterministic eval
    norm_mean = config["preprocessing"]["normalization"]["mean"]
    norm_std = config["preprocessing"]["normalization"]["std"]
    aug_cfg = config["preprocessing"].get("augmentation", {})

    train_transform = get_train_transforms(
        mean=tuple(norm_mean),
        std=tuple(norm_std),
        horizontal_flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
        vertical_flip_prob=aug_cfg.get("vertical_flip_prob", 0.5),
        rotation_degrees=aug_cfg.get("rotation_degrees", 15.0),
        brightness_jitter=aug_cfg.get("brightness_jitter", 0.1),
        contrast_jitter=aug_cfg.get("contrast_jitter", 0.1)
    )
    eval_transform = get_eval_transforms(mean=tuple(norm_mean), std=tuple(norm_std))

    # Preprocessed cache support
    cache_base = Path(config.get("data", {}).get("cache_dir", "data/cache")) / f"cache_{img_size}_{prep_method}"
    train_cache = cache_base / "train" if (cache_base / "train").exists() else None

    train_dataset = IDRiDDataset(
        dataframe=df_train,
        image_dir=TRAIN_IMG_DIR,
        task=task,
        binary_threshold=binary_threshold,
        image_size=img_size,
        preprocessing_method=prep_method,
        crop_border=True,
        cache_dir=train_cache,
        transform=train_transform
    )

    val_dataset = IDRiDDataset(
        dataframe=df_val,
        image_dir=TRAIN_IMG_DIR,
        task=task,
        binary_threshold=binary_threshold,
        image_size=img_size,
        preprocessing_method=prep_method,
        crop_border=True,
        cache_dir=train_cache,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    return train_loader, val_loader, class_weights

if __name__ == "__main__":
    manifest = generate_stratified_kfold_splits(num_folds=5, seed=42)
    print("\nStratified K-Fold Split Summary:")
    for fold_name, stats in manifest["fold_statistics"].items():
        print(f"  {fold_name}: {stats['total_samples']} samples | {stats['class_distribution']}")
