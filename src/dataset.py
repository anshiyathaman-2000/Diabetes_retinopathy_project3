"""
PyTorch Dataset and DataLoader Pipeline for IDRiD Disease Grading.
Handles:
1. Label parsing and validation.
2. 5-class DR severity vs. Binary Referable DR mapping.
3. Stratified Train / Validation splitting.
4. Class weight computation for addressing severe class imbalance.
5. High-performance PyTorch DataLoader instantiation.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.preprocessing import (
    preprocess_fundus_image,
    get_train_transforms,
    get_eval_transforms
)

DR_CLASSES_5 = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]

DR_CLASSES_BINARY = [
    "Non-Referable DR (Grade 0-1)",
    "Referable DR (Grade 2-4)"
]

# Cryptographically verified intra-training duplicate clusters from SHA-256 image audits
VERIFIED_INTRA_TRAIN_DUPLICATE_PAIRS = [
    ("IDRiD_045", "IDRiD_333"),
    ("IDRiD_079", "IDRiD_279"),
    ("IDRiD_028", "IDRiD_327"),
    ("IDRiD_021", "IDRiD_391"),
    ("IDRiD_073", "IDRiD_375"),
]

class IDRiDDataset(Dataset):
    """
    IDRiD Retinal Fundus Dataset.
    Loads and preprocesses images with corresponding diabetic retinopathy labels.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        task: str = "five_class",
        binary_threshold: int = 2,
        image_size: int = 256,
        preprocessing_method: str = "clahe",
        crop_border: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        transform = None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.task = task
        self.binary_threshold = binary_threshold
        self.image_size = image_size
        self.preprocessing_method = preprocessing_method
        self.crop_border = crop_border
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform

        # Validate entries and map labels
        self.image_names = self.df['Image name'].astype(str).tolist()
        self.raw_grades = self.df['Retinopathy grade'].astype(int).tolist()

        if self.task == "five_class":
            self.labels = self.raw_grades
            self.num_classes = 5
        elif self.task == "binary":
            self.labels = [1 if g >= self.binary_threshold else 0 for g in self.raw_grades]
            self.num_classes = 2
        else:
            raise ValueError(f"Invalid task '{task}'. Use 'five_class' or 'binary'.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_name = self.image_names[idx]
        label = self.labels[idx]

        # Check if pre-cached image exists
        if self.cache_dir:
            cached_path = self.cache_dir / f"{img_name}.png"
            if cached_path.exists():
                pil_img = Image.open(cached_path).convert("RGB")
            else:
                fpath = self.image_dir / f"{img_name}.jpg"
                pil_img = preprocess_fundus_image(
                    image_input=str(fpath),
                    image_size=self.image_size,
                    method=self.preprocessing_method,
                    crop_border=self.crop_border
                )
        else:
            fpath = self.image_dir / f"{img_name}.jpg"
            if not fpath.exists():
                raise FileNotFoundError(f"Image file not found: {fpath}")

            pil_img = preprocess_fundus_image(
                image_input=str(fpath),
                image_size=self.image_size,
                method=self.preprocessing_method,
                crop_border=self.crop_border
            )

        # Apply torchvision transform (tensor conversion, augmentation, normalization)
        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        return tensor_img, label, img_name

def load_and_clean_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read CSV, strip whitespace, remove empty columns from trailing commas."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df.columns = df.columns.str.strip()
    df['Image name'] = df['Image name'].astype(str).str.strip()
    df['Retinopathy grade'] = df['Retinopathy grade'].astype(int)
    return df

def compute_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = "inverse",
    power: float = 1.0
) -> torch.Tensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
    - method='inverse': w_j = (N / (C * N_j))^power (standard inverse freq when power=1.0)
    - method='sqrt': w_j = (N / (C * N_j))^0.5 (smoother weighting to prevent gradient spikes on extreme minority)
    - method='none': w_j = 1.0
    """
    if method == "none":
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    effective_power = 0.5 if method == "sqrt" else power

    for c in range(num_classes):
        if counts[c] > 0:
            base_w = total_samples / (num_classes * counts[c])
            weights[c] = float(base_w ** effective_power)
        else:
            weights[c] = 1.0
    return torch.tensor(weights, dtype=torch.float32)

def get_idrid_dataloaders(
    config: Dict[str, Any],
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Prepares stratified Train, Validation, and Test DataLoaders.
    Uses official 103 test samples as holdout.
    Performs stratified split on the 413 official training samples.
    """
    train_csv_path = Path(config["data"]["train_csv"])
    test_csv_path = Path(config["data"]["test_csv"])
    train_img_dir = Path(config["data"]["train_images_dir"])
    test_img_dir = Path(config["data"]["test_images_dir"])

    df_full_train = load_and_clean_csv(train_csv_path)
    df_test = load_and_clean_csv(test_csv_path)

    task = config["task"]["type"]
    num_classes = config["task"]["num_classes"]
    binary_threshold = config["task"].get("binary_threshold", 2)
    val_ratio = config["data"].get("val_split_ratio", 0.20)
    img_size = config["preprocessing"]["image_size"]
    prep_method = config["preprocessing"].get("method", "clahe")
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 2)

    # Determine stratification target labels
    if task == "five_class":
        strat_target = df_full_train['Retinopathy grade'].values
    else:
        strat_target = (df_full_train['Retinopathy grade'] >= binary_threshold).astype(int).values

    # Split selection: strict leakage-aware manifest vs explicit standard split
    split_type = config.get("data", {}).get("split_type", "leakage_aware").lower()
    manifest_path_cfg = config.get("data", {}).get("split_manifest", "data/splits/leakage_aware_split_manifest.json")
    manifest_path = Path(manifest_path_cfg)

    if split_type == "leakage_aware":
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"[SPLIT ERROR] Configured leakage-aware manifest not found: '{manifest_path}'. "
                f"Silent fallback to random split is strictly disabled to prevent train-val leakage."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "train" not in manifest or "val" not in manifest:
            raise ValueError(f"[SPLIT ERROR] Manifest '{manifest_path}' must contain 'train' and 'val' lists.")

        train_names = manifest["train"]
        val_names = manifest["val"]
        train_names_set = set(train_names)
        val_names_set = set(val_names)

        # 1. Intra-list duplicate validation
        if len(train_names) != len(train_names_set):
            raise ValueError(f"[SPLIT ERROR] Duplicate sample IDs detected inside manifest 'train' list.")
        if len(val_names) != len(val_names_set):
            raise ValueError(f"[SPLIT ERROR] Duplicate sample IDs detected inside manifest 'val' list.")

        # 2. Strict disjointness check
        overlap = train_names_set & val_names_set
        if overlap:
            raise ValueError(f"[SPLIT ERROR] Critical data leakage! Train and Val partitions overlap by {len(overlap)} samples: {overlap}")

        # 3. Training CSV alignment check
        full_train_ids = set(df_full_train["Image name"])
        manifest_all_ids = train_names_set | val_names_set
        unknown_ids = manifest_all_ids - full_train_ids
        if unknown_ids:
            raise ValueError(f"[SPLIT ERROR] Manifest contains IDs absent from training CSV: {unknown_ids}")

        # 4. Duplicate cluster cohesion validation (ensure byte-identical images never cross train/val)
        for img1, img2 in VERIFIED_INTRA_TRAIN_DUPLICATE_PAIRS:
            if img1 in manifest_all_ids and img2 in manifest_all_ids:
                in_train_1 = img1 in train_names_set
                in_train_2 = img2 in train_names_set
                in_val_1 = img1 in val_names_set
                in_val_2 = img2 in val_names_set
                if (in_train_1 and in_val_2) or (in_val_1 and in_train_2):
                    raise ValueError(
                        f"[SPLIT ERROR] Critical train-val leakage! Cryptographic duplicate pair "
                        f"({img1}, {img2}) split across train and val partitions in '{manifest_path}'."
                    )

        df_train = df_full_train[df_full_train["Image name"].isin(train_names_set)].reset_index(drop=True)
        df_val = df_full_train[df_full_train["Image name"].isin(val_names_set)].reset_index(drop=True)

        if len(df_train) != len(train_names_set):
            raise ValueError(f"[SPLIT ERROR] Mismatch slicing train set: expected {len(train_names_set)}, got {len(df_train)}.")
        if len(df_val) != len(val_names_set):
            raise ValueError(f"[SPLIT ERROR] Mismatch slicing val set: expected {len(val_names_set)}, got {len(df_val)}.")

        print(f"\n[Split Verification] Successfully loaded leakage-aware manifest: '{manifest_path}'")
        print(f"  • Training samples  : {len(df_train)}")
        print(f"  • Validation samples: {len(df_val)}")
        print(f"  • Partition overlap : 0 (Strict Disjointness PASS)")
        print(f"  • Duplicate cohesion: PASS (0 duplicate pairs cross train/val)")
    elif split_type == "standard":
        print(f"\n[WARNING] Using unconstrained StratifiedShuffleSplit. Duplicate pairs may cross train/val!")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(df_full_train, strat_target))
        df_train = df_full_train.iloc[train_idx].reset_index(drop=True)
        df_val = df_full_train.iloc[val_idx].reset_index(drop=True)
    else:
        raise ValueError(f"[SPLIT ERROR] Unknown split_type '{split_type}'. Choose 'leakage_aware' or 'standard'.")

    # Optional test protocol: official (N=103) vs deduplicated (N=102, excluding IDRiD_064)
    test_protocol = config.get("data", {}).get("test_protocol", "official").lower()
    if test_protocol in ["deduplicated", "clean"]:
        df_test = df_test[df_test['Image name'] != 'IDRiD_064'].reset_index(drop=True)

    # Compute class weights on training subset only (prevents data leakage)
    if task == "five_class":
        train_labels = df_train['Retinopathy grade'].tolist()
    else:
        train_labels = (df_train['Retinopathy grade'] >= binary_threshold).astype(int).tolist()

    weight_method = config.get("training", {}).get("class_weight_method", "inverse")
    weight_power = float(config.get("training", {}).get("class_weight_power", 1.0))
    class_weights = compute_class_weights(
        train_labels,
        num_classes,
        method=weight_method,
        power=weight_power
    )

    # Transforms
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

    # Check for preprocessed cache
    cache_base = Path(config.get("data", {}).get("cache_dir", "data/cache")) / f"cache_{img_size}_{prep_method}"
    train_cache = cache_base / "train" if (cache_base / "train").exists() else None
    test_cache = cache_base / "test" if (cache_base / "test").exists() else None

    # Datasets
    train_dataset = IDRiDDataset(
        dataframe=df_train,
        image_dir=train_img_dir,
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
        image_dir=train_img_dir,
        task=task,
        binary_threshold=binary_threshold,
        image_size=img_size,
        preprocessing_method=prep_method,
        crop_border=True,
        cache_dir=train_cache,
        transform=eval_transform
    )

    test_dataset = IDRiDDataset(
        dataframe=df_test,
        image_dir=test_img_dir,
        task=task,
        binary_threshold=binary_threshold,
        image_size=img_size,
        preprocessing_method=prep_method,
        crop_border=True,
        cache_dir=test_cache,
        transform=eval_transform
    )

    # DataLoaders
    # Note: On macOS, num_workers=0 or 2 with persistent_workers=False avoids multiprocessing overhead
    sampler_type = config.get("training", {}).get("sampler_type", "none").lower()
    if sampler_type == "weighted":
        from torch.utils.data import WeightedRandomSampler
        counts = np.bincount(train_labels, minlength=num_classes)
        sampler_power = float(config.get("training", {}).get("sampler_power", 1.0))
        # Per-class sampling weight inversely proportional to class frequency
        class_sample_weights = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            if counts[c] > 0:
                class_sample_weights[c] = (1.0 / counts[c]) ** sampler_power
            else:
                class_sample_weights[c] = 0.0
        
        sample_weights = [class_sample_weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_weights
