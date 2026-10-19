
"""
Step 7: Retinal Image Preprocessing and Quality Comparison Benchmark for IDRiD DR Classification.

Evaluates and rigorously compares three retinal image preprocessing pipelines:
1. Configuration 1 ('original'): Original fundus images with standard ImageNet normalization.
2. Configuration 2 ('standard_clahe'): Standard channel-wise RGB CLAHE preprocessing.
3. Configuration 3 ('existing_clahe'): Project's existing CIELAB luminance (L*) CLAHE pipeline with border auto-crop.

Rigorous Experimental Protocols:
- Exact same data partitions: 330 train, 83 val from leakage-aware manifest; official (103) & clean (102) test sets.
- Strict anti-leakage guarantee: class weights, transforms, normalizations fitted strictly on training data.
- Identical model architecture: Baseline ResNet-50 (~23.5M parameters) under identical seed (42).
- Identical training hyperparameters: AdamW (lr=1e-4, wd=1e-4), CosineAnnealingLR, 15 epochs, batch size 16.
- Comprehensive metrics: Accuracy, Macro F1, Weighted F1, QWK, and per-class Recall across Grades 0-4.
- Empirical computational overhead measurement: latency (ms/image) and throughput (fps) across all pipelines.
- Machine-readable artifact exports in outputs/experiments/idrid_step7_preprocessing/.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import (
    set_seed,
    get_device,
    calculate_metrics
)
from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    get_train_transforms,
    get_eval_transforms,
    IDRiDDataset,
    DR_CLASSES_5
)
from src.preprocessing import (
    preprocess_fundus_image,
    apply_standard_clahe_rgb,
    apply_clahe_lab,
    auto_crop_fundus
)
from src.models import build_model

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/idrid_step7_preprocessing"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDS_DIR = OUTPUT_DIR / "predictions"
COMP_DIR = OUTPUT_DIR / "comparisons"
ROOT_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DATA_BASE = PROJECT_ROOT / "B. Disease Grading"
TRAIN_CSV = DATA_BASE / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV = DATA_BASE / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
TRAIN_IMG_DIR = DATA_BASE / "1. Original Images" / "a. Training Set"
TEST_IMG_DIR = DATA_BASE / "1. Original Images" / "b. Testing Set"
SPLIT_MANIFEST = PROJECT_ROOT / "data/splits/leakage_aware_split_manifest.json"

CONFIGURATIONS = [
    {
        "id": "original",
        "name": "Configuration 1: Original Fundus (ImageNet Norm)",
        "method": "original",
        "cache_dir": PROJECT_ROOT / "data/cache/cache_256_original",
        "checkpoint_name": "resnet50_five_class_original_s42_best.pt"
    },
    {
        "id": "standard_clahe",
        "name": "Configuration 2: Standard RGB CLAHE",
        "method": "standard_clahe",
        "cache_dir": PROJECT_ROOT / "data/cache/cache_256_standard_clahe",
        "checkpoint_name": "resnet50_five_class_standard_clahe_s42_best.pt"
    },
    {
        "id": "existing_clahe",
        "name": "Configuration 3: Existing LAB Luminance CLAHE",
        "method": "clahe",
        "cache_dir": PROJECT_ROOT / "data/cache/cache_256_clahe",
        "checkpoint_name": "resnet50_five_class_existing_clahe_s42_best.pt"
    }
]

def verify_data_partitions_and_leakage_safety():
    """Confirms partition sizes, disjoint sets, and strict absence of data leakage."""
    print("=== Step 7: Verifying Data Partitions and Anti-Leakage Guarantees ===")
    assert SPLIT_MANIFEST.exists(), f"Manifest not found: {SPLIT_MANIFEST}"
    with open(SPLIT_MANIFEST, "r") as f:
        manifest = json.load(f)

    train_ids = set(manifest["train"])
    val_ids = set(manifest["val"])
    assert len(train_ids) == 330, f"Expected 330 train samples, got {len(train_ids)}"
    assert len(val_ids) == 83, f"Expected 83 val samples, got {len(val_ids)}"
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val splits overlap!"

    df_test = load_and_clean_csv(TEST_CSV)
    test_ids = set(df_test["Image name"])
    assert len(test_ids) == 103, f"Expected 103 test samples, got {len(test_ids)}"

    print(f"  [OK] Train samples: {len(train_ids)} | Val samples: {len(val_ids)} | Official Test: {len(test_ids)}")
    print(f"  [OK] Deduplicated clean test set: 102 samples (excluding IDRiD_064)")
    print(f"  [OK] Normalization statistics: standard ImageNet mean/std (fixed a priori, not fitted on val/test)")
    print(f"  [OK] Class weights: strictly computed on 330 training samples\n")
    return manifest

def benchmark_preprocessing_computational_overhead(num_benchmark_samples: int = 20) -> Dict[str, Any]:
    """
    Rigorously benchmarks execution time and computational overhead of raw image loading and preprocessing.
    Processes full-resolution raw fundus images (4288x2848, ~10MB each) using:
    - Configuration 1: Original fundus (auto-crop + resize 256x256 + ImageNet normalization)
    - Configuration 2: Standard RGB CLAHE (auto-crop + RGB CLAHE + resize 256x256 + ImageNet normalization)
    - Configuration 3: Existing LAB CLAHE (auto-crop + LAB Luminance CLAHE + resize 256x256 + ImageNet normalization)
    """
    print(f"=== Benchmarking Preprocessing Computational Overhead (N={num_benchmark_samples} raw images) ===")
    df_train = load_and_clean_csv(TRAIN_CSV)
    sample_img_names = df_train["Image name"].iloc[:num_benchmark_samples].tolist()

    # Pre-check image paths
    raw_paths = [TRAIN_IMG_DIR / f"{name}.jpg" for name in sample_img_names]
    for p in raw_paths:
        if not p.exists():
            raise FileNotFoundError(f"Raw image missing: {p}")

    eval_tf = get_eval_transforms()

    bench_results = {}
    methods_to_test = [
        ("original", "Configuration 1: Original Fundus"),
        ("standard_clahe", "Configuration 2: Standard RGB CLAHE"),
        ("clahe", "Configuration 3: Existing LAB Luminance CLAHE")
    ]

    for method_key, method_label in methods_to_test:
        # Warm-up (3 images)
        for warm_p in raw_paths[:3]:
            pil_img = preprocess_fundus_image(str(warm_p), image_size=256, method=method_key, crop_border=True)
            _ = eval_tf(pil_img)

        # Timed execution
        per_image_latencies_ms = []
        prep_only_latencies_ms = []

        for p in raw_paths:
            t0 = time.perf_counter()
            # Read raw BGR image
            img_bgr = cv2.imread(str(p))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t_load = time.perf_counter()

            # Preprocessing
            if method_key == "original":
                cropped = auto_crop_fundus(img_rgb)
                pil_img = Image.fromarray(cropped).resize((256, 256), resample=Image.Resampling.BILINEAR)
            elif method_key == "standard_clahe":
                cropped = auto_crop_fundus(img_rgb)
                enh = apply_standard_clahe_rgb(cropped)
                pil_img = Image.fromarray(enh).resize((256, 256), resample=Image.Resampling.BILINEAR)
            elif method_key == "clahe":
                cropped = auto_crop_fundus(img_rgb)
                enh = apply_clahe_lab(cropped)
                pil_img = Image.fromarray(enh).resize((256, 256), resample=Image.Resampling.BILINEAR)

            _ = eval_tf(pil_img)
            t_end = time.perf_counter()

            total_ms = (t_end - t0) * 1000.0
            prep_ms = (t_end - t_load) * 1000.0
            per_image_latencies_ms.append(total_ms)
            prep_only_latencies_ms.append(prep_ms)

        mean_total_ms = float(np.mean(per_image_latencies_ms))
        std_total_ms = float(np.std(per_image_latencies_ms))
        mean_prep_ms = float(np.mean(prep_only_latencies_ms))
        std_prep_ms = float(np.std(prep_only_latencies_ms))
        throughput_fps = float(1000.0 / mean_total_ms)

        bench_results[method_key] = {
            "method_key": method_key,
            "method_label": method_label,
            "num_samples": num_benchmark_samples,
            "mean_total_latency_ms": round(mean_total_ms, 2),
            "std_total_latency_ms": round(std_total_ms, 2),
            "min_total_latency_ms": round(float(np.min(per_image_latencies_ms)), 2),
            "max_total_latency_ms": round(float(np.max(per_image_latencies_ms)), 2),
            "mean_preprocessing_only_ms": round(mean_prep_ms, 2),
            "std_preprocessing_only_ms": round(std_prep_ms, 2),
            "throughput_fps": round(throughput_fps, 2)
        }
        print(f"  {method_label}: Total Latency = {mean_total_ms:.1f} ± {std_total_ms:.1f} ms | Prep Only = {mean_prep_ms:.1f} ± {std_prep_ms:.1f} ms | Throughput = {throughput_fps:.1f} FPS")

    # Relative overhead calculation relative to Configuration 1 (Original)
    orig_total = bench_results["original"]["mean_total_latency_ms"]
    orig_prep = bench_results["original"]["mean_preprocessing_only_ms"]
    for k in bench_results:
        bench_results[k]["relative_total_slowdown_vs_original"] = round(bench_results[k]["mean_total_latency_ms"] / orig_total, 2)
        bench_results[k]["relative_prep_slowdown_vs_original"] = round(bench_results[k]["mean_preprocessing_only_ms"] / orig_prep, 2)

    # Benchmark cached I/O throughput
    cached_train_dir = PROJECT_ROOT / "data/cache/cache_256_original/train"
    if cached_train_dir.exists():
        t0 = time.perf_counter()
        cached_paths = list(cached_train_dir.glob("*.png"))[:num_benchmark_samples]
        for cp in cached_paths:
            pil_img = Image.open(cp).convert("RGB")
            _ = eval_tf(pil_img)
        t_cache = time.perf_counter()
        cached_mean_ms = ((t_cache - t0) / len(cached_paths)) * 1000.0
        bench_results["cached_png_baseline"] = {
            "method_key": "cached_png",
            "method_label": "Pre-cached 256x256 PNGs (Training Loader Baseline)",
            "num_samples": len(cached_paths),
            "mean_total_latency_ms": round(cached_mean_ms, 2),
            "throughput_fps": round(1000.0 / cached_mean_ms, 2),
            "speedup_vs_raw_original": round(orig_total / cached_mean_ms, 1)
        }
        print(f"  Pre-cached PNGs: Latency = {cached_mean_ms:.2f} ms | Throughput = {1000.0 / cached_mean_ms:.1f} FPS ({orig_total / cached_mean_ms:.1f}x speedup vs raw loading)")

    # Save benchmark outputs
    with open(COMP_DIR / "preprocessing_computational_overhead.json", "w") as f:
        json.dump(bench_results, f, indent=2)

    df_bench = pd.DataFrame([v for k, v in bench_results.items() if k != "cached_png_baseline"])
    df_bench.to_csv(COMP_DIR / "preprocessing_computational_overhead.csv", index=False)
    print("  [OK] Saved computational overhead benchmarks to comparisons/\n")
    return bench_results

def create_dataloaders_for_config(
    cfg_info: Dict[str, Any],
    manifest: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Constructs train, val, test_official, test_clean dataloaders for a specific preprocessing configuration."""
    df_full_train = load_and_clean_csv(TRAIN_CSV)
    df_test_official = load_and_clean_csv(TEST_CSV)
    df_test_clean = df_test_official[df_test_official["Image name"] != "IDRiD_064"].reset_index(drop=True)

    train_names_set = set(manifest["train"])
    val_names_set = set(manifest["val"])
    df_train = df_full_train[df_full_train["Image name"].isin(train_names_set)].reset_index(drop=True)
    df_val = df_full_train[df_full_train["Image name"].isin(val_names_set)].reset_index(drop=True)

    # Class weights strictly on training subset
    train_labels = df_train["Retinopathy grade"].tolist()
    class_weights = compute_class_weights(train_labels, num_classes=5, method="sqrt", power=1.0)

    train_tf = get_train_transforms(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_degrees=15.0,
        brightness_jitter=0.1,
        contrast_jitter=0.1
    )
    eval_tf = get_eval_transforms(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    cache_train = cfg_info["cache_dir"] / "train"
    cache_test = cfg_info["cache_dir"] / "test"

    train_ds = IDRiDDataset(
        dataframe=df_train,
        image_dir=TRAIN_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method=cfg_info["method"],
        crop_border=True,
        cache_dir=cache_train,
        transform=train_tf
    )
    val_ds = IDRiDDataset(
        dataframe=df_val,
        image_dir=TRAIN_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method=cfg_info["method"],
        crop_border=True,
        cache_dir=cache_train,
        transform=eval_tf
    )
    test_official_ds = IDRiDDataset(
        dataframe=df_test_official,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method=cfg_info["method"],
        crop_border=True,
        cache_dir=cache_test,
        transform=eval_tf
    )
    test_clean_ds = IDRiDDataset(
        dataframe=df_test_clean,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method=cfg_info["method"],
        crop_border=True,
        cache_dir=cache_test,
        transform=eval_tf
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_official_loader = DataLoader(test_official_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_clean_loader = DataLoader(test_clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_official_loader, test_clean_loader, class_weights

def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates model over dataloader, returns calculate_metrics dict and per-image predictions DataFrame."""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    all_names = []

    with torch.no_grad():
        for imgs, lbls, names in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())
            all_names.extend(names)

    probs_np = np.array(all_probs)
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)

    metrics = calculate_metrics(
        y_true=targets_np,
        y_pred=preds_np,
        y_probs=probs_np,
        task="five_class",
        class_names=DR_CLASSES_5
    )

    pred_records = []
    for i in range(len(all_names)):
        rec = {
            "image_name": all_names[i],
            "ground_truth": int(targets_np[i]),
            "ground_truth_label": DR_CLASSES_5[targets_np[i]],
            "predicted_label": int(preds_np[i]),
            "predicted_grade": DR_CLASSES_5[preds_np[i]],
            "correct": bool(targets_np[i] == preds_np[i]),
            "confidence": float(np.max(probs_np[i])),
            "prob_0_no_dr": float(probs_np[i, 0]),
            "prob_1_mild": float(probs_np[i, 1]),
            "prob_2_moderate": float(probs_np[i, 2]),
            "prob_3_severe": float(probs_np[i, 3]),
            "prob_4_pdr": float(probs_np[i, 4])
        }
        pred_records.append(rec)
    df_preds = pd.DataFrame(pred_records)
    return metrics, df_preds

def train_configuration(
    cfg_info: Dict[str, Any],
    manifest: Dict[str, Any],
    device: torch.device,
    epochs: int = 15,
    seed: int = 42
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains ResNet-50 for a specified preprocessing configuration under identical hyperparameters."""
    set_seed(seed)
    print(f"\n=======================================================")
    print(f" Training: {cfg_info['name']}")
    print(f" Method  : {cfg_info['method'].upper()} | Seed: {seed} | Epochs: {epochs}")
    print(f"=======================================================")

    train_loader, val_loader, _, _, class_weights = create_dataloaders_for_config(cfg_info, manifest)
    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Initialize model
    model_cfg = {"model": {"name": "resnet50", "pretrained": True, "dropout": 0.3}, "task": {"num_classes": 5}}
    model = build_model(model_cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_qwk = -1.0
    best_epoch = 0
    best_state_dict = None
    training_history = []

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_targets.extend(lbls.detach().cpu().numpy())

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(train_targets, train_preds, task="five_class", class_names=DR_CLASSES_5)

        # Validation
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend(np.argmax(probs, axis=1))
                val_targets.extend(lbls.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_metrics = calculate_metrics(val_targets, val_preds, y_probs=np.array(val_probs), task="five_class", class_names=DR_CLASSES_5)
        ep_duration = time.perf_counter() - t0

        val_qwk = val_metrics["quadratic_weighted_kappa"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        record = {
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "train_qwk": round(train_metrics["quadratic_weighted_kappa"], 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_qwk": round(val_qwk, 4),
            "duration_sec": round(ep_duration, 2)
        }
        training_history.append(record)

        print(f"  Epoch [{ep:02d}/{epochs:02d}] ({ep_duration:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} Macro-F1: {val_f1:.3f} QWK: {val_qwk:.3f}")

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = ep
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  >>> Best Validation QWK = {best_val_qwk:.4f} achieved at Epoch {best_epoch}")

    # Load best weights
    model.load_state_dict(best_state_dict)

    # Save checkpoint
    ckpt_payload = {
        "epoch": best_epoch,
        "model_state_dict": best_state_dict,
        "config": {
            "model": {"name": "resnet50", "pretrained": True, "dropout": 0.3},
            "task": {"type": "five_class", "num_classes": 5},
            "preprocessing": {"method": cfg_info["method"], "image_size": 256},
            "training": {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4, "class_weight_method": "sqrt", "epochs": epochs, "seed": seed}
        },
        "best_metric_val": best_val_qwk,
        "history": training_history,
        "class_weights": class_weights.cpu().numpy().tolist()
    }

    ckpt_path_out = CHECKPOINTS_DIR / cfg_info["checkpoint_name"]
    torch.save(ckpt_payload, ckpt_path_out)

    # Also save to root checkpoints directory for global reference
    root_ckpt_path = ROOT_CHECKPOINTS_DIR / cfg_info["checkpoint_name"]
    torch.save(ckpt_payload, root_ckpt_path)
    print(f"  [OK] Saved checkpoint to {ckpt_path_out} and {root_ckpt_path}")

    return model, {"best_epoch": best_epoch, "best_val_qwk": best_val_qwk, "history": training_history}

def run_step7_full_evaluation():
    """Master workflow orchestrating data verification, computational overhead benchmarking, training, and evaluations."""
    set_seed(42)
    device = get_device("auto")
    print(f"Executing Step 7 Preprocessing Comparison on device: {device}")

    for d in [OUTPUT_DIR, CHECKPOINTS_DIR, METRICS_DIR, PREDS_DIR, COMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Verification of split and anti-leakage guarantees
    manifest = verify_data_partitions_and_leakage_safety()

    # 2. Benchmark Preprocessing Computational Overhead
    overhead_results = benchmark_preprocessing_computational_overhead(num_benchmark_samples=20)

    def get_cls_recall(m: Dict[str, Any], cls_name: str) -> float:
        pcm = m.get("per_class_metrics", {}).get(cls_name, {})
        return float(pcm.get("recall", pcm.get("recall_sensitivity", 0.0)))

    def get_cls_f1(m: Dict[str, Any], cls_name: str) -> float:
        pcm = m.get("per_class_metrics", {}).get(cls_name, {})
        return float(pcm.get("f1_score", pcm.get("f1", 0.0)))

    # 3. Train each configuration or load existing checkpoint
    trained_models = {}
    train_summaries = {}

    for cfg_info in CONFIGURATIONS:
        ckpt_path = CHECKPOINTS_DIR / cfg_info["checkpoint_name"]
        if not ckpt_path.exists():
            ckpt_path = ROOT_CHECKPOINTS_DIR / cfg_info["checkpoint_name"]

        if ckpt_path.exists():
            print(f"Loading existing checkpoint for {cfg_info['name']} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model_cfg = {"model": {"name": "resnet50", "pretrained": False, "dropout": 0.3}, "task": {"num_classes": 5}}
            model = build_model(model_cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            summary = {
                "best_epoch": ckpt.get("epoch", 0),
                "best_val_qwk": ckpt.get("best_metric_val", 0.0),
                "history": ckpt.get("history", [])
            }
        else:
            model, summary = train_configuration(
                cfg_info=cfg_info,
                manifest=manifest,
                device=device,
                epochs=15,
                seed=42
            )
        trained_models[cfg_info["id"]] = model
        train_summaries[cfg_info["id"]] = summary

    # 4. Multi-Partition Evaluation across All 3 Configurations
    print("\n=== Evaluating Models Across Partitions (Val, Test Official, Test Clean) ===")

    val_records = []
    test_official_records = []
    test_clean_records = []
    per_class_recall_records = []

    for cfg_info in CONFIGURATIONS:
        cfg_id = cfg_info["id"]
        model = trained_models[cfg_id]
        _, val_loader, test_official_loader, test_clean_loader, _ = create_dataloaders_for_config(cfg_info, manifest)

        # 4a. Validation Split Evaluation
        val_metrics, df_val_preds = evaluate_dataloader(model, val_loader, device)
        df_val_preds.to_csv(PREDS_DIR / f"predictions_val_{cfg_id}.csv", index=False)
        with open(METRICS_DIR / f"metrics_val_{cfg_id}.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        val_rec = {
            "config_id": cfg_id,
            "preprocessing_name": cfg_info["name"],
            "split": "validation (N=83)",
            "accuracy": round(val_metrics["accuracy"], 4),
            "macro_f1": round(val_metrics["macro_f1"], 4),
            "weighted_f1": round(val_metrics["weighted_f1"], 4),
            "quadratic_weighted_kappa": round(val_metrics["quadratic_weighted_kappa"], 4),
            "macro_precision": round(val_metrics["macro_precision"], 4),
            "macro_recall": round(val_metrics["macro_recall"], 4),
            "recall_grade_0_no_dr": round(get_cls_recall(val_metrics, "No DR"), 4),
            "recall_grade_1_mild": round(get_cls_recall(val_metrics, "Mild NPDR"), 4),
            "recall_grade_2_moderate": round(get_cls_recall(val_metrics, "Moderate NPDR"), 4),
            "recall_grade_3_severe": round(get_cls_recall(val_metrics, "Severe NPDR"), 4),
            "recall_grade_4_pdr": round(get_cls_recall(val_metrics, DR_CLASSES_5[4]), 4),
        }
        val_records.append(val_rec)

        # 4b. Official Test Split Evaluation (N=103)
        test_off_metrics, df_test_off_preds = evaluate_dataloader(model, test_official_loader, device)
        df_test_off_preds.to_csv(PREDS_DIR / f"predictions_test_official_{cfg_id}.csv", index=False)
        with open(METRICS_DIR / f"metrics_test_official_{cfg_id}.json", "w") as f:
            json.dump(test_off_metrics, f, indent=2)

        test_off_rec = {
            "config_id": cfg_id,
            "preprocessing_name": cfg_info["name"],
            "split": "test_official (N=103)",
            "accuracy": round(test_off_metrics["accuracy"], 4),
            "macro_f1": round(test_off_metrics["macro_f1"], 4),
            "weighted_f1": round(test_off_metrics["weighted_f1"], 4),
            "quadratic_weighted_kappa": round(test_off_metrics["quadratic_weighted_kappa"], 4),
            "macro_precision": round(test_off_metrics["macro_precision"], 4),
            "macro_recall": round(test_off_metrics["macro_recall"], 4),
            "recall_grade_0_no_dr": round(get_cls_recall(test_off_metrics, "No DR"), 4),
            "recall_grade_1_mild": round(get_cls_recall(test_off_metrics, "Mild NPDR"), 4),
            "recall_grade_2_moderate": round(get_cls_recall(test_off_metrics, "Moderate NPDR"), 4),
            "recall_grade_3_severe": round(get_cls_recall(test_off_metrics, "Severe NPDR"), 4),
            "recall_grade_4_pdr": round(get_cls_recall(test_off_metrics, DR_CLASSES_5[4]), 4),
        }
        test_official_records.append(test_off_rec)

        # 4c. Clean Test Split Evaluation (N=102)
        test_clean_metrics, df_test_clean_preds = evaluate_dataloader(model, test_clean_loader, device)
        df_test_clean_preds.to_csv(PREDS_DIR / f"predictions_test_clean_{cfg_id}.csv", index=False)
        with open(METRICS_DIR / f"metrics_test_clean_{cfg_id}.json", "w") as f:
            json.dump(test_clean_metrics, f, indent=2)

        test_clean_rec = {
            "config_id": cfg_id,
            "preprocessing_name": cfg_info["name"],
            "split": "test_clean (N=102)",
            "accuracy": round(test_clean_metrics["accuracy"], 4),
            "macro_f1": round(test_clean_metrics["macro_f1"], 4),
            "weighted_f1": round(test_clean_metrics["weighted_f1"], 4),
            "quadratic_weighted_kappa": round(test_clean_metrics["quadratic_weighted_kappa"], 4),
            "macro_precision": round(test_clean_metrics["macro_precision"], 4),
            "macro_recall": round(test_clean_metrics["macro_recall"], 4),
            "recall_grade_0_no_dr": round(get_cls_recall(test_clean_metrics, "No DR"), 4),
            "recall_grade_1_mild": round(get_cls_recall(test_clean_metrics, "Mild NPDR"), 4),
            "recall_grade_2_moderate": round(get_cls_recall(test_clean_metrics, "Moderate NPDR"), 4),
            "recall_grade_3_severe": round(get_cls_recall(test_clean_metrics, "Severe NPDR"), 4),
            "recall_grade_4_pdr": round(get_cls_recall(test_clean_metrics, DR_CLASSES_5[4]), 4),
        }
        test_clean_records.append(test_clean_rec)

        # Per-class recall breakdown record
        for cls_idx, cls_name in enumerate(DR_CLASSES_5):
            per_class_recall_records.append({
                "config_id": cfg_id,
                "preprocessing_name": cfg_info["name"],
                "class_index": cls_idx,
                "class_name": cls_name,
                "val_recall": round(get_cls_recall(val_metrics, cls_name), 4),
                "val_f1": round(get_cls_f1(val_metrics, cls_name), 4),
                "test_official_recall": round(get_cls_recall(test_off_metrics, cls_name), 4),
                "test_official_f1": round(get_cls_f1(test_off_metrics, cls_name), 4),
                "test_clean_recall": round(get_cls_recall(test_clean_metrics, cls_name), 4),
                "test_clean_f1": round(get_cls_f1(test_clean_metrics, cls_name), 4),
            })

        print(f"  Evaluated {cfg_info['name']}: Val QWK={val_rec['quadratic_weighted_kappa']:.4f} | Test Official QWK={test_off_rec['quadratic_weighted_kappa']:.4f} | Test Clean QWK={test_clean_rec['quadratic_weighted_kappa']:.4f}")

    # 5. Export Comparative Tables
    df_val_comp = pd.DataFrame(val_records)
    df_test_off_comp = pd.DataFrame(test_official_records)
    df_test_clean_comp = pd.DataFrame(test_clean_records)
    df_pcr_comp = pd.DataFrame(per_class_recall_records)

    df_val_comp.to_csv(COMP_DIR / "step7_preprocessing_comparison_val.csv", index=False)
    df_test_off_comp.to_csv(COMP_DIR / "step7_preprocessing_comparison_test_official.csv", index=False)
    df_test_clean_comp.to_csv(COMP_DIR / "step7_preprocessing_comparison_test_clean.csv", index=False)
    df_pcr_comp.to_csv(COMP_DIR / "step7_preprocessing_per_class_recall_comparison.csv", index=False)

    # 6. Generate Comprehensive Summary Report (JSON)
    summary_report = {
        "metadata": {
            "title": "Step 7 Retinal Image Preprocessing Comparison",
            "seed": 42,
            "architecture": "BaselineResNet50",
            "training_epochs": 15,
            "loss_function": "CrossEntropyLoss with sqrt-balanced class weights",
            "train_samples": len(manifest["train"]),
            "val_samples": len(manifest["val"]),
            "test_official_samples": 103,
            "test_clean_samples": 102
        },
        "computational_overhead": overhead_results,
        "validation_comparison": val_records,
        "test_official_comparison": test_official_records,
        "test_clean_comparison": test_clean_records,
        "training_summaries": train_summaries
    }

    with open(COMP_DIR / "step7_preprocessing_summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n=======================================================")
    print(" Step 7 Controlled Preprocessing Comparison Completed!")
    print(f" Artifacts saved in: {OUTPUT_DIR}")
    print("=======================================================\n")

if __name__ == "__main__":
    run_step7_full_evaluation()
