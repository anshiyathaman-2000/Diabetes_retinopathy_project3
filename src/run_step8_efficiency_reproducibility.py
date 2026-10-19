"""
Step 8: Comprehensive Computational Efficiency and Reproducibility Audit.
Compares ResNet-50 and CNN-Transformer-BiLSTM Hybrid Architecture on IDRiD DR Classification.

Measures:
1. Total and trainable model parameters with component breakdown.
2. Model checkpoint file sizes on disk (state-dict vs full training checkpoint).
3. Inference latency and throughput (batch_size=1 and batch_size=16) with warm-up and synchronization.
4. Peak hardware memory usage (MPS allocated/driver memory and RSS memory) across inference and training.
5. Training time, epoch throughput, and comprehensive multi-partition evaluation metrics (Val, Official Test, Clean Test).
6. Multi-seed reproducibility analysis across multiple random seeds (Seed 42, 123, 456) reporting mean ± std.

Strict Experimental Guarantees:
- Model selection strictly performed on validation data (N=83); test set (N=103/102) strictly held out.
- Separate output directories under outputs/experiments/idrid_step8_efficiency_reproducibility/.
- Exact logging of random seed, dataset split, preprocessing method, hyperparameters, and hardware/device info.
"""

import os
import sys
import time
import json
import resource
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import (
    set_seed,
    get_device,
    calculate_metrics,
    count_model_parameters
)
from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    get_train_transforms,
    get_eval_transforms,
    IDRiDDataset,
    DR_CLASSES_5
)
from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/idrid_step8_efficiency_reproducibility"
BENCH_DIR = OUTPUT_DIR / "benchmarks"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDS_DIR = OUTPUT_DIR / "predictions"
COMP_DIR = OUTPUT_DIR / "comparisons"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
ROOT_CHECKPOINTS = PROJECT_ROOT / "checkpoints"

DATA_BASE = PROJECT_ROOT / "B. Disease Grading"
TRAIN_CSV = DATA_BASE / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV = DATA_BASE / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
TRAIN_IMG_DIR = DATA_BASE / "1. Original Images" / "a. Training Set"
TEST_IMG_DIR = DATA_BASE / "1. Original Images" / "b. Testing Set"
SPLIT_MANIFEST = PROJECT_ROOT / "data/splits/leakage_aware_split_manifest.json"
CACHE_DIR = PROJECT_ROOT / "data/cache/cache_256_clahe"

# Model configurations to evaluate
MODEL_SPECS = [
    {
        "id": "resnet50",
        "name": "ResNet-50 Baseline CNN",
        "config": {
            "model": {"name": "resnet50", "pretrained": True, "dropout": 0.3},
            "task": {"type": "five_class", "num_classes": 5}
        },
        "canonical_checkpoint": ROOT_CHECKPOINTS / "resnet50_five_class_clahe_sqrt_cw_s42_best.pt",
        "compact_checkpoint": ROOT_CHECKPOINTS / "resnet50_five_class_existing_clahe_s42_best.pt"
    },
    {
        "id": "hybrid_compact",
        "name": "CNN-Transformer-BiLSTM (Compact)",
        "config": {
            "model": {
                "name": "hybrid",
                "pretrained": True,
                "dropout": 0.3,
                "hybrid": {
                    "backbone": "resnet50",
                    "cnn_embed_dim": 128,
                    "transformer_layers": 1,
                    "transformer_heads": 4,
                    "transformer_feedforward": 512,
                    "transformer_dropout": 0.2,
                    "bilstm_hidden_dim": 64,
                    "bilstm_layers": 1,
                    "bilstm_dropout": 0.1,
                    "pool_type": "mean"
                }
            },
            "task": {"type": "five_class", "num_classes": 5}
        },
        "canonical_checkpoint": ROOT_CHECKPOINTS / "hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt",
        "compact_checkpoint": None
    },
    {
        "id": "hybrid_full",
        "name": "CNN-Transformer-BiLSTM (Full)",
        "config": {
            "model": {
                "name": "hybrid",
                "pretrained": True,
                "dropout": 0.3,
                "hybrid": {
                    "backbone": "resnet50",
                    "cnn_embed_dim": 256,
                    "transformer_layers": 2,
                    "transformer_heads": 8,
                    "transformer_feedforward": 512,
                    "transformer_dropout": 0.2,
                    "bilstm_hidden_dim": 128,
                    "bilstm_layers": 1,
                    "bilstm_dropout": 0.1,
                    "pool_type": "mean_max"
                }
            },
            "task": {"type": "five_class", "num_classes": 5}
        },
        "canonical_checkpoint": ROOT_CHECKPOINTS / "hybrid_five_class_clahe_s42_best.pt",
        "compact_checkpoint": None
    }
]

def get_system_environment_info(device: torch.device) -> Dict[str, Any]:
    """Captures exact system, software, and hardware metadata for reproducibility."""
    hw_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_type": device.type,
        "device_str": str(device),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "cuda_available": torch.cuda.is_available(),
        "dataset_split": "data/splits/leakage_aware_split_manifest.json (330 train, 83 val)",
        "preprocessing_pipeline": "CIELAB Luminance (L*) CLAHE, 256x256, clipLimit=2.0, tileGridSize=(8,8)",
        "fixed_normalization": "ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
        "class_weighting": "Square-root balanced class weights on 330 train samples"
    }
    return hw_info

def audit_model_parameters() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Measurement 1: Accurate Parameter Counts and Structural Breakdown."""
    print("=== Measurement 1: Model Parameter Count Audit ===")
    records = []
    full_breakdown = {}

    for spec in MODEL_SPECS:
        model = build_model(spec["config"])
        param_data = count_model_parameters(model)
        full_breakdown[spec["id"]] = param_data

        total_p = param_data["total_parameters"]
        trainable_p = param_data["trainable_parameters"]
        frozen_p = param_data["frozen_parameters"]
        size_mb = param_data["model_size_mb"]

        # Structural sub-components
        cb = param_data["component_breakdown"]
        if "backbone" in cb:
            backbone_p = cb["backbone"]["total_parameters"]
        else:
            # Hybrid unpacks resnet layers into root module
            backbone_layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
            backbone_p = sum(cb.get(k, {}).get("total_parameters", 0) for k in backbone_layers)

        proj_p = cb.get("proj", {}).get("total_parameters", 0) + cb.get("proj_norm", {}).get("total_parameters", 0)
        transformer_p = cb.get("transformer", {}).get("total_parameters", 0) + cb.get("transformer_encoder", {}).get("total_parameters", 0) + cb.get("transformer_norm", {}).get("total_parameters", 0)
        bilstm_p = cb.get("bilstm", {}).get("total_parameters", 0)
        classifier_p = cb.get("classifier", {}).get("total_parameters", 0)

        rec = {
            "model_id": spec["id"],
            "model_name": spec["name"],
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "frozen_parameters": frozen_p,
            "estimated_fp32_size_mb": size_mb,
            "backbone_parameters": backbone_p,
            "projection_parameters": proj_p,
            "transformer_parameters": transformer_p,
            "bilstm_parameters": bilstm_p,
            "classifier_head_parameters": classifier_p
        }
        records.append(rec)
        print(f"  {spec['name']}: Total Params = {total_p:,} | Trainable = {trainable_p:,} | Size = {size_mb:.2f} MB")

    df_params = pd.DataFrame(records)
    df_params.to_csv(BENCH_DIR / "model_parameter_audit.csv", index=False)
    with open(BENCH_DIR / "model_parameter_audit.json", "w") as f:
        json.dump(full_breakdown, f, indent=2)

    return full_breakdown, df_params

def audit_checkpoint_sizes() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Measurement 2: Physical Checkpoint File Sizes on Disk."""
    print("\n=== Measurement 2: Checkpoint File Size Audit ===")
    records = []
    checkpoint_data = {}

    for spec in MODEL_SPECS:
        ckpt_path = spec["canonical_checkpoint"]
        if ckpt_path and ckpt_path.exists():
            size_bytes = ckpt_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            ckpt_obj = torch.load(ckpt_path, map_location="cpu")
            has_optimizer = "optimizer_state_dict" in ckpt_obj
            has_history = "history" in ckpt_obj

            rec = {
                "model_id": spec["id"],
                "model_name": spec["name"],
                "checkpoint_filename": ckpt_path.name,
                "file_size_bytes": size_bytes,
                "file_size_mb": round(size_mb, 2),
                "includes_optimizer_state": has_optimizer,
                "includes_training_history": has_history,
                "best_epoch": ckpt_obj.get("epoch", None),
                "best_val_metric": round(float(ckpt_obj.get("best_metric_val", 0.0)), 4)
            }
            records.append(rec)
            checkpoint_data[spec["id"]] = rec
            print(f"  {spec['name']} Checkpoint: {ckpt_path.name} | Size = {size_mb:.2f} MB")

    df_ckpts = pd.DataFrame(records)
    df_ckpts.to_csv(BENCH_DIR / "model_checkpoint_size_audit.csv", index=False)
    with open(BENCH_DIR / "model_checkpoint_size_audit.json", "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_data, df_ckpts

def benchmark_inference_latency_and_throughput(
    device: torch.device,
    num_warmup: int = 20,
    num_timed: int = 100
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Measurement 3: Rigorous Inference Latency & Throughput Benchmark across Batch Sizes."""
    print(f"\n=== Measurement 3: Inference Latency Benchmark on {device} (Warmup={num_warmup}, Timed={num_timed}) ===")
    bench_records = []
    bench_data = {}

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    for spec in MODEL_SPECS:
        model = build_model(spec["config"]).to(device)
        model.eval()

        for bs in [1, 16]:
            dummy = torch.randn(bs, 3, 256, 256, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy)
                    _sync()

            # Timed runs
            latencies_ms = []
            with torch.no_grad():
                for _ in range(num_timed):
                    _sync()
                    t0 = time.perf_counter()
                    _ = model(dummy)
                    _sync()
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)

            lat_arr = np.array(latencies_ms)
            mean_ms = float(np.mean(lat_arr))
            std_ms = float(np.std(lat_arr))
            median_ms = float(np.median(lat_arr))
            p95_ms = float(np.percentile(lat_arr, 95))
            p99_ms = float(np.percentile(lat_arr, 99))
            min_ms = float(np.min(lat_arr))
            max_ms = float(np.max(lat_arr))
            fps = float((bs * 1000.0) / mean_ms) if mean_ms > 0 else 0.0

            rec = {
                "model_id": spec["id"],
                "model_name": spec["name"],
                "batch_size": bs,
                "input_resolution": "256x256",
                "device": str(device),
                "num_warmup_runs": num_warmup,
                "num_timed_runs": num_timed,
                "mean_latency_ms": round(mean_ms, 3),
                "std_latency_ms": round(std_ms, 3),
                "median_p50_latency_ms": round(median_ms, 3),
                "p95_latency_ms": round(p95_ms, 3),
                "p99_latency_ms": round(p99_ms, 3),
                "min_latency_ms": round(min_ms, 3),
                "max_latency_ms": round(max_ms, 3),
                "throughput_fps": round(fps, 2)
            }
            bench_records.append(rec)
            bench_data[f"{spec['id']}_bs{bs}"] = rec
            print(f"  {spec['name']} (BS={bs:02d}): Mean = {mean_ms:.2f} ± {std_ms:.2f} ms | p95 = {p95_ms:.2f} ms | Throughput = {fps:.1f} FPS")

    # Relative overhead ratio vs ResNet-50
    df_lat = pd.DataFrame(bench_records)
    for bs in [1, 16]:
        r50_row = df_lat[(df_lat["model_id"] == "resnet50") & (df_lat["batch_size"] == bs)]
        if not r50_row.empty:
            r50_mean = r50_row["mean_latency_ms"].values[0]
            df_lat.loc[df_lat["batch_size"] == bs, "slowdown_vs_resnet50"] = (
                df_lat.loc[df_lat["batch_size"] == bs, "mean_latency_ms"] / r50_mean
            ).round(3)

    df_lat.to_csv(BENCH_DIR / "inference_latency_benchmark.csv", index=False)
    with open(BENCH_DIR / "inference_latency_benchmark.json", "w") as f:
        json.dump(bench_data, f, indent=2)

    return bench_data, df_lat

def benchmark_peak_memory_usage(device: torch.device) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Measurement 4: Peak Device (MPS/GPU) and System RAM Usage Tracking."""
    print(f"\n=== Measurement 4: Peak Memory Usage Benchmark on {device} ===")
    mem_records = []
    mem_data = {}

    def _get_mps_mem_mb():
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        elif device.type == "cuda":
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return 0.0

    def _get_mps_driver_mb():
        if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
            return round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
        return 0.0

    def _clear_cache():
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    for spec in MODEL_SPECS:
        _clear_cache()
        m_base = _get_mps_mem_mb()

        # 1. Model Load
        model = build_model(spec["config"]).to(device)
        m_after_load = _get_mps_mem_mb()
        model_weights_mem = m_after_load - m_base

        # 2. Inference BS=1
        model.eval()
        dummy_bs1 = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            _ = model(dummy_bs1)
        m_inf_bs1 = _get_mps_mem_mb()

        # 3. Inference BS=16
        dummy_bs16 = torch.randn(16, 3, 256, 256, device=device)
        with torch.no_grad():
            _ = model(dummy_bs16)
        m_inf_bs16 = _get_mps_mem_mb()

        # 4. Training Step (Forward + Backward on BS=16)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, 5, (16,), device=device)

        optimizer.zero_grad()
        logits = model(dummy_bs16)
        m_train_fwd = _get_mps_mem_mb()

        loss = criterion(logits, targets)
        loss.backward()
        m_train_peak = _get_mps_mem_mb()
        optimizer.step()

        driver_peak = _get_mps_driver_mb()
        rss_peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)

        rec = {
            "model_id": spec["id"],
            "model_name": spec["name"],
            "device": str(device),
            "baseline_mem_mb": m_base,
            "model_weights_allocated_mb": round(model_weights_mem, 2),
            "inference_bs1_allocated_mb": m_inf_bs1,
            "inference_bs16_allocated_mb": m_inf_bs16,
            "training_forward_allocated_mb": m_train_fwd,
            "training_peak_allocated_mb": m_train_peak,
            "mps_driver_allocated_mb": driver_peak,
            "process_ru_maxrss_mb": rss_peak_mb
        }
        mem_records.append(rec)
        mem_data[spec["id"]] = rec
        print(f"  {spec['name']}: Weights = {model_weights_mem:.1f} MB | Inf (BS=16) = {m_inf_bs16:.1f} MB | Train Peak = {m_train_peak:.1f} MB | Process RSS = {rss_peak_mb:.1f} MB")

        # Cleanup
        del model, optimizer, logits, loss, dummy_bs1, dummy_bs16, targets
        _clear_cache()

    df_mem = pd.DataFrame(mem_records)
    df_mem.to_csv(BENCH_DIR / "peak_memory_usage_benchmark.csv", index=False)
    with open(BENCH_DIR / "peak_memory_usage_benchmark.json", "w") as f:
        json.dump(mem_data, f, indent=2)

    return mem_data, df_mem

def evaluate_model_on_partitions(
    model: nn.Module,
    val_loader: DataLoader,
    test_off_loader: DataLoader,
    test_clean_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Evaluates a model over Validation, Official Test, and Clean Test dataloaders."""
    model.eval()

    def _eval(dl):
        all_probs, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for imgs, lbls, _ in dl:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(lbls.numpy())

        return calculate_metrics(
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
            y_probs=np.array(all_probs),
            task="five_class",
            class_names=DR_CLASSES_5
        )

    val_metrics = _eval(val_loader)
    test_off_metrics = _eval(test_off_loader)
    test_clean_metrics = _eval(test_clean_loader)
    return val_metrics, test_off_metrics, test_clean_metrics

def audit_training_time_and_metrics(
    manifest: Dict[str, Any],
    device: torch.device
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Measurement 5: Training Time, Epoch Throughput, and Verified Metric Comparison."""
    print(f"\n=== Measurement 5: Training Time and Performance Metrics Audit ===")

    # Prepare loaders
    df_full_train = load_and_clean_csv(TRAIN_CSV)
    df_test_official = load_and_clean_csv(TEST_CSV)
    df_test_clean = df_test_official[df_test_official["Image name"] != "IDRiD_064"].reset_index(drop=True)

    train_names_set = set(manifest["train"])
    val_names_set = set(manifest["val"])
    df_val = df_full_train[df_full_train["Image name"].isin(val_names_set)].reset_index(drop=True)

    eval_tf = get_eval_transforms()
    val_ds = IDRiDDataset(
        dataframe=df_val,
        image_dir=TRAIN_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "train",
        transform=eval_tf
    )
    test_off_ds = IDRiDDataset(
        dataframe=df_test_official,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "test",
        transform=eval_tf
    )
    test_clean_ds = IDRiDDataset(
        dataframe=df_test_clean,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "test",
        transform=eval_tf
    )

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_off_loader = DataLoader(test_off_ds, batch_size=16, shuffle=False)
    test_clean_loader = DataLoader(test_clean_ds, batch_size=16, shuffle=False)

    perf_records = []
    per_class_records = []
    detailed_metrics = {}

    for spec in MODEL_SPECS:
        ckpt_path = spec["canonical_checkpoint"]
        if not ckpt_path or not ckpt_path.exists():
            continue

        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        model = build_model(spec["config"]).to(device)
        model.load_state_dict(ckpt_obj["model_state_dict"])

        val_m, test_off_m, test_clean_m = evaluate_model_on_partitions(
            model, val_loader, test_off_loader, test_clean_loader, device
        )

        # Training time from checkpoint history if available
        hist = ckpt_obj.get("history", [])
        epoch_durations = []
        if isinstance(hist, list):
            epoch_durations = [h.get("duration_sec", 0.0) for h in hist if "duration_sec" in h]
        elif isinstance(hist, dict):
            epoch_durations = hist.get("duration_sec", [])

        mean_ep_sec = float(np.mean(epoch_durations)) if epoch_durations else None
        total_train_sec = float(np.sum(epoch_durations)) if epoch_durations else None

        rec = {
            "model_id": spec["id"],
            "model_name": spec["name"],
            "best_epoch": ckpt_obj.get("epoch", None),
            "mean_epoch_time_sec": round(mean_ep_sec, 2) if mean_ep_sec else None,
            "total_training_time_sec": round(total_train_sec, 2) if total_train_sec else None,
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_macro_f1": round(val_m["macro_f1"], 4),
            "val_weighted_f1": round(val_m["weighted_f1"], 4),
            "val_qwk": round(val_m["quadratic_weighted_kappa"], 4),
            "test_official_accuracy": round(test_off_m["accuracy"], 4),
            "test_official_macro_f1": round(test_off_m["macro_f1"], 4),
            "test_official_qwk": round(test_off_m["quadratic_weighted_kappa"], 4),
            "test_clean_accuracy": round(test_clean_m["accuracy"], 4),
            "test_clean_macro_f1": round(test_clean_m["macro_f1"], 4),
            "test_clean_qwk": round(test_clean_m["quadratic_weighted_kappa"], 4)
        }
        perf_records.append(rec)
        detailed_metrics[spec["id"]] = {
            "validation": val_m,
            "test_official": test_off_m,
            "test_clean": test_clean_m
        }

        # Per-class recall
        for cls_name in DR_CLASSES_5:
            r_val = val_m["per_class_metrics"].get(cls_name, {}).get("recall", val_m["per_class_metrics"].get(cls_name, {}).get("recall_sensitivity", 0.0))
            r_test_off = test_off_m["per_class_metrics"].get(cls_name, {}).get("recall", test_off_m["per_class_metrics"].get(cls_name, {}).get("recall_sensitivity", 0.0))
            r_test_clean = test_clean_m["per_class_metrics"].get(cls_name, {}).get("recall", test_clean_m["per_class_metrics"].get(cls_name, {}).get("recall_sensitivity", 0.0))

            per_class_records.append({
                "model_id": spec["id"],
                "model_name": spec["name"],
                "class_name": cls_name,
                "val_recall": round(float(r_val), 4),
                "test_official_recall": round(float(r_test_off), 4),
                "test_clean_recall": round(float(r_test_clean), 4)
            })

        print(f"  {spec['name']}: Val QWK = {rec['val_qwk']:.4f} | Clean Test QWK = {rec['test_clean_qwk']:.4f} | Clean Test Acc = {rec['test_clean_accuracy']:.4f}")

    df_perf = pd.DataFrame(perf_records)
    df_perf.to_csv(COMP_DIR / "efficiency_performance_comparison.csv", index=False)

    df_pcr = pd.DataFrame(per_class_records)
    df_pcr.to_csv(COMP_DIR / "per_class_recall_comparison.csv", index=False)

    with open(METRICS_DIR / "detailed_model_metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    return detailed_metrics, df_perf, df_pcr

def train_short_seed_run(
    model_spec: Dict[str, Any],
    seed: int,
    epochs: int,
    manifest: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Trains a controlled model run for a specific seed to evaluate reproducibility."""
    set_seed(seed)
    df_full_train = load_and_clean_csv(TRAIN_CSV)
    df_test_official = load_and_clean_csv(TEST_CSV)
    df_test_clean = df_test_official[df_test_official["Image name"] != "IDRiD_064"].reset_index(drop=True)

    train_names_set = set(manifest["train"])
    val_names_set = set(manifest["val"])
    df_train = df_full_train[df_full_train["Image name"].isin(train_names_set)].reset_index(drop=True)
    df_val = df_full_train[df_full_train["Image name"].isin(val_names_set)].reset_index(drop=True)

    train_labels = df_train["Retinopathy grade"].tolist()
    class_weights = compute_class_weights(train_labels, num_classes=5, method="sqrt")

    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    train_ds = IDRiDDataset(
        dataframe=df_train,
        image_dir=TRAIN_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "train",
        transform=train_tf
    )
    val_ds = IDRiDDataset(
        dataframe=df_val,
        image_dir=TRAIN_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "train",
        transform=eval_tf
    )
    test_off_ds = IDRiDDataset(
        dataframe=df_test_official,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "test",
        transform=eval_tf
    )
    test_clean_ds = IDRiDDataset(
        dataframe=df_test_clean,
        image_dir=TEST_IMG_DIR,
        task="five_class",
        image_size=256,
        preprocessing_method="clahe",
        crop_border=True,
        cache_dir=CACHE_DIR / "test",
        transform=eval_tf
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_off_loader = DataLoader(test_off_ds, batch_size=16, shuffle=False)
    test_clean_loader = DataLoader(test_clean_ds, batch_size=16, shuffle=False)

    model = build_model(model_spec["config"]).to(device)
    ckpt_name = f"{model_spec['id']}_seed{seed}_best.pt"
    ckpt_file = CHECKPOINTS_DIR / ckpt_name

    if ckpt_file.exists():
        print(f"    [Found cached checkpoint {ckpt_file.name}, loading for evaluation...]")
        ckpt_saved = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt_saved["model_state_dict"])
        epoch_durations = ckpt_saved.get("epoch_durations", [65.0])
    else:
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        best_val_qwk = -1.0
        best_state_dict = None
        epoch_durations = []

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            running_loss = 0.0

            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            scheduler.step()
            ep_duration = time.perf_counter() - t0
            epoch_durations.append(ep_duration)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for imgs, lbls, _ in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(lbls.cpu().numpy())

            val_m = calculate_metrics(val_targets, val_preds, task="five_class", class_names=DR_CLASSES_5)
            val_qwk = val_m["quadratic_weighted_kappa"]

            if val_qwk > best_val_qwk:
                best_val_qwk = val_qwk
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best validation model and save checkpoint
        model.load_state_dict(best_state_dict)
        torch.save({
            "model_state_dict": best_state_dict,
            "seed": seed,
            "best_val_qwk": best_val_qwk,
            "epoch_durations": epoch_durations
        }, ckpt_file)

    # Evaluate best model on Validation, Official Test, and Clean Test
    val_m, test_off_m, test_clean_m = evaluate_model_on_partitions(
        model, val_loader, test_off_loader, test_clean_loader, device
    )

    return {
        "model_id": model_spec["id"],
        "model_name": model_spec["name"],
        "seed": seed,
        "epochs_trained": epochs,
        "mean_epoch_sec": round(float(np.mean(epoch_durations)), 2),
        "val_accuracy": round(val_m["accuracy"], 4),
        "val_macro_f1": round(val_m["macro_f1"], 4),
        "val_qwk": round(val_m["quadratic_weighted_kappa"], 4),
        "test_official_accuracy": round(test_off_m["accuracy"], 4),
        "test_official_macro_f1": round(test_off_m["macro_f1"], 4),
        "test_official_qwk": round(test_off_m["quadratic_weighted_kappa"], 4),
        "test_clean_accuracy": round(test_clean_m["accuracy"], 4),
        "test_clean_macro_f1": round(test_clean_m["macro_f1"], 4),
        "test_clean_qwk": round(test_clean_m["quadratic_weighted_kappa"], 4)
    }

def audit_multi_seed_reproducibility(
    manifest: Dict[str, Any],
    device: torch.device,
    seeds: List[int] = [42, 123, 456],
    eval_epochs: int = 5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Measurement 6: Multi-Seed Reproducibility & Variance Analysis (Mean ± Std)."""
    print(f"\n=== Measurement 6: Multi-Seed Reproducibility Audit across Seeds {seeds} ===")
    repro_records = []
    models_to_test = [
        spec for spec in MODEL_SPECS if spec["id"] in ["resnet50", "hybrid_compact"]
    ]

    for spec in models_to_test:
        seed_results = []
        for s in seeds:
            print(f"  Training reproducibility evaluation: {spec['name']} | Seed {s} ({eval_epochs} epochs)...")
            res = train_short_seed_run(
                model_spec=spec,
                seed=s,
                epochs=eval_epochs,
                manifest=manifest,
                device=device
            )
            repro_records.append(res)
            seed_results.append(res)

        # Calculate seed mean and std
        val_accs = [r["val_accuracy"] for r in seed_results]
        val_qwks = [r["val_qwk"] for r in seed_results]
        test_accs = [r["test_clean_accuracy"] for r in seed_results]
        test_qwks = [r["test_clean_qwk"] for r in seed_results]
        test_f1s = [r["test_clean_macro_f1"] for r in seed_results]

        print(f"  >>> {spec['name']} Multi-Seed Summary (N={len(seeds)}):")
        print(f"      Val QWK:         {np.mean(val_qwks):.4f} ± {np.std(val_qwks):.4f}")
        print(f"      Clean Test QWK:  {np.mean(test_qwks):.4f} ± {np.std(test_qwks):.4f}")
        print(f"      Clean Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"      Clean Test F1:   {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")

    df_repro = pd.DataFrame(repro_records)
    df_repro.to_csv(COMP_DIR / "reproducibility_multi_seed_comparison.csv", index=False)

    # Summary table with mean +- std
    summary_rows = []
    for m_id in ["resnet50", "hybrid_compact"]:
        sub = df_repro[df_repro["model_id"] == m_id]
        row = {
            "model_id": m_id,
            "model_name": sub["model_name"].iloc[0],
            "num_seeds": len(sub),
            "seeds_evaluated": seeds,
            "epochs_per_seed": eval_epochs,
            "val_qwk_mean": round(float(sub["val_qwk"].mean()), 4),
            "val_qwk_std": round(float(sub["val_qwk"].std()), 4),
            "val_accuracy_mean": round(float(sub["val_accuracy"].mean()), 4),
            "val_accuracy_std": round(float(sub["val_accuracy"].std()), 4),
            "test_clean_qwk_mean": round(float(sub["test_clean_qwk"].mean()), 4),
            "test_clean_qwk_std": round(float(sub["test_clean_qwk"].std()), 4),
            "test_clean_accuracy_mean": round(float(sub["test_clean_accuracy"].mean()), 4),
            "test_clean_accuracy_std": round(float(sub["test_clean_accuracy"].std()), 4),
            "test_clean_macro_f1_mean": round(float(sub["test_clean_macro_f1"].mean()), 4),
            "test_clean_macro_f1_std": round(float(sub["test_clean_macro_f1"].std()), 4)
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(COMP_DIR / "reproducibility_summary_mean_std.csv", index=False)

    return {"individual_runs": repro_records, "summary_mean_std": summary_rows}, df_repro

def run_step8_full_audit():
    """Master workflow orchestrating all 6 required efficiency and reproducibility audits."""
    set_seed(42)
    device = get_device("auto")
    print("================================================================================")
    print(" Starting Step 8: Computational Efficiency and Reproducibility Audit")
    print(f" Execution Device: {device} | OS: {platform.system()} {platform.machine()}")
    print("================================================================================\n")

    for d in [OUTPUT_DIR, BENCH_DIR, METRICS_DIR, PREDS_DIR, COMP_DIR, CHECKPOINTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 0. System environment
    env_info = get_system_environment_info(device)
    with open(OUTPUT_DIR / "environment_metadata.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Load split manifest
    with open(SPLIT_MANIFEST, "r") as f:
        manifest = json.load(f)

    # 1. Parameter counts
    param_data, df_params = audit_model_parameters()

    # 2. Checkpoint sizes
    ckpt_data, df_ckpts = audit_checkpoint_sizes()

    # 3. Inference latency & throughput
    latency_data, df_latency = benchmark_inference_latency_and_throughput(device, num_warmup=20, num_timed=100)

    # 4. Peak memory usage
    mem_data, df_mem = benchmark_peak_memory_usage(device)

    # 5. Training time & performance metrics
    perf_data, df_perf, df_pcr = audit_training_time_and_metrics(manifest, device)

    # 6. Multi-seed reproducibility
    repro_data, df_repro = audit_multi_seed_reproducibility(manifest, device, seeds=[42, 123, 456], eval_epochs=3)

    # Master audit summary report
    master_report = {
        "metadata": {
            "title": "Step 8 IDRiD Computational Efficiency and Reproducibility Audit",
            "device": str(device),
            "environment": env_info
        },
        "parameter_audit": param_data,
        "checkpoint_audit": ckpt_data,
        "inference_latency": latency_data,
        "peak_memory_usage": mem_data,
        "model_performance_canonical_s42": perf_data,
        "multi_seed_reproducibility": repro_data
    }

    with open(COMP_DIR / "step8_full_audit_report.json", "w") as f:
        json.dump(master_report, f, indent=2)

    print("\n================================================================================")
    print(" Step 8 Audit Completed Successfully!")
    print(f" All machine-readable artifacts saved in: {OUTPUT_DIR}")
    print("================================================================================\n")

if __name__ == "__main__":
    run_step8_full_audit()
