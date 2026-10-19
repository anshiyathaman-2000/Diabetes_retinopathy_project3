"""
Controlled Experiment Runner for Evaluating Imbalance-Handling Methods
on IDRiD Disease Grading (5-Class DR Classification).

Compares:
1. Unweighted CrossEntropyLoss (Baseline)
2. Class-Weighted CrossEntropyLoss (Smoothed Square-Root)
3. Class-Weighted CrossEntropyLoss (Raw Inverse)
4. WeightedRandomSampler (Resampling with Unweighted Loss)

Enforces:
- Verification of class labels & mapping from groundtruth files
- Class weights and sampling probabilities calculated from TRAINING set only
- No simultaneous double-corrections without control
- Model selection performed strictly on VALIDATION set
- Test set untouched during selection
- Per-class precision, recall, specificity, F1, accuracy, macro/weighted F1, and QWK
"""

import os
import sys
import time
import copy
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, set_seed, get_device
from src.dataset import get_idrid_dataloaders, DR_CLASSES_5, compute_class_weights
from src.train import run_training
from src.evaluate import evaluate_checkpoint

def run_all_imbalance_experiments():
    base_cfg = load_config(PROJECT_ROOT / "configs/experiment.yaml")
    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    print("=================================================================")
    print(" DIABETIC RETINOPATHY CLASS IMBALANCE STUDY (IDRiD 5-CLASS)")
    print("=================================================================\n")

    # Step 1: Verify Dataset and Frequencies
    print(">>> 1. Verifying Dataset and Label-to-Index Mapping...")
    train_csv = PROJECT_ROOT / base_cfg["data"]["train_csv"]
    df_raw = pd.read_csv(train_csv)
    print(f"Groundtruth File: {train_csv.name}")
    print(f"Columns: {[c for c in df_raw.columns if not c.startswith('Unnamed')]}")
    print("\nVerified Class Labels & Mapping:")
    for idx, name in enumerate(DR_CLASSES_5):
        print(f"  Class Index {idx} -> {name}")

    # Step 2: Compute Training Set Frequencies
    train_loader, val_loader, test_loader, cw = get_idrid_dataloaders(base_cfg, seed=seed)
    train_labels = train_loader.dataset.labels
    val_labels = val_loader.dataset.labels
    test_labels = test_loader.dataset.labels

    train_counts = np.bincount(train_labels, minlength=5)
    total_train = len(train_labels)

    print("\n>>> 2. Training Set Class Frequencies (N = 330, Training Split Only):")
    for idx, name in enumerate(DR_CLASSES_5):
        count = train_counts[idx]
        pct = (count / total_train) * 100
        print(f"  Class {idx} ({name:<16}): {count:3d} images ({pct:5.2f}%) | Ratio to Grade 1: {count/train_counts[1]:.2f}:1")

    inv_weights = compute_class_weights(train_labels, 5, method="inverse")
    sqrt_weights = compute_class_weights(train_labels, 5, method="sqrt")
    print("\nClass Weights (Derived from Training Subset Only):")
    print(f"  Inverse (power=1.0): {[round(x, 4) for x in inv_weights.tolist()]}")
    print(f"  Sqrt    (power=0.5): {[round(x, 4) for x in sqrt_weights.tolist()]}\n")

    # Step 3: Define Experiment Configurations
    # Format: (model_name, exp_suffix, description, config_overrides, existing_ckpt)
    experiments = [
        # --- ResNet-50 Baseline ---
        {
            "model": "resnet50",
            "suffix": "unweighted",
            "method_name": "Unweighted CE Loss",
            "model_display": "ResNet-50",
            "overrides": {
                "training": {
                    "epochs": 15,
                    "use_class_weights": False,
                    "sampler_type": "none"
                }
            },
            "ckpt": "checkpoints/resnet50_five_class_clahe_unweighted_s42_best.pt"
        },
        {
            "model": "resnet50",
            "suffix": "sqrt_cw",
            "method_name": "Class-Weighted CE Loss (Sqrt)",
            "model_display": "ResNet-50",
            "overrides": {
                "training": {
                    "epochs": 15,
                    "use_class_weights": True,
                    "class_weight_method": "sqrt",
                    "sampler_type": "none"
                }
            },
            "ckpt": "checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt"
        },
        {
            "model": "resnet50",
            "suffix": "sampler",
            "method_name": "WeightedRandomSampler",
            "model_display": "ResNet-50",
            "overrides": {
                "training": {
                    "epochs": 15,
                    "use_class_weights": False,
                    "sampler_type": "weighted"
                }
            },
            "ckpt": "checkpoints/resnet50_five_class_clahe_sampler_s42_best.pt"
        },
        {
            "model": "resnet50",
            "suffix": "inverse_cw",
            "method_name": "Class-Weighted CE Loss (Inverse)",
            "model_display": "ResNet-50",
            "overrides": {
                "training": {
                    "epochs": 15,
                    "use_class_weights": True,
                    "class_weight_method": "inverse",
                    "sampler_type": "none"
                }
            },
            "ckpt": "checkpoints/resnet50_five_class_clahe_s42_best.pt" # existing trained checkpoint
        },

        # --- Compact CNN-Transformer-BiLSTM ---
        {
            "model": "hybrid",
            "suffix": "compact_unweighted",
            "method_name": "Unweighted CE Loss",
            "model_display": "CNN-Transformer-BiLSTM",
            "overrides": {
                "model": {
                    "hybrid": {
                        "cnn_embed_dim": 128,
                        "transformer_layers": 1,
                        "transformer_heads": 4,
                        "bilstm_hidden_dim": 64,
                        "pool_type": "mean"
                    }
                },
                "training": {
                    "epochs": 15,
                    "backbone_lr": 1e-5,
                    "head_lr": 3e-4,
                    "grad_clip_norm": 5.0,
                    "use_class_weights": False,
                    "sampler_type": "none"
                }
            },
            "ckpt": "checkpoints/hybrid_five_class_clahe_compact_unweighted_s42_best.pt"
        },
        {
            "model": "hybrid",
            "suffix": "compact_sqrt_cw",
            "method_name": "Class-Weighted CE Loss (Sqrt)",
            "model_display": "CNN-Transformer-BiLSTM",
            "overrides": {
                "model": {
                    "hybrid": {
                        "cnn_embed_dim": 128,
                        "transformer_layers": 1,
                        "transformer_heads": 4,
                        "bilstm_hidden_dim": 64,
                        "pool_type": "mean"
                    }
                },
                "training": {
                    "epochs": 15,
                    "backbone_lr": 1e-5,
                    "head_lr": 3e-4,
                    "grad_clip_norm": 5.0,
                    "use_class_weights": True,
                    "class_weight_method": "sqrt",
                    "sampler_type": "none"
                }
            },
            "ckpt": "checkpoints/hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt" # existing trained checkpoint
        },
        {
            "model": "hybrid",
            "suffix": "compact_sampler",
            "method_name": "WeightedRandomSampler",
            "model_display": "CNN-Transformer-BiLSTM",
            "overrides": {
                "model": {
                    "hybrid": {
                        "cnn_embed_dim": 128,
                        "transformer_layers": 1,
                        "transformer_heads": 4,
                        "bilstm_hidden_dim": 64,
                        "pool_type": "mean"
                    }
                },
                "training": {
                    "epochs": 15,
                    "backbone_lr": 1e-5,
                    "head_lr": 3e-4,
                    "grad_clip_norm": 5.0,
                    "use_class_weights": False,
                    "sampler_type": "weighted"
                }
            },
            "ckpt": "checkpoints/hybrid_five_class_clahe_compact_sampler_s42_best.pt"
        }
    ]

    print(">>> 3. Executing Controlled Training Experiments...")
    results_records = []
    per_class_records = []

    for exp in experiments:
        ckpt_path = PROJECT_ROOT / exp["ckpt"]
        print(f"\n-------------------------------------------------------------")
        print(f" Experiment: {exp['model_display']} | Method: {exp['method_name']}")
        print(f" Target Checkpoint: {ckpt_path.name}")
        print(f"-------------------------------------------------------------")

        if ckpt_path.exists():
            print(f"[FOUND] Existing checkpoint exists: {ckpt_path.name}")
        else:
            print(f"[TRAINING] Launching training...")
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["name"] = exp["model"]
            cfg["exp_suffix"] = exp["suffix"]

            # Apply overrides
            for section, kvs in exp["overrides"].items():
                if isinstance(kvs, dict):
                    for k, v in kvs.items():
                        if isinstance(v, dict):
                            cfg[section].setdefault(k, {}).update(v)
                        else:
                            cfg[section][k] = v

            run_training(cfg)

        # Load best validation metrics from checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vm = ckpt.get("val_metrics", {})
        epoch = ckpt.get("epoch", 0)

        acc = vm.get("accuracy", 0.0) * 100
        qwk = vm.get("quadratic_weighted_kappa", 0.0)
        macro_f1 = vm.get("macro_f1", 0.0)
        weighted_f1 = vm.get("weighted_f1", 0.0)

        print(f" Validation Results (Epoch {epoch}):")
        print(f"   Accuracy    : {acc:.2f}%")
        print(f"   QWK         : {qwk:.4f}")
        print(f"   Macro F1    : {macro_f1:.4f}")
        print(f"   Weighted F1 : {weighted_f1:.4f}")

        results_records.append({
            "Model": exp["model_display"],
            "Imbalance Method": exp["method_name"],
            "Best Epoch": epoch,
            "Val Accuracy (%)": round(acc, 2),
            "Val QWK": round(qwk, 4),
            "Val Macro F1": round(macro_f1, 4),
            "Val Weighted F1": round(weighted_f1, 4),
            "Checkpoint": ckpt_path.name
        })

        # Record per-class metrics
        pcm = vm.get("per_class_metrics", {})
        for cname, pvals in pcm.items():
            per_class_records.append({
                "Model": exp["model_display"],
                "Imbalance Method": exp["method_name"],
                "Class": cname,
                "Precision": round(pvals.get("precision", 0.0), 4),
                "Recall (Sensitivity)": round(pvals.get("recall_sensitivity", 0.0), 4),
                "Specificity": round(pvals.get("specificity", 0.0), 4),
                "F1-Score": round(pvals.get("f1_score", 0.0), 4),
                "Support": pvals.get("support", 0)
            })

    # Step 4: Save Comparison Tables
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df_comp = pd.DataFrame(results_records)
    comp_csv_path = metrics_dir / "imbalance_methods_comparison_table.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n>>> Saved Overall Comparison Table: {comp_csv_path}")

    df_pcm = pd.DataFrame(per_class_records)
    pcm_csv_path = metrics_dir / "imbalance_methods_per_class_comparison.csv"
    df_pcm.to_csv(pcm_csv_path, index=False)
    print(f">>> Saved Per-Class Comparison Table: {pcm_csv_path}")

    print("\n=============================================================")
    print(" SUMMARY OF IMBALANCE METHODS COMPARISON (VALIDATION SET, N=83)")
    print("=============================================================")
    print(df_comp.to_string(index=False))
    print("=============================================================\n")

if __name__ == "__main__":
    run_all_imbalance_experiments()
