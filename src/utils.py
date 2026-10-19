"""
Evaluation Metrics, Reproducibility Utilities, Hardware Management, and Plotting.
Supports:
1. Quadratic Weighted Kappa (QWK), Macro/Weighted F1, ROC-AUC, PR-AUC, Specificity/Sensitivity.
2. Device configuration (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback).
3. Metric curve generation, confusion matrix plotting, and ROC/PR curve visualization.
4. Experiment tracking and configuration serialization.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS seed
        torch.mps.manual_seed(seed)

def get_device(requested: str = "auto") -> torch.device:
    """
    Selects hardware execution device based on environment.
    Supports Apple Silicon MPS, NVIDIA CUDA, or CPU fallback.
    """
    if requested.lower() == "cpu":
        return torch.device("cpu")
    elif requested.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested.lower() == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Auto detection
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML experiment configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    task: str = "five_class",
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive research metrics for diabetic retinopathy classification:
    - Accuracy, Macro/Weighted Precision, Recall, F1
    - Quadratic Weighted Kappa (QWK) for ordinal DR severity grades
    - One-vs-rest Specificity and Sensitivity
    - Multiclass / Binary ROC-AUC and Average Precision (PR-AUC)
    - Confusion Matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = 5 if task == "five_class" else 2

    if class_names is None:
        if task == "five_class":
            class_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
        else:
            class_names = ["Non-Referable", "Referable"]

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class specificity
    # For class i: Specificity = TN / (TN + FP)
    specificities = []
    sensitivities = []
    total_samples = len(y_true)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificities.append(spec)
        sensitivities.append(sens)

    # Quadratic Weighted Kappa (crucial for ordinal DR severity)
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    # ROC-AUC and PR-AUC
    roc_auc = None
    pr_auc = None
    weighted_roc_auc = None
    weighted_pr_auc = None
    per_class_auc = {}
    per_class_ap = {}

    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            if num_classes == 2:
                # Binary classification
                probs_pos = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs[:, 0]
                if len(np.unique(y_true)) > 1 and np.isfinite(probs_pos).all():
                    roc_auc = float(roc_auc_score(y_true, probs_pos))
                    pr_auc = float(average_precision_score(y_true, probs_pos))
            else:
                # Multi-class One-vs-Rest AUC and PR-AUC with robust missing-class handling
                valid_aucs = []
                valid_aps = []
                valid_supports = []

                for i in range(num_classes):
                    cname = class_names[i]
                    y_binary = (y_true == i).astype(int)
                    n_pos = int(np.sum(y_binary == 1))
                    n_neg = int(np.sum(y_binary == 0))

                    if n_pos > 0 and n_neg > 0:
                        score_col = y_probs[:, i]
                        if np.isfinite(score_col).all():
                            try:
                                cls_auc = float(roc_auc_score(y_binary, score_col))
                                cls_ap = float(average_precision_score(y_binary, score_col))
                                per_class_auc[cname] = cls_auc
                                per_class_ap[cname] = cls_ap
                                valid_aucs.append(cls_auc)
                                valid_aps.append(cls_ap)
                                valid_supports.append(n_pos)
                            except Exception:
                                per_class_auc[cname] = None
                                per_class_ap[cname] = None
                        else:
                            per_class_auc[cname] = None
                            per_class_ap[cname] = None
                    else:
                        # Safely record unrepresented or all-positive class as None
                        per_class_auc[cname] = None
                        per_class_ap[cname] = None

                # Compute Macro-averaged OvR metrics across valid classes
                if len(valid_aucs) >= 2:
                    roc_auc = float(np.mean(valid_aucs))
                elif len(valid_aucs) == 1:
                    roc_auc = float(valid_aucs[0])

                if len(valid_aps) > 0:
                    pr_auc = float(np.mean(valid_aps))

                # Weighted averages
                total_sup = sum(valid_supports)
                if total_sup > 0:
                    if len(valid_aucs) == len(valid_supports):
                        weighted_roc_auc = float(np.average(valid_aucs, weights=valid_supports))
                    if len(valid_aps) == len(valid_supports):
                        weighted_pr_auc = float(np.average(valid_aps, weights=valid_supports))
        except Exception:
            roc_auc = None
            pr_auc = None
            weighted_roc_auc = None
            weighted_pr_auc = None

    per_class_table = {}
    for i in range(num_classes):
        cname = class_names[i]
        per_class_table[cname] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "recall_sensitivity": float(r_class[i]),
            "specificity": float(specificities[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support[i]),
            "roc_auc": per_class_auc.get(cname),
            "pr_auc": per_class_ap.get(cname)
        }

    results = {
        "accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "quadratic_weighted_kappa": qwk,
        "overall_roc_auc": roc_auc,
        "overall_pr_auc": pr_auc,
        "weighted_roc_auc": weighted_roc_auc,
        "weighted_pr_auc": weighted_pr_auc,
        "per_class_metrics": per_class_table,
        "confusion_matrix": cm.tolist()
    }
    return results

def count_model_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Accurately computes total, trainable, and frozen parameters,
    along with estimated model memory footprint in megabytes (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)

    breakdown = {}
    for name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        breakdown[name] = {
            "total_parameters": child_total,
            "trainable_parameters": child_trainable,
            "size_mb": round((child_total * 4) / (1024 * 1024), 2)
        }

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "model_size_mb": round(model_size_mb, 2),
        "component_breakdown": breakdown
    }

def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Rigorously benchmarks model inference latency and throughput.
    Includes proper device synchronization (MPS/CUDA) and warm-up cycles.
    """
    import time
    import platform

    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    actual_shape = (batch_size, input_shape[1], input_shape[2], input_shape[3])
    dummy_input = torch.randn(*actual_shape, device=device)

    # Hardware detection
    hardware_name = platform.processor() or platform.machine()
    system_str = f"{platform.system()} {platform.release()} ({hardware_name})"
    device_str = str(device)
    if device.type == "cuda":
        device_str = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif device.type == "mps":
        device_str = f"Apple Silicon MPS ({platform.machine()})"
    elif device.type == "cpu":
        device_str = f"CPU ({platform.processor() or platform.machine()})"

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    # Warm-up phase
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            _sync()

    # Timed benchmarking phase
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            _sync()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            _sync()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    if was_training:
        model.train()

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    throughput_fps = float((batch_size * 1000.0) / mean_lat) if mean_lat > 0 else 0.0

    return {
        "hardware": system_str,
        "device": device_str,
        "batch_size": batch_size,
        "input_shape": list(actual_shape),
        "num_warmup_runs": num_warmup,
        "num_timed_runs": num_runs,
        "mean_latency_ms": round(mean_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "throughput_fps": round(throughput_fps, 2)
    }

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
    metric_name: str = "qwk"
) -> None:
    """Plots training and validation loss and performance metric curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_title("Training and Validation Cross-Entropy Loss", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # Metric plot
    val_metric_key = f"val_{metric_name}"
    train_metric_key = f"train_{metric_name}"
    
    if train_metric_key in history:
        axes[1].plot(epochs, history[train_metric_key], 'b-o', label=f'Train {metric_name.upper()}', linewidth=2)
    if val_metric_key in history:
        axes[1].plot(epochs, history[val_metric_key], 'g-^', label=f'Val {metric_name.upper()}', linewidth=2)
    
    axes[1].set_title(f"Model Evaluation Metric: {metric_name.upper()}", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel(metric_name.upper(), fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix"
) -> None:
    """Plots normalized and raw annotated confusion matrix using Matplotlib."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax)

    # Set labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha='left', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold', pad=25)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(
    y_true: List[int],
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path]
) -> None:
    """Plots One-vs-Rest ROC curves for all classes."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_true = np.array(y_true)
    num_classes = len(class_names)

    if num_classes == 2:
        probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs[:, 0]
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"Referable DR (AUC = {auc_val:.3f})", linewidth=2)
    else:
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
                auc_val = roc_auc_score(y_binary, y_probs[:, i])
                ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_val:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for handling severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights well-classified majority samples and forces gradients
    to focus on rare/hard minority samples (such as Mild NPDR Grade 1).
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather target probabilities
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            target_alpha = alpha.gather(dim=0, index=targets)
            loss = loss * target_alpha

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

