"""
Evaluation and Research Benchmark Pipeline for IDRiD Test Set.
Computes:
1. Accuracy, Macro F1, Weighted F1, Quadratic Weighted Kappa (QWK).
2. Per-class Precision, Sensitivity, Specificity, F1-score, Support.
3. Multiclass & Binary ROC-AUC and PR-AUC.
4. Generates publication-ready Confusion Matrices, ROC curves, and CSV tables.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils import (
    set_seed,
    get_device,
    load_config,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    count_model_parameters,
    measure_inference_latency
)
from src.dataset import (
    get_idrid_dataloaders,
    DR_CLASSES_5,
    DR_CLASSES_BINARY
)
from src.models import build_model

def evaluate_checkpoint(
    checkpoint_path: str,
    test_loader = None,
    output_dir: Optional[str] = None,
    test_protocol: str = "official"
) -> Dict[str, Any]:
    """Evaluates a saved checkpoint on the held-out untouched IDRiD test set."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    if test_protocol in ["deduplicated", "clean"]:
        config.setdefault("data", {})["test_protocol"] = "deduplicated"
    task = config["task"]["type"]
    num_classes = config["task"]["num_classes"]
    class_names = DR_CLASSES_5 if task == "five_class" else DR_CLASSES_BINARY

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config["hardware"].get("device", "auto"))

    # Prepare model
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Model parameters
    param_info = count_model_parameters(model)
    total_params = param_info["total_parameters"]

    # Benchmark inference latency with warmup and synchronization
    img_size = config["preprocessing"]["image_size"]
    print(f"Benchmarking inference latency on {device} (warmup=20, runs=100)...")
    latency_bs1 = measure_inference_latency(
        model, input_shape=(1, 3, img_size, img_size), batch_size=1, device=device, num_warmup=20, num_runs=100
    )
    latency_bs16 = measure_inference_latency(
        model, input_shape=(1, 3, img_size, img_size), batch_size=16, device=device, num_warmup=10, num_runs=50
    )

    # Prepare dataloader if not provided
    if test_loader is None:
        _, _, test_loader, _ = get_idrid_dataloaders(config, seed=seed)

    print(f"\nEvaluating Model Checkpoint: {ckpt_path.name}")
    print(f"Task: {task} | Device: {device} | Test samples: {len(test_loader.dataset)}")

    all_preds = []
    all_targets = []
    all_probs = []
    all_names = []

    start_eval = time.time()
    with torch.no_grad():
        for images, labels, names in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_names.extend(names)

    total_eval_time = time.time() - start_eval
    avg_inference_time_ms = (total_eval_time / len(test_loader.dataset)) * 1000

    y_probs_np = np.array(all_probs)
    metrics = calculate_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=y_probs_np,
        task=task,
        class_names=class_names
    )

    metrics["parameter_counts"] = param_info
    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = param_info["trainable_parameters"]
    metrics["frozen_parameters"] = param_info["frozen_parameters"]
    metrics["model_size_mb"] = param_info["model_size_mb"]
    metrics["latency_benchmark_bs1"] = latency_bs1
    metrics["latency_benchmark_bs16"] = latency_bs16
    metrics["total_eval_time_seconds"] = total_eval_time
    metrics["avg_inference_time_per_image_ms"] = avg_inference_time_ms
    metrics["model_name"] = config["model"]["name"]
    metrics["preprocessing"] = config["preprocessing"]["method"]
    metrics["seed"] = seed
    metrics["task"] = task
    metrics["test_protocol"] = test_protocol

    # Display clean evaluation summary
    print(f"\n=======================================================")
    print(f" Test Evaluation Results ({config['model']['name'].upper()})")
    print(f"=======================================================")
    print(f" Accuracy                 : {metrics['accuracy']*100:.2f}%")
    print(f" Macro F1-Score           : {metrics['macro_f1']:.4f}")
    print(f" Weighted F1-Score        : {metrics['weighted_f1']:.4f}")
    print(f" Quadratic Weighted Kappa : {metrics['quadratic_weighted_kappa']:.4f}")
    if metrics['overall_roc_auc'] is not None:
        print(f" Multiclass ROC-AUC (OvR) : {metrics['overall_roc_auc']:.4f}")
    if metrics['overall_pr_auc'] is not None:
        print(f" Multiclass PR-AUC (OvR)  : {metrics['overall_pr_auc']:.4f}")
    print(f" Latency (Batch Size 1)   : {latency_bs1['mean_latency_ms']:.2f} \u00b1 {latency_bs1['std_latency_ms']:.2f} ms")
    print(f" Throughput (Batch Size 1): {latency_bs1['throughput_fps']:.1f} fps")
    print(f" Total Parameters         : {total_params:,} ({param_info['model_size_mb']:.2f} MB)")
    print(f"=======================================================\n")

    # Output directory
    if output_dir is None:
        output_metrics_dir = Path(config["paths"]["metrics_dir"])
        output_plots_dir = Path(config["paths"]["plots_dir"])
        output_preds_dir = Path(config["paths"]["predictions_dir"])
    else:
        output_metrics_dir = Path(output_dir) / "metrics"
        output_plots_dir = Path(output_dir) / "plots"
        output_preds_dir = Path(output_dir) / "predictions"

    for p in [output_metrics_dir, output_plots_dir, output_preds_dir]:
        p.mkdir(parents=True, exist_ok=True)

    base_name = ckpt_path.stem.replace("_best", "")
    if test_protocol in ["deduplicated", "clean"]:
        base_name = f"{base_name}_dedup"

    # Save metrics JSON
    json_path = output_metrics_dir / f"{base_name}_test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class metrics table to CSV
    df_per_class = pd.DataFrame.from_dict(metrics["per_class_metrics"], orient="index")
    per_class_csv_path = output_metrics_dir / f"{base_name}_per_class_table.csv"
    df_per_class.to_csv(per_class_csv_path, index_label="Class")

    # Save Confusion Matrix as CSV
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )
    cm_csv_path = output_metrics_dir / f"{base_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Save individual predictions to CSV
    df_preds = pd.DataFrame({
        "Image name": all_names,
        "Ground Truth": all_targets,
        "Predicted Class": all_preds,
        "Confidence": np.max(y_probs_np, axis=1)
    })
    for i, cname in enumerate(class_names):
        df_preds[f"prob_{cname}"] = y_probs_np[:, i]
    preds_csv_path = output_preds_dir / f"{base_name}_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)

    # Plot Confusion Matrix
    cm_plot_path = output_plots_dir / f"{base_name}_confusion_matrix.png"
    plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        class_names=class_names,
        save_path=cm_plot_path,
        title=f"Test Confusion Matrix: {config['model']['name'].upper()}"
    )

    # Plot ROC curves
    roc_plot_path = output_plots_dir / f"{base_name}_roc_curves.png"
    plot_roc_curves(
        y_true=all_targets,
        y_probs=y_probs_np,
        class_names=class_names,
        save_path=roc_plot_path
    )

    print(f"Saved evaluation artifacts:")
    print(f"  - Metrics JSON    : {json_path}")
    print(f"  - Per-Class CSV   : {per_class_csv_path}")
    print(f"  - Predictions CSV : {preds_csv_path}")
    print(f"  - Confusion Matrix: {cm_plot_path}")
    print(f"  - ROC Curves      : {roc_plot_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Checkpoint on IDRiD Test Set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation artifacts")
    parser.add_argument("--test_protocol", type=str, default="official", help="Test protocol ('official' for N=103 or 'deduplicated' for N=102)")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, output_dir=args.output_dir, test_protocol=args.test_protocol)
