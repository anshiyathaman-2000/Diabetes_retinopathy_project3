"""
Controlled Model Evaluation Runner for IDRiD Diabetic Retinopathy.

Evaluates checkpoints under two strictly separated protocols:
1. Official Benchmark Protocol (N=103): All official test images with original ground-truth labels.
2. Sensitivity Analysis Protocol (N=102): Excludes verified cross-split duplicate IDRiD_064.

Generates:
- Separate JSON metric files for each model under each protocol.
- Separate CSV prediction files with per-sample predictions and class probabilities.
- Comparative CSV tables with Accuracy, QWK, Macro-F1, Weighted-F1, and Deltas.
"""

import sys
from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import set_seed, get_device, calculate_metrics
from src.dataset import get_idrid_dataloaders, DR_CLASSES_5
from src.models import build_model


def evaluate_model_on_protocol(checkpoint_path: Path, protocol: str = "official") -> dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    cfg.setdefault("data", {})["test_protocol"] = protocol
    device = get_device(cfg["hardware"].get("device", "auto"))

    set_seed(cfg.get("seed", 42))

    # Build model and load weights
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Get test dataloader
    _, _, test_loader, _ = get_idrid_dataloaders(cfg, seed=cfg.get("seed", 42))
    expected_count = 103 if protocol == "official" else 102
    assert len(test_loader.dataset) == expected_count, (
        f"Expected {expected_count} test samples for protocol '{protocol}', got {len(test_loader.dataset)}"
    )

    all_preds = []
    all_targets = []
    all_probs = []
    all_names = []

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

    y_probs_np = np.array(all_probs)
    metrics = calculate_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=y_probs_np,
        task="five_class",
        class_names=DR_CLASSES_5
    )

    metrics["test_protocol"] = protocol
    metrics["test_samples"] = len(test_loader.dataset)
    metrics["checkpoint_epoch"] = ckpt.get("epoch")
    metrics["best_val_qwk"] = ckpt.get("best_metric_val")
    metrics["model_name"] = cfg["model"]["name"]
    metrics["checkpoint_path"] = str(ckpt_path)

    # Predictions DataFrame
    df_preds = pd.DataFrame({
        "Image name": all_names,
        "Ground Truth": all_targets,
        "Predicted Class": all_preds,
        "Confidence": np.max(y_probs_np, axis=1)
    })
    for i, cname in enumerate(DR_CLASSES_5):
        df_preds[f"prob_{cname}"] = y_probs_np[:, i]

    return metrics, df_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Controlled Model Checkpoints.")
    parser.add_argument("--resnet_ckpt", type=str, required=True, help="Path to ResNet-50 checkpoint")
    parser.add_argument("--hybrid_ckpt", type=str, required=True, help="Path to Compact Hybrid checkpoint")
    parser.add_argument("--output_tag", type=str, default="controlled_leakage_free", help="Output identifier tag")
    args = parser.parse_args()

    models = [
        ("ResNet-50", Path(args.resnet_ckpt)),
        ("Compact Hybrid", Path(args.hybrid_ckpt)),
    ]

    metrics_dir = ROOT / "outputs/metrics"
    preds_dir = ROOT / "outputs/predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("\n" + "=" * 90)
    print("RUNNING DUAL-PROTOCOL CONTROLLED EVALUATION")
    print("=" * 90)

    for model_name, ckpt_path in models:
        for protocol in ["official", "deduplicated"]:
            proto_label = "Official (N=103)" if protocol == "official" else "Sensitivity (N=102, excl IDRiD_064)"
            print(f"\nEvaluating {model_name} on {proto_label}...")

            metrics, df_preds = evaluate_model_on_protocol(ckpt_path, protocol=protocol)

            # Unique file paths
            clean_name = model_name.lower().replace(" ", "_").replace("-", "")
            metrics_file = metrics_dir / f"{clean_name}_{args.output_tag}_{protocol}_test_metrics.json"
            preds_file = preds_dir / f"{clean_name}_{args.output_tag}_{protocol}_predictions.csv"

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            df_preds.to_csv(preds_file, index=False)

            summary_rows.append({
                "Model": model_name,
                "Protocol": "Official Benchmark" if protocol == "official" else "Sensitivity Analysis",
                "Test_Samples": metrics["test_samples"],
                "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
                "QWK": round(metrics["quadratic_weighted_kappa"], 4),
                "Macro_F1": round(metrics["macro_f1"], 4),
                "Weighted_F1": round(metrics["weighted_f1"], 4),
                "Multiclass_ROC_AUC": round(metrics["overall_roc_auc"], 4) if metrics.get("overall_roc_auc") is not None else "N/A",
                "Best_Val_QWK": round(metrics["best_val_qwk"], 4),
                "Best_Epoch": metrics["checkpoint_epoch"],
                "Metrics_JSON": str(metrics_file.relative_to(ROOT)),
                "Predictions_CSV": str(preds_file.relative_to(ROOT)),
            })

            print(f"  Accuracy : {metrics['accuracy']*100:.2f}% | QWK: {metrics['quadratic_weighted_kappa']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
            print(f"  Saved JSON: {metrics_file.name}")
            print(f"  Saved CSV : {preds_file.name}")

    # Create Comparison DataFrame
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = metrics_dir / f"controlled_comparison_{args.output_tag}.csv"
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90)
    print("CONTROLLED EVALUATION COMPLETED")
    print(f"Summary table saved to: {summary_csv}")
    print("=" * 90)
    print(df_summary[["Model", "Protocol", "Test_Samples", "Accuracy (%)", "QWK", "Macro_F1", "Weighted_F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
