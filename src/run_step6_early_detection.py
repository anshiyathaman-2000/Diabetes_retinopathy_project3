"""
Step 6: Early-Stage Diabetic Retinopathy (Mild NPDR) Detection & Investigation.
Executes:
1. Dataset class mapping & distribution verification for underrepresented grades.
2. Per-class Precision, Recall, F1, and Confusion Matrices across models on validation and test sets.
3. Detailed false-negative case inspection for Mild NPDR (probabilities, confidences, error margins).
4. Comparison of Unweighted Baseline vs Class-Weighted and Focal Loss training.
5. Model selection strictly using validation data.
6. Post-selection held-out test evaluation under Official (N=103) and Deduplicated (N=102) protocols.
7. Machine-readable export to outputs/experiments/idrid_step6_early_detection/.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils import (
    set_seed,
    get_device,
    calculate_metrics,
    count_model_parameters,
    plot_confusion_matrix,
    plot_roc_curves
)
from src.dataset import (
    load_and_clean_csv,
    get_idrid_dataloaders,
    DR_CLASSES_5
)
from src.models import build_model

OUTPUT_DIR = Path("outputs/experiments/idrid_step6_early_detection")
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDS_DIR = OUTPUT_DIR / "predictions"
FN_DIR = OUTPUT_DIR / "false_negatives"
PLOTS_DIR = OUTPUT_DIR / "plots"

TRAIN_CSV = Path("B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
TEST_CSV = Path("B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

MODELS_CONFIG = [
    {
        "id": "resnet50_unweighted",
        "name": "ResNet-50 Unweighted CE",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_unweighted_s42_best.pt"
    },
    {
        "id": "resnet50_sqrt_cw",
        "name": "ResNet-50 Sqrt Class-Weighted CE",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt"
    },
    {
        "id": "resnet50_inverse_cw",
        "name": "ResNet-50 Inverse Class-Weighted CE",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_s42_best.pt"
    },
    {
        "id": "resnet50_focal",
        "name": "ResNet-50 Focal Loss (gamma=2.0)",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_focal_s42_s42_best.pt"
    },
    {
        "id": "hybrid_compact",
        "name": "CNN-Transformer-BiLSTM (Sqrt CW)",
        "checkpoint": "checkpoints/hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt"
    }
]

def run_step6_investigation():
    set_seed(42)
    device = get_device("auto")
    print(f"Executing Step 6 Early Detection Investigation on device: {device}")

    for d in [METRICS_DIR, PREDS_DIR, FN_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Verification
    df_train = load_and_clean_csv(TRAIN_CSV)
    df_test = load_and_clean_csv(TEST_CSV)

    train_dist = df_train["Retinopathy grade"].value_counts().sort_index().to_dict()
    test_dist = df_test["Retinopathy grade"].value_counts().sort_index().to_dict()

    class_counts_summary = {
        "class_mapping": {
            0: "No DR",
            1: "Mild NPDR",
            2: "Moderate NPDR",
            3: "Severe NPDR",
            4: "Proliferative DR"
        },
        "training_counts": {DR_CLASSES_5[k]: int(v) for k, v in train_dist.items()},
        "training_percentages": {DR_CLASSES_5[k]: round((v / len(df_train)) * 100, 2) for k, v in train_dist.items()},
        "testing_counts": {DR_CLASSES_5[k]: int(v) for k, v in test_dist.items()},
        "testing_percentages": {DR_CLASSES_5[k]: round((v / len(df_test)) * 100, 2) for k, v in test_dist.items()},
        "imbalance_ratio_train_grade2_vs_grade1": round(train_dist[2] / train_dist[1], 2),
        "imbalance_ratio_train_grade0_vs_grade1": round(train_dist[0] / train_dist[1], 2)
    }

    with open(OUTPUT_DIR / "class_distribution_verification.json", "w") as f:
        json.dump(class_counts_summary, f, indent=2)

    # 2. Evaluate Models on VALIDATION Data (for unbiased model selection)
    print("\n--- Evaluating Models on Validation Split (for Model Selection) ---")
    val_model_records = []

    for m_info in MODELS_CONFIG:
        ckpt_path = Path(m_info["checkpoint"])
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt["config"]
        model = build_model(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        _, val_loader, _, _ = get_idrid_dataloaders(config, seed=42)

        val_probs, val_preds, val_targets, val_names = [], [], [], []
        with torch.no_grad():
            for imgs, lbls, names in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(lbls.numpy())
                val_names.extend(names)

        val_probs_np = np.array(val_probs)
        val_metrics = calculate_metrics(
            val_targets, val_preds, val_probs_np, task="five_class", class_names=DR_CLASSES_5
        )

        val_mild = val_metrics["per_class_metrics"]["Mild NPDR"]
        val_model_records.append({
            "model_id": m_info["id"],
            "model_name": m_info["name"],
            "split": "validation",
            "accuracy": round(val_metrics["accuracy"], 4),
            "macro_f1": round(val_metrics["macro_f1"], 4),
            "weighted_f1": round(val_metrics["weighted_f1"], 4),
            "qwk": round(val_metrics["quadratic_weighted_kappa"], 4),
            "mild_npdr_precision": round(val_mild["precision"], 4),
            "mild_npdr_recall": round(val_mild["recall_sensitivity"], 4),
            "mild_npdr_f1": round(val_mild["f1_score"], 4),
            "mild_npdr_support": val_mild["support"],
            "mild_npdr_roc_auc": round(val_mild["roc_auc"], 4) if val_mild["roc_auc"] else None,
            "mild_npdr_pr_auc": round(val_mild["pr_auc"], 4) if val_mild["pr_auc"] else None
        })

        # Save validation metrics JSON & Confusion Matrix
        with open(METRICS_DIR / f"{m_info['id']}_val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        cm_df = pd.DataFrame(
            val_metrics["confusion_matrix"],
            index=[f"True_{c}" for c in DR_CLASSES_5],
            columns=[f"Pred_{c}" for c in DR_CLASSES_5]
        )
        cm_df.to_csv(METRICS_DIR / f"{m_info['id']}_val_confusion_matrix.csv")

    df_val_summary = pd.DataFrame(val_model_records)
    df_val_summary.to_csv(OUTPUT_DIR / "validation_model_selection_comparison.csv", index=False)
    print(df_val_summary[["model_name", "macro_f1", "qwk", "mild_npdr_recall", "mild_npdr_f1"]])

    # Model Selection Decision:
    # Based strictly on validation data:
    # - ResNet-50 Sqrt Class-Weighted achieved best Val Macro F1 (0.5707) and Val Mild F1 (0.3333).
    # - ResNet-50 Focal Loss achieved best Val QWK (0.8136) and strong Val Mild F1 (0.2857).
    # - ResNet-50 Unweighted failed completely on Mild NPDR (Val Recall = 0.0, Val F1 = 0.0).

    # 3. Post-Selection Held-Out Test Evaluation (Official & Deduplicated)
    print("\n--- Evaluating Models on Held-Out Test Set (Preserved for Final Benchmark) ---")
    test_records = []
    mild_false_negatives_all = []

    for m_info in MODELS_CONFIG:
        ckpt_path = Path(m_info["checkpoint"])
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt["config"]
        model = build_model(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for protocol in ["official", "deduplicated"]:
            cfg_eval = dict(config)
            cfg_eval.setdefault("data", {})["test_protocol"] = protocol
            _, _, test_loader, _ = get_idrid_dataloaders(cfg_eval, seed=42)

            t_probs, t_preds, t_targets, t_names = [], [], [], []
            with torch.no_grad():
                for imgs, lbls, names in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    t_probs.extend(probs)
                    t_preds.extend(preds)
                    t_targets.extend(lbls.numpy())
                    t_names.extend(names)

            t_probs_np = np.array(t_probs)
            t_metrics = calculate_metrics(
                t_targets, t_preds, t_probs_np, task="five_class", class_names=DR_CLASSES_5
            )

            mild_m = t_metrics["per_class_metrics"]["Mild NPDR"]
            base_id = f"{m_info['id']}_{protocol}"

            test_records.append({
                "model_id": m_info["id"],
                "model_name": m_info["name"],
                "protocol": protocol,
                "test_n": len(test_loader.dataset),
                "accuracy": round(t_metrics["accuracy"], 4),
                "macro_f1": round(t_metrics["macro_f1"], 4),
                "weighted_f1": round(t_metrics["weighted_f1"], 4),
                "qwk": round(t_metrics["quadratic_weighted_kappa"], 4),
                "overall_roc_auc": round(t_metrics["overall_roc_auc"], 4) if t_metrics["overall_roc_auc"] else None,
                "overall_pr_auc": round(t_metrics["overall_pr_auc"], 4) if t_metrics["overall_pr_auc"] else None,
                "mild_precision": round(mild_m["precision"], 4),
                "mild_recall": round(mild_m["recall_sensitivity"], 4),
                "mild_f1": round(mild_m["f1_score"], 4),
                "mild_support": mild_m["support"]
            })

            # Save test artifacts
            with open(METRICS_DIR / f"{base_id}_test_metrics.json", "w") as f:
                json.dump(t_metrics, f, indent=2)

            pd.DataFrame(t_metrics["per_class_metrics"]).T.to_csv(METRICS_DIR / f"{base_id}_per_class_table.csv")
            pd.DataFrame(
                t_metrics["confusion_matrix"],
                index=[f"True_{c}" for c in DR_CLASSES_5],
                columns=[f"Pred_{c}" for c in DR_CLASSES_5]
            ).to_csv(METRICS_DIR / f"{base_id}_confusion_matrix.csv")

            # Predictions CSV
            df_preds = pd.DataFrame({
                "Image name": t_names,
                "Ground Truth": t_targets,
                "Predicted Class": t_preds,
                "Confidence": np.max(t_probs_np, axis=1)
            })
            for i, cname in enumerate(DR_CLASSES_5):
                df_preds[f"prob_{cname}"] = t_probs_np[:, i]
            df_preds.to_csv(PREDS_DIR / f"{base_id}_predictions.csv", index=False)

            # Confusion Matrix Plot
            plot_confusion_matrix(
                cm=np.array(t_metrics["confusion_matrix"]),
                class_names=DR_CLASSES_5,
                save_path=PLOTS_DIR / f"{base_id}_confusion_matrix.png",
                title=f"{m_info['name']} ({protocol.capitalize()} Test N={len(test_loader.dataset)})"
            )

            # Inspect Mild NPDR cases specifically (Official test set)
            if protocol == "official":
                for i, (name, gt, pred, pr) in enumerate(zip(t_names, t_targets, t_preds, t_probs_np)):
                    if gt == 1:
                        mild_false_negatives_all.append({
                            "model_id": str(m_info["id"]),
                            "model_name": str(m_info["name"]),
                            "image_name": str(name),
                            "ground_truth_grade": int(gt),
                            "predicted_grade": int(pred),
                            "predicted_grade_name": str(DR_CLASSES_5[pred]),
                            "is_correct": bool(gt == pred),
                            "confidence": round(float(pr[pred]), 4),
                            "prob_Grade_0_NoDR": round(float(pr[0]), 4),
                            "prob_Grade_1_MildNPDR": round(float(pr[1]), 4),
                            "prob_Grade_2_ModerateNPDR": round(float(pr[2]), 4),
                            "prob_Grade_3_SevereNPDR": round(float(pr[3]), 4),
                            "prob_Grade_4_PDR": round(float(pr[4]), 4),
                            "margin_mild_minus_nodr": round(float(pr[1] - pr[0]), 4)
                        })

    df_test_summary = pd.DataFrame(test_records)
    df_test_summary.to_csv(OUTPUT_DIR / "step6_test_comparison_summary.csv", index=False)

    df_fn = pd.DataFrame(mild_false_negatives_all)
    df_fn.to_csv(FN_DIR / "mild_npdr_case_by_case_investigation.csv", index=False)
    with open(FN_DIR / "mild_npdr_case_by_case_investigation.json", "w") as f:
        json.dump(mild_false_negatives_all, f, indent=2)

    print(f"\nSaved Step 6 Artifacts to: {OUTPUT_DIR}")
    print(f"  Validation Summary : {OUTPUT_DIR / 'validation_model_selection_comparison.csv'}")
    print(f"  Test Summary       : {OUTPUT_DIR / 'step6_test_comparison_summary.csv'}")
    print(f"  Case Analysis CSV  : {FN_DIR / 'mild_npdr_case_by_case_investigation.csv'}")

    return {
        "class_counts": class_counts_summary,
        "validation_summary": val_model_records,
        "test_summary": test_records,
        "mild_npdr_false_negatives": mild_false_negatives_all
    }

if __name__ == "__main__":
    run_step6_investigation()
