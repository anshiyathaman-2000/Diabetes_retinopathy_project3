"""
Comprehensive Step 3 Evaluation and Uncertainty Quantification Script.
Evaluates all validated model checkpoints under both Official (N=103) and Deduplicated (N=102) protocols.
Computes 95% Bootstrap Confidence Intervals (B=1000) for rigorous scientific reporting.
Saves all outputs to: outputs/experiments/idrid_step3_final/
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, roc_auc_score

# Add root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_checkpoint

OUTPUT_DIR = Path("outputs/experiments/idrid_step3_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [
    {
        "id": "resnet50_unweighted_baseline",
        "name": "ResNet-50 Baseline (Unweighted)",
        "role": "Initial CNN Baseline",
        "path": "checkpoints/resnet50_five_class_clahe_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "resnet50_sqrt_cw_baseline",
        "name": "ResNet-50 (Sqrt CW)",
        "role": "Step 1 Enhanced CNN",
        "path": "checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "hybrid_initial_v1",
        "name": "Hybrid Initial v1 (Overparameterized)",
        "role": "Initial Hybrid Baseline",
        "path": "checkpoints/hybrid_five_class_clahe_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "cnn_transformer_ablation",
        "name": "CNN-Transformer (No BiLSTM)",
        "role": "Architectural Ablation 1",
        "path": "checkpoints/cnn_transformer_five_class_clahe_ablation_transformer_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "cnn_bilstm_ablation",
        "name": "CNN-BiLSTM (No Transformer)",
        "role": "Architectural Ablation 2",
        "path": "checkpoints/cnn_bilstm_five_class_clahe_ablation_bilstm_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "hybrid_compact_sqrt_cw",
        "name": "Corrected Compact Hybrid (v2)",
        "role": "Corrected Hybrid Proposed",
        "path": "checkpoints/hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "resnet50_focal_loss",
        "name": "ResNet-50 Focal Loss (gamma=2.0)",
        "role": "Step 2 Focal Loss Model",
        "path": "checkpoints/resnet50_five_class_clahe_focal_s42_s42_best.pt",
        "task": "five_class"
    },
    {
        "id": "resnet50_binary_referable",
        "name": "ResNet-50 Binary Referable DR",
        "role": "Clinical Triage Task",
        "path": "checkpoints/resnet50_binary_clahe_binary_referable_s42_s42_best.pt",
        "task": "binary"
    }
]

def bootstrap_metric_cis(y_true, y_pred, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    accs, qwks, f1s = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]
        
        accs.append(accuracy_score(yt_sample, yp_sample))
        qwks.append(cohen_kappa_score(yt_sample, yp_sample, weights="quadratic"))
        _, _, f1, _ = precision_recall_fscore_support(yt_sample, yp_sample, average="macro", zero_division=0)
        f1s.append(f1)
        
    return {
        "acc_ci": (round(float(np.percentile(accs, 2.5) * 100), 2), round(float(np.percentile(accs, 97.5) * 100), 2)),
        "qwk_ci": (round(float(np.percentile(qwks, 2.5)), 4), round(float(np.percentile(qwks, 97.5)), 4)),
        "macro_f1_ci": (round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4))
    }

def run_all_evaluations():
    comparison_records = []
    experiment_configs = []
    
    for item in CHECKPOINTS:
        ckpt_path = Path(item["path"])
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue
            
        print(f"\n=======================================================")
        print(f" Evaluating: {item['name']} ({item['role']})")
        print(f" Checkpoint: {ckpt_path}")
        print(f"=======================================================")
        
        # 1. Official Evaluation (N=103)
        metrics_off = evaluate_checkpoint(
            checkpoint_path=str(ckpt_path),
            output_dir=str(OUTPUT_DIR),
            test_protocol="official"
        )
        
        # 2. Deduplicated Clean Evaluation (N=102)
        metrics_dedup = evaluate_checkpoint(
            checkpoint_path=str(ckpt_path),
            output_dir=str(OUTPUT_DIR),
            test_protocol="deduplicated"
        )
        
        # Compute Bootstrap CIs on Official Test
        base_stem = ckpt_path.stem.replace("_best", "")
        pred_off_path = OUTPUT_DIR / "predictions" / f"{base_stem}_predictions.csv"
        df_pred_off = pd.read_csv(pred_off_path)
        cis = bootstrap_metric_cis(df_pred_off["Ground Truth"], df_pred_off["Predicted Class"])
        
        # Per-class metrics
        pcm_off = metrics_off["per_class_metrics"]
        pcm_dedup = metrics_dedup["per_class_metrics"]
        
        if item["task"] == "five_class":
            mild_off = pcm_off["Mild NPDR"]
            mild_dedup = pcm_dedup["Mild NPDR"]
            mild_r_off, mild_f1_off = mild_off["recall_sensitivity"], mild_off["f1_score"]
            mild_r_dedup, mild_f1_dedup = mild_dedup["recall_sensitivity"], mild_dedup["f1_score"]
        else:
            mild_r_off, mild_f1_off = None, None
            mild_r_dedup, mild_f1_dedup = None, None
            
        comparison_records.append({
            "Model_ID": item["id"],
            "Model_Name": item["name"],
            "Role": item["role"],
            "Task": item["task"],
            "Parameters": metrics_off["total_parameters"],
            "Official_Accuracy": round(metrics_off["accuracy"] * 100, 2),
            "Official_Acc_95CI": f"[{cis['acc_ci'][0]}, {cis['acc_ci'][1]}]",
            "Official_QWK": round(metrics_off["quadratic_weighted_kappa"], 4),
            "Official_QWK_95CI": f"[{cis['qwk_ci'][0]}, {cis['qwk_ci'][1]}]",
            "Official_Macro_F1": round(metrics_off["macro_f1"], 4),
            "Official_F1_95CI": f"[{cis['macro_f1_ci'][0]}, {cis['macro_f1_ci'][1]}]",
            "Official_OvR_ROC_AUC": round(metrics_off["overall_roc_auc"], 4) if metrics_off["overall_roc_auc"] else "N/A",
            "Official_Mild_NPDR_Recall": round(mild_r_off, 4) if mild_r_off is not None else "N/A",
            "Official_Mild_NPDR_F1": round(mild_f1_off, 4) if mild_f1_off is not None else "N/A",
            "Deduplicated_Accuracy": round(metrics_dedup["accuracy"] * 100, 2),
            "Deduplicated_QWK": round(metrics_dedup["quadratic_weighted_kappa"], 4),
            "Deduplicated_Macro_F1": round(metrics_dedup["macro_f1"], 4),
            "Deduplicated_Mild_NPDR_Recall": round(mild_r_dedup, 4) if mild_r_dedup is not None else "N/A",
            "Deduplicated_Mild_NPDR_F1": round(mild_f1_dedup, 4) if mild_f1_dedup is not None else "N/A",
        })
        
        experiment_configs.append({
            "model_id": item["id"],
            "name": item["name"],
            "role": item["role"],
            "task": item["task"],
            "checkpoint_path": str(ckpt_path),
            "total_parameters": metrics_off["total_parameters"],
            "official_test_samples": 103,
            "deduplicated_test_samples": 102
        })
        
    df_comp = pd.DataFrame(comparison_records)
    csv_out = OUTPUT_DIR / "final_controlled_experiment_comparison.csv"
    df_comp.to_csv(csv_out, index=False)
    print(f"\nSaved comparison table to: {csv_out}")
    
    json_cfg_out = OUTPUT_DIR / "experiment_configurations.json"
    with open(json_cfg_out, "w") as f:
        json.dump(experiment_configs, f, indent=2)
    print(f"Saved experiment configs to: {json_cfg_out}")

if __name__ == "__main__":
    run_all_evaluations()
