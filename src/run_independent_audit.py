"""
Independent Audit and Verification Script for IDRiD DR Research Project.
Executes non-destructive recalculations directly from raw prediction files and original labels.
Outputs: outputs/metrics/independent_audit_verification.json
"""

import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score
)

def get_file_md5_sha256(filepath):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), os.path.getsize(filepath)

def run_independent_audit():
    audit_results = {}
    base_dir = Path(".")
    
    # 1. Ground Truths
    train_csv_path = base_dir / "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv_path = base_dir / "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    
    df_train_gt = pd.read_csv(train_csv_path).dropna(subset=["Image name"])
    df_train_gt["Image name"] = df_train_gt["Image name"].str.strip()
    df_train_gt["Retinopathy grade"] = df_train_gt["Retinopathy grade"].astype(int)
    train_gt_dict = dict(zip(df_train_gt["Image name"], df_train_gt["Retinopathy grade"]))
    
    df_test_gt = pd.read_csv(test_csv_path).dropna(subset=["Image name"])
    df_test_gt["Image name"] = df_test_gt["Image name"].str.strip()
    df_test_gt["Retinopathy grade"] = df_test_gt["Retinopathy grade"].astype(int)
    test_gt_dict = dict(zip(df_test_gt["Image name"], df_test_gt["Retinopathy grade"]))
    
    audit_results["ground_truth_counts"] = {
        "train_samples": len(df_train_gt),
        "test_samples": len(df_test_gt)
    }
    
    # 2. Duplicate Cluster Verification
    duplicate_clusters = [
        {"cluster": 1, "img1": "IDRiD_118", "dir1": "a. Training Set", "img2": "IDRiD_064", "dir2": "b. Testing Set", "g1": 0, "g2": 3},
        {"cluster": 2, "img1": "IDRiD_375", "dir1": "a. Training Set", "img2": "IDRiD_073", "dir2": "a. Training Set", "g1": 2, "g2": 2},
        {"cluster": 3, "img1": "IDRiD_333", "dir1": "a. Training Set", "img2": "IDRiD_045", "dir2": "a. Training Set", "g1": 2, "g2": 2},
        {"cluster": 4, "img1": "IDRiD_028", "dir1": "a. Training Set", "img2": "IDRiD_327", "dir2": "a. Training Set", "g1": 4, "g2": 3},
        {"cluster": 5, "img1": "IDRiD_079", "dir1": "a. Training Set", "img2": "IDRiD_279", "dir2": "a. Training Set", "g1": 1, "g2": 1},
        {"cluster": 6, "img1": "IDRiD_021", "dir1": "a. Training Set", "img2": "IDRiD_391", "dir2": "a. Training Set", "g1": 1, "g2": 2},
    ]
    
    cluster_audits = []
    for c in duplicate_clusters:
        p1 = base_dir / "B. Disease Grading/1. Original Images" / c["dir1"] / f"{c['img1']}.jpg"
        p2 = base_dir / "B. Disease Grading/1. Original Images" / c["dir2"] / f"{c['img2']}.jpg"
        
        md5_1, sha_1, size_1 = get_file_md5_sha256(p1)
        md5_2, sha_2, size_2 = get_file_md5_sha256(p2)
        
        actual_g1 = train_gt_dict.get(c["img1"]) if "Training" in c["dir1"] else test_gt_dict.get(c["img1"])
        actual_g2 = train_gt_dict.get(c["img2"]) if "Training" in c["dir2"] else test_gt_dict.get(c["img2"])
        
        hash_identical = (md5_1 == md5_2) and (sha_1 == sha_2) and (size_1 == size_2)
        labels_conflict = (actual_g1 != actual_g2)
        
        cluster_audits.append({
            "cluster_id": c["cluster"],
            "image_1": c["img1"],
            "image_2": c["img2"],
            "hash_identical": hash_identical,
            "md5": md5_1,
            "sha256": sha_1,
            "size_bytes": size_1,
            "actual_grade_1": actual_g1,
            "actual_grade_2": actual_g2,
            "labels_conflict": labels_conflict
        })
    audit_results["duplicate_clusters_audit"] = cluster_audits

    # 3. Leakage-Aware Split Manifest Verification
    manifest_path = base_dir / "data/splits/leakage_aware_split_manifest.json"
    assert manifest_path.exists(), "Manifest missing!"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_names = set(manifest["train"])
    val_names = set(manifest["val"])
    intersection = train_names.intersection(val_names)
    
    audit_results["manifest_audit"] = {
        "train_count": len(train_names),
        "val_count": len(val_names),
        "total_count": len(train_names) + len(val_names),
        "cross_split_overlap_count": len(intersection),
        "cross_split_overlap_names": list(intersection)
    }

    # 4. Independent Metric Recalculation for All Prediction Files
    eval_configs = [
        {
            "id": "resnet50_five_class_official",
            "model": "ResNet-50 Baseline",
            "task": "five_class",
            "protocol": "official",
            "pred_file": "outputs/predictions/resnet50_five_class_clahe_sqrt_cw_s42_predictions.csv",
            "expected_n": 103
        },
        {
            "id": "resnet50_five_class_dedup",
            "model": "ResNet-50 Baseline",
            "task": "five_class",
            "protocol": "deduplicated",
            "pred_file": "outputs/predictions/resnet50_five_class_clahe_sqrt_cw_s42_dedup_predictions.csv",
            "expected_n": 102
        },
        {
            "id": "hybrid_five_class_official",
            "model": "Corrected Hybrid Baseline",
            "task": "five_class",
            "protocol": "official",
            "pred_file": "outputs/predictions/hybrid_five_class_clahe_compact_sqrt_cw_s42_predictions.csv",
            "expected_n": 103
        },
        {
            "id": "hybrid_five_class_dedup",
            "model": "Corrected Hybrid Baseline",
            "task": "five_class",
            "protocol": "deduplicated",
            "pred_file": "outputs/predictions/hybrid_five_class_clahe_compact_sqrt_cw_s42_dedup_predictions.csv",
            "expected_n": 102
        },
        {
            "id": "resnet50_focal_official",
            "model": "ResNet-50 Focal Loss",
            "task": "five_class",
            "protocol": "official",
            "pred_file": "outputs/predictions/resnet50_five_class_clahe_focal_s42_s42_predictions.csv",
            "expected_n": 103
        },
        {
            "id": "resnet50_focal_dedup",
            "model": "ResNet-50 Focal Loss",
            "task": "five_class",
            "protocol": "deduplicated",
            "pred_file": "outputs/predictions/resnet50_five_class_clahe_focal_s42_s42_dedup_predictions.csv",
            "expected_n": 102
        },
        {
            "id": "resnet50_binary_official",
            "model": "ResNet-50 Binary Referable",
            "task": "binary",
            "protocol": "official",
            "pred_file": "outputs/predictions/resnet50_binary_clahe_binary_referable_s42_s42_predictions.csv",
            "expected_n": 103
        },
        {
            "id": "resnet50_binary_dedup",
            "model": "ResNet-50 Binary Referable",
            "task": "binary",
            "protocol": "deduplicated",
            "pred_file": "outputs/predictions/resnet50_binary_clahe_binary_referable_s42_s42_dedup_predictions.csv",
            "expected_n": 102
        }
    ]

    recalculated_metrics = []
    
    for cfg in eval_configs:
        pf = base_dir / cfg["pred_file"]
        assert pf.exists(), f"Prediction file missing: {pf}"
        df_pred = pd.read_csv(pf)
        
        # Check image count
        actual_n = len(df_pred)
        assert actual_n == cfg["expected_n"], f"Count mismatch in {cfg['id']}: {actual_n} vs {cfg['expected_n']}"
        
        # Verify deduplicated protocol excludes only IDRiD_064
        if cfg["protocol"] == "deduplicated":
            assert "IDRiD_064" not in df_pred["Image name"].values, f"IDRiD_064 found in dedup file: {pf}"
            assert actual_n == 102
        else:
            assert "IDRiD_064" in df_pred["Image name"].values, f"IDRiD_064 missing from official file: {pf}"
            assert actual_n == 103
            
        # Ground truth verification directly against official test CSV
        image_names = df_pred["Image name"].tolist()
        if cfg["task"] == "five_class":
            y_true_verified = [test_gt_dict[name] for name in image_names]
            class_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
            num_classes = 5
        else:
            y_true_verified = [1 if test_gt_dict[name] >= 2 else 0 for name in image_names]
            class_names = ["Non-Referable DR (Grade 0-1)", "Referable DR (Grade 2-4)"]
            num_classes = 2
            
        # Compare ground truth in prediction file with verified ground truth from raw CSV
        y_true_pred_file = df_pred["Ground Truth"].tolist()
        assert y_true_verified == y_true_pred_file, f"Ground truth mismatch in {pf}"
        
        y_pred = df_pred["Predicted Class"].tolist()
        
        # Extract probabilities
        prob_cols = [c for c in df_pred.columns if c.startswith("prob_")]
        if len(prob_cols) == num_classes:
            y_probs = df_pred[prob_cols].values
        else:
            y_probs = None
            
        # Recalculate metrics
        acc = float(accuracy_score(y_true_verified, y_pred))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_verified, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_verified, y_pred, average="weighted", zero_division=0
        )
        qwk = float(cohen_kappa_score(y_true_verified, y_pred, weights="quadratic"))
        cm = confusion_matrix(y_true_verified, y_pred, labels=list(range(num_classes))).tolist()
        
        # Per-class metrics
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
            y_true_verified, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        )
        
        # ROC-AUC calculation
        roc_auc = None
        pr_auc = None
        if y_probs is not None:
            if num_classes == 2:
                prob_pos = y_probs[:, 1]
                roc_auc = float(roc_auc_score(y_true_verified, prob_pos))
                pr_auc = float(average_precision_score(y_true_verified, prob_pos))
            else:
                roc_auc = float(roc_auc_score(y_true_verified, y_probs, multi_class="ovr", average="macro"))
                
        # Specificity / Sensitivity calculation
        specificities = []
        sensitivities = []
        cm_np = np.array(cm)
        tot = len(y_true_verified)
        for i in range(num_classes):
            tp = cm_np[i, i]
            fn = np.sum(cm_np[i, :]) - tp
            fp = np.sum(cm_np[:, i]) - tp
            tn = tot - (tp + fn + fp)
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificities.append(spec)
            sensitivities.append(sens)
            
        record = {
            "id": cfg["id"],
            "model": cfg["model"],
            "task": cfg["task"],
            "protocol": cfg["protocol"],
            "sample_count": actual_n,
            "accuracy": round(acc, 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "quadratic_weighted_kappa": round(qwk, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix": cm,
            "per_class": {}
        }
        
        for i, cname in enumerate(class_names):
            record["per_class"][cname] = {
                "precision": round(float(p_c[i]), 4),
                "recall_sensitivity": round(float(r_c[i]), 4),
                "specificity": round(float(specificities[i]), 4),
                "f1_score": round(float(f1_c[i]), 4),
                "support": int(sup_c[i])
            }
            
        recalculated_metrics.append(record)
        
    audit_results["recalculated_metrics"] = recalculated_metrics

    # Save audit verification report
    out_path = base_dir / "outputs/metrics/independent_audit_verification.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Independent audit report generated: {out_path}")

if __name__ == "__main__":
    run_independent_audit()
