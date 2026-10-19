"""
Script to compile comprehensive empirical comparison CSV tables for IDRiD deliverables:
1. baseline_vs_improved_five_class_comparison.csv
2. mild_npdr_improvement_comparison.csv
3. deduplicated_vs_official_test_comparison.csv
"""

import json
from pathlib import Path
import pandas as pd

METRICS_DIR = Path("outputs/metrics")

def load_json(name):
    path = METRICS_DIR / name
    if not path.exists():
        print(f"Warning: {path} does not exist.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def generate_tables():
    # Load JSON files
    m_r50_sqrt_off = load_json("resnet50_five_class_clahe_sqrt_cw_s42_test_metrics.json")
    m_r50_sqrt_dedup = load_json("resnet50_five_class_clahe_sqrt_cw_s42_dedup_test_metrics.json")
    m_hyb_sqrt_off = load_json("hybrid_five_class_clahe_compact_sqrt_cw_s42_test_metrics.json")
    m_hyb_sqrt_dedup = load_json("hybrid_five_class_clahe_compact_sqrt_cw_s42_dedup_test_metrics.json")
    m_r50_focal_off = load_json("resnet50_five_class_clahe_focal_s42_s42_test_metrics.json")
    m_r50_focal_dedup = load_json("resnet50_five_class_clahe_focal_s42_s42_dedup_test_metrics.json")

    # Table 1: Baseline vs Improved Five-Class Comparison Table
    records_1 = []
    configs = [
        ("ResNet-50 Baseline", "Sqrt Class Weights", "CrossEntropy", "Official Benchmark", 103, m_r50_sqrt_off),
        ("ResNet-50 Baseline", "Sqrt Class Weights", "CrossEntropy", "Deduplicated Clean", 102, m_r50_sqrt_dedup),
        ("Corrected Hybrid", "Sqrt Class Weights", "CrossEntropy", "Official Benchmark", 103, m_hyb_sqrt_off),
        ("Corrected Hybrid", "Sqrt Class Weights", "CrossEntropy", "Deduplicated Clean", 102, m_hyb_sqrt_dedup),
        ("ResNet-50 Improved", "Sqrt CW + Focal Loss (gamma=2.0)", "Focal Loss", "Official Benchmark", 103, m_r50_focal_off),
        ("ResNet-50 Improved", "Sqrt CW + Focal Loss (gamma=2.0)", "Focal Loss", "Deduplicated Clean", 102, m_r50_focal_dedup),
    ]

    for model, imb, loss, proto, n_samples, m in configs:
        if m is None:
            continue
        mild_m = m["per_class_metrics"]["Mild NPDR"]
        records_1.append({
            "Model": model,
            "Imbalance_Strategy": imb,
            "Loss_Function": loss,
            "Evaluation_Protocol": proto,
            "Test_Samples": n_samples,
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Macro_F1": round(m["macro_f1"], 4),
            "Weighted_F1": round(m["weighted_f1"], 4),
            "QWK": round(m["quadratic_weighted_kappa"], 4),
            "OvR_ROC_AUC": round(m["overall_roc_auc"], 4) if m["overall_roc_auc"] else "N/A",
            "Mild_NPDR_Precision": round(mild_m["precision"], 4),
            "Mild_NPDR_Recall": round(mild_m["recall_sensitivity"], 4),
            "Mild_NPDR_F1": round(mild_m["f1_score"], 4),
            "Mild_NPDR_Support": mild_m["support"]
        })

    df1 = pd.DataFrame(records_1)
    df1_path = METRICS_DIR / "baseline_vs_improved_five_class_comparison.csv"
    df1.to_csv(df1_path, index=False)
    print(f"Saved: {df1_path}")

    # Table 2: Mild NPDR and Per-Class Breakdown Table
    records_2 = []
    for model, imb, loss, proto, n_samples, m in configs:
        if m is None:
            continue
        pcm = m["per_class_metrics"]
        records_2.append({
            "Model": f"{model} ({proto})",
            "Grade_0_NoDR_F1": round(pcm["No DR"]["f1_score"], 4),
            "Grade_1_Mild_NPDR_Precision": round(pcm["Mild NPDR"]["precision"], 4),
            "Grade_1_Mild_NPDR_Recall": round(pcm["Mild NPDR"]["recall_sensitivity"], 4),
            "Grade_1_Mild_NPDR_F1": round(pcm["Mild NPDR"]["f1_score"], 4),
            "Grade_1_Mild_NPDR_Support": pcm["Mild NPDR"]["support"],
            "Grade_2_Moderate_F1": round(pcm["Moderate NPDR"]["f1_score"], 4),
            "Grade_3_Severe_F1": round(pcm["Severe NPDR"]["f1_score"], 4),
            "Grade_4_PDR_F1": round(pcm.get("Proliferative DR", pcm.get("PDR", {}))["f1_score"], 4),
            "Macro_F1": round(m["macro_f1"], 4),
            "QWK": round(m["quadratic_weighted_kappa"], 4)
        })

    df2 = pd.DataFrame(records_2)
    df2_path = METRICS_DIR / "mild_npdr_improvement_comparison.csv"
    df2.to_csv(df2_path, index=False)
    print(f"Saved: {df2_path}")

    # Table 3: Impact of Test Set Deduplication (Removing Contradictory IDRiD_064)
    m_bin_off = load_json("resnet50_binary_clahe_binary_referable_s42_s42_test_metrics.json")
    m_bin_dedup = load_json("resnet50_binary_clahe_binary_referable_s42_s42_dedup_test_metrics.json")

    records_3 = []
    pairs = [
        ("ResNet-50 Baseline (5-Class)", m_r50_sqrt_off, m_r50_sqrt_dedup),
        ("Corrected Hybrid (5-Class)", m_hyb_sqrt_off, m_hyb_sqrt_dedup),
        ("ResNet-50 Focal Loss (5-Class)", m_r50_focal_off, m_r50_focal_dedup),
        ("ResNet-50 Binary Referable DR", m_bin_off, m_bin_dedup)
    ]
    for model_name, m_off, m_dedup in pairs:
        if m_off is None or m_dedup is None:
            continue
        records_3.append({
            "Model": model_name,
            "Official_Accuracy": round(m_off["accuracy"] * 100, 2),
            "Deduplicated_Accuracy": round(m_dedup["accuracy"] * 100, 2),
            "Accuracy_Delta": round((m_dedup["accuracy"] - m_off["accuracy"]) * 100, 2),
            "Official_QWK": round(m_off["quadratic_weighted_kappa"], 4),
            "Deduplicated_QWK": round(m_dedup["quadratic_weighted_kappa"], 4),
            "QWK_Delta": round(m_dedup["quadratic_weighted_kappa"] - m_off["quadratic_weighted_kappa"], 4),
            "Official_Macro_F1": round(m_off["macro_f1"], 4),
            "Deduplicated_Macro_F1": round(m_dedup["macro_f1"], 4),
            "Macro_F1_Delta": round(m_dedup["macro_f1"] - m_off["macro_f1"], 4),
            "Note": "IDRiD_064 (Test Grade 3, Train IDRiD_118 Grade 0) removed in Deduplicated split"
        })

    df3 = pd.DataFrame(records_3)
    df3_path = METRICS_DIR / "deduplicated_vs_official_test_comparison.csv"
    df3.to_csv(df3_path, index=False)
    print(f"Saved: {df3_path}")

    # Table 4: Binary Referable DR Experiment (Clinical Triage Task)
    if m_bin_off and m_bin_dedup:
        records_4 = []
        for proto, n_samples, m in [("Official Benchmark", 103, m_bin_off), ("Deduplicated Clean", 102, m_bin_dedup)]:
            pcm = m["per_class_metrics"]
            records_4.append({
                "Evaluation_Protocol": proto,
                "Test_Samples": n_samples,
                "Accuracy": round(m["accuracy"] * 100, 2),
                "Macro_F1": round(m["macro_f1"], 4),
                "QWK": round(m["quadratic_weighted_kappa"], 4),
                "ROC_AUC": round(m["overall_roc_auc"], 4),
                "PR_AUC": round(m["overall_pr_auc"], 4),
                "NonReferable_Precision": round(pcm["Non-Referable DR (Grade 0-1)"]["precision"], 4),
                "NonReferable_Recall_Specificity": round(pcm["Non-Referable DR (Grade 0-1)"]["recall_sensitivity"], 4),
                "NonReferable_F1": round(pcm["Non-Referable DR (Grade 0-1)"]["f1_score"], 4),
                "NonReferable_Support": pcm["Non-Referable DR (Grade 0-1)"]["support"],
                "Referable_Precision": round(pcm["Referable DR (Grade 2-4)"]["precision"], 4),
                "Referable_Sensitivity": round(pcm["Referable DR (Grade 2-4)"]["recall_sensitivity"], 4),
                "Referable_F1": round(pcm["Referable DR (Grade 2-4)"]["f1_score"], 4),
                "Referable_Support": pcm["Referable DR (Grade 2-4)"]["support"]
            })
        df4 = pd.DataFrame(records_4)
        df4_path = METRICS_DIR / "binary_referable_dr_comparison.csv"
        df4.to_csv(df4_path, index=False)
        print(f"Saved: {df4_path}")

if __name__ == "__main__":
    generate_tables()
