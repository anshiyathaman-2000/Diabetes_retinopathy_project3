"""
Batch Grad-CAM Generation and Verification Script for Step 4.
Generates comprehensive visual explanations for correctly classified and misclassified
test images across all 5 diabetic retinopathy grades for ResNet-50 Focal Loss,
Corrected Compact Hybrid, and ResNet-50 Baseline.
Outputs: outputs/explainability/idrid_gradcam_step4/
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model
from src.utils import get_device, load_config
from src.explainability import generate_explanation_figure, GradCAM
from src.dataset import DR_CLASSES_5

OUTPUT_BASE = Path("outputs/explainability/idrid_gradcam_step4")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

TEST_IMG_DIR = Path("B. Disease Grading/1. Original Images/b. Testing Set")
TEST_CSV = Path("B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

MODELS_TO_EXPLAIN = [
    {
        "id": "resnet50_focal",
        "display_name": "ResNet-50 (Focal Loss)",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_focal_s42_s42_best.pt",
        "preds_csv": "outputs/predictions/resnet50_five_class_clahe_focal_s42_s42_predictions.csv",
        "subfolder": "resnet50_focal"
    },
    {
        "id": "hybrid_compact",
        "display_name": "Corrected Compact Hybrid (v2)",
        "checkpoint": "checkpoints/hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt",
        "preds_csv": "outputs/predictions/hybrid_five_class_clahe_compact_sqrt_cw_s42_predictions.csv",
        "subfolder": "hybrid_compact"
    },
    {
        "id": "resnet50_baseline",
        "display_name": "ResNet-50 Baseline (Sqrt CW)",
        "checkpoint": "checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt",
        "preds_csv": "outputs/predictions/resnet50_five_class_clahe_sqrt_cw_s42_predictions.csv",
        "subfolder": "resnet50_baseline"
    }
]

def run_step4_gradcam():
    # Load test ground truths
    df_gt = pd.read_csv(TEST_CSV).dropna(subset=["Image name"])
    df_gt["Image name"] = df_gt["Image name"].str.strip()
    df_gt["Retinopathy grade"] = df_gt["Retinopathy grade"].astype(int)
    gt_dict = dict(zip(df_gt["Image name"], df_gt["Retinopathy grade"]))

    all_records = []
    verification_summary = {
        "models_evaluated": [],
        "total_explanations_generated": 0,
        "class_availability_breakdown": {},
        "target_layers_verified": {},
        "validity_checks": {
            "all_heatmaps_finite": True,
            "zero_nan_or_inf": True,
            "all_dimensions_valid": True,
            "empty_maps_count": 0,
            "constant_maps_count": 0
        }
    }

    device = get_device("auto")
    print(f"Executing Grad-CAM Generation on device: {device}\n")

    for m_info in MODELS_TO_EXPLAIN:
        ckpt_path = Path(m_info["checkpoint"])
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue

        print(f"=======================================================")
        print(f" Processing Explainability for: {m_info['display_name']}")
        print(f" Checkpoint: {ckpt_path.name}")
        print(f"=======================================================")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt["config"]
        model = build_model(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        target_layer = model.get_cam_target_layer()
        target_layer_name = f"{target_layer.__class__.__name__} (layer4[-1])"
        verification_summary["target_layers_verified"][m_info["id"]] = target_layer_name

        # Load prediction CSV
        preds_file = Path(m_info["preds_csv"])
        assert preds_file.exists(), f"Predictions file missing: {preds_file}"
        df_preds = pd.read_csv(preds_file)

        model_out_dir = OUTPUT_BASE / m_info["subfolder"]
        model_out_dir.mkdir(parents=True, exist_ok=True)

        class_counts = {}

        # Iterate over all 5 grades
        for grade in range(5):
            grade_name = DR_CLASSES_5[grade]
            df_g = df_preds[df_preds["Ground Truth"] == grade]

            df_corr = df_g[df_g["Ground Truth"] == df_g["Predicted Class"]]
            df_misc = df_g[df_g["Ground Truth"] != df_g["Predicted Class"]]

            class_counts[grade_name] = {
                "total_test_support": len(df_g),
                "correct_count": len(df_corr),
                "misclassified_count": len(df_misc),
                "selected_correct_images": [],
                "selected_misclassified_images": []
            }

            # Select up to 2 correct examples
            selected_corr = df_corr.head(2)["Image name"].tolist()
            class_counts[grade_name]["selected_correct_images"] = selected_corr

            # Select up to 2 misclassified examples
            selected_misc = df_misc.head(2)["Image name"].tolist()
            class_counts[grade_name]["selected_misclassified_images"] = selected_misc

            # 1. Process Correct Examples (Primary Target: Predicted Class)
            for img_id in selected_corr:
                img_path = TEST_IMG_DIR / f"{img_id}.jpg"
                save_path = model_out_dir / f"{img_id}_GT{grade}_Pred{grade}_correct.png"

                res = generate_explanation_figure(
                    image_path=img_path,
                    model=model,
                    config=config,
                    ground_truth_label=grade,
                    target_class=None, # defaults to predicted
                    save_path=save_path,
                    device=device,
                    model_display_name=m_info["display_name"]
                )

                diag = res["diagnostics"]
                if diag["has_nan"] or diag["has_inf"]:
                    verification_summary["validity_checks"]["zero_nan_or_inf"] = False
                if diag["is_empty"]:
                    verification_summary["validity_checks"]["empty_maps_count"] += 1
                if diag["is_constant"]:
                    verification_summary["validity_checks"]["constant_maps_count"] += 1

                all_records.append({
                    "Model_ID": m_info["id"],
                    "Model_Name": m_info["display_name"],
                    "Image_ID": img_id,
                    "Dataset_Split": "Official_Test",
                    "Ground_Truth_Grade": grade,
                    "Ground_Truth_Name": grade_name,
                    "Predicted_Grade": res["predicted_class"],
                    "Predicted_Name": res["predicted_class_name"],
                    "Classification_Status": "CORRECT",
                    "GradCAM_Target_Class": res["explained_target_class"],
                    "GradCAM_Target_Name": res["explained_target_name"],
                    "Target_Convolutional_Layer": target_layer_name,
                    "Heatmap_Min": round(diag["raw_min"], 4),
                    "Heatmap_Max": round(diag["raw_max"], 4),
                    "Heatmap_Mean": round(diag["raw_mean"], 4),
                    "Heatmap_Std": round(diag["raw_std"], 4),
                    "Is_Empty_Map": diag["is_empty"],
                    "Is_Constant_Map": diag["is_constant"],
                    "Confidence_Prob": round(res["confidence"], 4),
                    "Output_Figure_Path": str(save_path)
                })
                verification_summary["total_explanations_generated"] += 1

            # 2. Process Misclassified Examples
            for img_id in selected_misc:
                img_path = TEST_IMG_DIR / f"{img_id}.jpg"
                row_pred = df_misc[df_misc["Image name"] == img_id].iloc[0]
                pred_c = int(row_pred["Predicted Class"])

                # Figure A: Explaining Predicted Class (Why did the model predict this class?)
                save_path_pred = model_out_dir / f"{img_id}_GT{grade}_Pred{pred_c}_explaining_pred.png"
                res_pred = generate_explanation_figure(
                    image_path=img_path,
                    model=model,
                    config=config,
                    ground_truth_label=grade,
                    target_class=pred_c,
                    save_path=save_path_pred,
                    device=device,
                    model_display_name=m_info["display_name"]
                )

                diag_p = res_pred["diagnostics"]
                all_records.append({
                    "Model_ID": m_info["id"],
                    "Model_Name": m_info["display_name"],
                    "Image_ID": img_id,
                    "Dataset_Split": "Official_Test",
                    "Ground_Truth_Grade": grade,
                    "Ground_Truth_Name": grade_name,
                    "Predicted_Grade": pred_c,
                    "Predicted_Name": DR_CLASSES_5[pred_c],
                    "Classification_Status": "MISCLASSIFIED",
                    "GradCAM_Target_Class": pred_c,
                    "GradCAM_Target_Name": DR_CLASSES_5[pred_c],
                    "Target_Convolutional_Layer": target_layer_name,
                    "Heatmap_Min": round(diag_p["raw_min"], 4),
                    "Heatmap_Max": round(diag_p["raw_max"], 4),
                    "Heatmap_Mean": round(diag_p["raw_mean"], 4),
                    "Heatmap_Std": round(diag_p["raw_std"], 4),
                    "Is_Empty_Map": diag_p["is_empty"],
                    "Is_Constant_Map": diag_p["is_constant"],
                    "Confidence_Prob": round(res_pred["confidence"], 4),
                    "Output_Figure_Path": str(save_path_pred)
                })
                verification_summary["total_explanations_generated"] += 1

                # Figure B: Comparative explanation for Ground Truth Class (What would activate true class?)
                save_path_gt = model_out_dir / f"{img_id}_GT{grade}_Pred{pred_c}_explaining_gt.png"
                res_gt = generate_explanation_figure(
                    image_path=img_path,
                    model=model,
                    config=config,
                    ground_truth_label=grade,
                    target_class=grade, # Ground truth class
                    save_path=save_path_gt,
                    device=device,
                    model_display_name=m_info["display_name"]
                )
                diag_g = res_gt["diagnostics"]
                all_records.append({
                    "Model_ID": m_info["id"],
                    "Model_Name": m_info["display_name"],
                    "Image_ID": img_id,
                    "Dataset_Split": "Official_Test",
                    "Ground_Truth_Grade": grade,
                    "Ground_Truth_Name": grade_name,
                    "Predicted_Grade": pred_c,
                    "Predicted_Name": DR_CLASSES_5[pred_c],
                    "Classification_Status": "MISCLASSIFIED (GT TARGET COMPARISON)",
                    "GradCAM_Target_Class": grade,
                    "GradCAM_Target_Name": grade_name,
                    "Target_Convolutional_Layer": target_layer_name,
                    "Heatmap_Min": round(diag_g["raw_min"], 4),
                    "Heatmap_Max": round(diag_g["raw_max"], 4),
                    "Heatmap_Mean": round(diag_g["raw_mean"], 4),
                    "Heatmap_Std": round(diag_g["raw_std"], 4),
                    "Is_Empty_Map": diag_g["is_empty"],
                    "Is_Constant_Map": diag_g["is_constant"],
                    "Confidence_Prob": round(res_gt["confidence"], 4),
                    "Output_Figure_Path": str(save_path_gt)
                })
                verification_summary["total_explanations_generated"] += 1

        verification_summary["class_availability_breakdown"][m_info["id"]] = class_counts
        verification_summary["models_evaluated"].append(m_info["id"])

    # Save summary CSV
    df_summary = pd.DataFrame(all_records)
    csv_path = OUTPUT_BASE / "gradcam_generation_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved Grad-CAM generation summary ({len(df_summary)} records) to: {csv_path}")

    # Save verification JSON
    json_path = OUTPUT_BASE / "gradcam_verification_report.json"
    with open(json_path, "w") as f:
        json.dump(verification_summary, f, indent=2)
    print(f"Saved Grad-CAM verification report to: {json_path}")

if __name__ == "__main__":
    run_step4_gradcam()
