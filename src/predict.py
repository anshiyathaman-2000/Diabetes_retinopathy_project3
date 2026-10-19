"""
Inference Script for Diabetic Retinopathy Severity Prediction.
Accepts an image path, preprocesses the fundus image, and produces
predicted DR grades, clinical diagnosis names, and confidence scores.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing import preprocess_fundus_image, get_eval_transforms
from src.dataset import DR_CLASSES_5, DR_CLASSES_BINARY
from src.models import build_model
from src.utils import get_device

GRADE_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy - Retinal vasculature appears normal; no microaneurysms detected.",
    1: "Mild NPDR - Presence of microaneurysms only; early clinical observation indicated.",
    2: "Moderate NPDR - More than microaneurysms but less than severe NPDR; referable DR threshold met.",
    3: "Severe NPDR - Extensive intraretinal hemorrhages, venous beading, or IRMA (4-2-1 rule). High risk of progression to PDR.",
    4: "Proliferative DR - Neovascularization or vitreous/preretinal hemorrhage observed. Urgent referral indicated."
}

def predict_single_image(
    image_path: str,
    checkpoint_path: str,
    device_str: str = "auto"
) -> Dict[str, Any]:
    """Runs inference on a single retinal fundus image."""
    img_p = Path(image_path)
    ckpt_p = Path(checkpoint_path)

    if not img_p.exists():
        raise FileNotFoundError(f"Image not found at: {img_p}")
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    ckpt = torch.load(ckpt_p, map_location="cpu")
    config = ckpt["config"]
    task = config["task"]["type"]
    class_names = DR_CLASSES_5 if task == "five_class" else DR_CLASSES_BINARY

    device = get_device(device_str)
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess
    img_size = config["preprocessing"]["image_size"]
    prep_method = config["preprocessing"].get("method", "clahe")
    pil_img = preprocess_fundus_image(
        str(img_p),
        image_size=img_size,
        method=prep_method,
        crop_border=True
    )

    norm_mean = config["preprocessing"]["normalization"]["mean"]
    norm_std = config["preprocessing"]["normalization"]["std"]
    eval_transform = get_eval_transforms(mean=tuple(norm_mean), std=tuple(norm_std))
    tensor_input = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    confidence = float(probs[pred_idx])

    result = {
        "image_file": img_p.name,
        "task": task,
        "predicted_grade": pred_idx,
        "predicted_class_name": class_names[pred_idx],
        "confidence": confidence,
        "class_probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "referable_status": "Referable DR (Grade >= 2)" if pred_idx >= 2 else "Non-Referable DR (Grade 0-1)",
        "clinical_guidance": GRADE_DESCRIPTIONS.get(pred_idx, "N/A"),
        "disclaimer": (
            "NOTICE: This model is developed exclusively for academic research and educational evaluation. "
            "It is not approved for clinical diagnostics or autonomous medical triage."
        )
    }

    return result

def main():
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy Severity from Fundus Image.")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus image (.jpg/.png)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'mps', 'cuda', 'cpu')")
    args = parser.parse_args()

    res = predict_single_image(args.image, args.checkpoint, device_str=args.device)

    print("\n=======================================================")
    print(" Diabetic Retinopathy Diagnostic Inference Result")
    print("=======================================================")
    print(f" Image File          : {res['image_file']}")
    print(f" Predicted Class     : {res['predicted_class_name']}")
    print(f" Confidence Score    : {res['confidence']*100:.2f}%")
    print(f" Referable Status    : {res['referable_status']}")
    print(f" Clinical Summary    : {res['clinical_guidance']}")
    print("-------------------------------------------------------")
    print(" Full Probability Distribution:")
    for cname, p in res["class_probabilities"].items():
        bar = "█" * int(p * 30)
        print(f"   {cname:<24}: {p*100:5.2f}% {bar}")
    print("=======================================================")
    print(f"\n{res['disclaimer']}\n")

if __name__ == "__main__":
    main()
