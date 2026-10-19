"""
Evaluation Script for Exp B1 (ResNet-50 512x512) Checkpoint.
Evaluates under:
1. Official Benchmark Protocol (N=103)
2. Sensitivity Analysis Protocol (N=102, excluding IDRiD_064)
Saves:
- Metrics JSON
- Predictions CSV
- Comparison against baseline ResNet-50 256x256
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import set_seed, get_device, calculate_metrics
from src.dataset import get_idrid_dataloaders, DR_CLASSES_5
from src.models import build_model

def evaluate_model_on_protocol(checkpoint_path: Path, protocol: str = "official"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    cfg.setdefault("data", {})["test_protocol"] = protocol
    device = get_device(cfg["hardware"].get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader, _ = get_idrid_dataloaders(cfg, seed=cfg.get("seed", 42))
    expected_count = 103 if protocol == "official" else 102
    assert len(test_loader.dataset) == expected_count, (
        f"Expected {expected_count} test samples, got {len(test_loader.dataset)}"
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
    metrics["checkpoint_path"] = str(checkpoint_path)

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
    ckpt_path = ROOT / "checkpoints/resnet50_five_class_clahe_exp_b1_res512_s42_best.pt"
    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"

    metrics_dir = ROOT / "outputs/metrics"
    preds_dir = ROOT / "outputs/predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for proto in ["official", "deduplicated"]:
        print(f"Evaluating Exp B1 (512x512) on {proto} test set...")
        m, df_p = evaluate_model_on_protocol(ckpt_path, protocol=proto)
        results[proto] = m
        
        json_path = metrics_dir / f"resnet50_exp_b1_res512_{proto}_test_metrics.json"
        csv_path = preds_dir / f"resnet50_exp_b1_res512_{proto}_predictions.csv"
        
        with open(json_path, "w") as f:
            json.dump(m, f, indent=2)
        df_p.to_csv(csv_path, index=False)
        print(f"  Accuracy: {m['accuracy']*100:.2f}% | QWK: {m['quadratic_weighted_kappa']:.4f} | Macro-F1: {m['macro_f1']:.4f}")
        print(f"  Saved: {json_path.name}")
        print(f"  Saved: {csv_path.name}")

    print("\n" + "=" * 70)
    print("EXP B1 TEST EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
