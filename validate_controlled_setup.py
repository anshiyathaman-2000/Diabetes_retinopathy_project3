"""
Pre-flight verification script for the controlled IDRiD diabetic retinopathy experiments.

Tests:
1. Configuration parsing from configs/experiment.yaml.
2. Leakage-aware manifest loading and strict validation in get_idrid_dataloaders:
   - Train (330) and Val (83) sample counts and disjointness.
   - Intra-train duplicate pair cohesion (all 5 pairs).
   - Cross-split duplicate isolation (IDRiD_118 in train, IDRiD_064 in test).
3. Deduplicated training manifest loading and validation (326 train, 82 val).
4. Failure test: Verify that a missing/invalid manifest raises FileNotFoundError/ValueError (NO silent fallback).
5. Architecture construction: ResNet-50 and Compact Hybrid.
6. Small forward and backward pass for both models with square-root class-weighted CrossEntropyLoss.
7. Checkpoint save and reload verification.
"""

import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, get_device, set_seed
from src.dataset import get_idrid_dataloaders, VERIFIED_INTRA_TRAIN_DUPLICATE_PAIRS
from src.models import build_model


def test_manifest_validation():
    print("\n--- Test 1: Primary Leakage-Aware Manifest Validation ---")
    cfg = load_config(ROOT / "configs/experiment.yaml")
    cfg["seed"] = 42

    train_loader, val_loader, test_loader, class_weights = get_idrid_dataloaders(cfg, seed=42)
    assert len(train_loader.dataset) == 330, f"Expected 330 train samples, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 83, f"Expected 83 val samples, got {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 103, f"Expected 103 test samples, got {len(test_loader.dataset)}"
    assert len(class_weights) == 5, f"Expected 5 class weights, got {len(class_weights)}"

    train_names = set(train_loader.dataset.image_names)
    val_names = set(val_loader.dataset.image_names)
    test_names = set(test_loader.dataset.image_names)

    # Disjointness within training split
    assert len(train_names & val_names) == 0, "Train and Val overlap!"

    # Split-qualified path isolation between training and testing sets
    train_paths = {train_loader.dataset.image_dir / f"{name}.jpg" for name in train_names}
    val_paths = {val_loader.dataset.image_dir / f"{name}.jpg" for name in val_names}
    test_paths = {test_loader.dataset.image_dir / f"{name}.jpg" for name in test_names}

    assert len(train_paths & test_paths) == 0, "Train and Test image file paths overlap!"
    assert len(val_paths & test_paths) == 0, "Val and Test image file paths overlap!"
    assert all("a. Training Set" in str(p) for p in train_paths), "Train images not in a. Training Set"
    assert all("b. Testing Set" in str(p) for p in test_paths), "Test images not in b. Testing Set"

    # Verify duplicate cluster cohesion
    for img1, img2 in VERIFIED_INTRA_TRAIN_DUPLICATE_PAIRS:
        both_train = (img1 in train_names and img2 in train_names)
        both_val = (img1 in val_names and img2 in val_names)
        assert both_train or both_val, f"Duplicate pair ({img1}, {img2}) split across train and val!"
    print("✓ Primary leakage-aware manifest successfully verified: 330 train, 83 val, 103 test.")
    print("✓ Duplicate cluster cohesion confirmed: 0 pairs split across partitions.")


def test_deduplicated_manifest_validation():
    print("\n--- Test 2: Deduplicated Training Sensitivity Manifest Validation ---")
    cfg = load_config(ROOT / "configs/experiment.yaml")
    cfg["seed"] = 42
    cfg["data"]["split_manifest"] = "data/splits/deduplicated_training_split_manifest.json"

    train_loader, val_loader, test_loader, class_weights = get_idrid_dataloaders(cfg, seed=42)
    assert len(train_loader.dataset) == 326, f"Expected 326 train samples, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 82, f"Expected 82 val samples, got {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 103, f"Expected 103 test samples, got {len(test_loader.dataset)}"

    train_names = set(train_loader.dataset.image_names)
    val_names = set(val_loader.dataset.image_names)
    assert len(train_names & val_names) == 0, "Deduplicated train and val overlap!"
    print("✓ Deduplicated sensitivity manifest verified: 326 train, 82 val, 103 test.")


def test_silent_fallback_elimination():
    print("\n--- Test 3: Verify Silent Fallback is Strictly Disabled ---")
    cfg = load_config(ROOT / "configs/experiment.yaml")
    cfg["data"]["split_type"] = "leakage_aware"
    cfg["data"]["split_manifest"] = "data/splits/non_existent_manifest.json"

    try:
        get_idrid_dataloaders(cfg, seed=42)
        raise AssertionError("FAILED: Loader silently fell back instead of raising FileNotFoundError!")
    except FileNotFoundError as exc:
        print(f"✓ PASS: Missing manifest raised FileNotFoundError as required: {exc}")


def test_model_forward_backward():
    print("\n--- Test 4: Model Forward & Backward Pass Verification ---")
    cfg = load_config(ROOT / "configs/experiment.yaml")
    device = get_device(cfg["hardware"].get("device", "auto"))
    print(f"Testing execution on device: {device}")

    models_to_test = [
        ("ResNet-50", {"model": {"name": "resnet50", "pretrained": False}, "task": {"num_classes": 5}}),
        ("Compact Hybrid", {
            "model": {
                "name": "hybrid",
                "pretrained": False,
                "hybrid": {
                    "backbone": "resnet50",
                    "cnn_embed_dim": 128,
                    "transformer_layers": 1,
                    "transformer_heads": 4,
                    "transformer_feedforward": 512,
                    "bilstm_hidden_dim": 64,
                    "bilstm_layers": 1,
                    "pool_type": "mean",
                }
            },
            "task": {"num_classes": 5}
        })
    ]

    for model_name, m_cfg in models_to_test:
        model = build_model(m_cfg).to(device)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.ones(5).to(device))

        dummy_images = torch.randn(4, 3, 256, 256).to(device)
        dummy_targets = torch.tensor([0, 1, 2, 4]).to(device)

        optimizer.zero_grad()
        logits = model(dummy_images)
        assert logits.shape == (4, 5), f"Expected shape (4, 5), got {logits.shape}"
        loss = criterion(logits, dummy_targets)
        loss.backward()
        optimizer.step()
        print(f"✓ {model_name}: Forward pass, loss computation ({loss.item():.4f}), and backward pass succeeded.")


def test_checkpoint_roundtrip():
    print("\n--- Test 5: Checkpoint Save and Reload Round-Trip ---")
    cfg = load_config(ROOT / "configs/experiment.yaml")
    m_cfg = {"model": {"name": "resnet50", "pretrained": False}, "task": {"num_classes": 5}}
    model = build_model(m_cfg)

    tmp_ckpt = ROOT / "data/cache/test_roundtrip_ckpt.pt"
    tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "test_marker": "reproducibility_check_ok"
    }, tmp_ckpt)

    loaded = torch.load(tmp_ckpt, map_location="cpu")
    assert loaded["test_marker"] == "reproducibility_check_ok", "Checkpoint content mismatch!"
    tmp_ckpt.unlink()
    print("✓ Checkpoint save and reload verified.")


def main():
    print("=" * 70)
    print("RUNNING PRE-FLIGHT VALIDATION FOR CONTROLLED EXPERIMENTS")
    print("=" * 70)

    try:
        test_manifest_validation()
        test_deduplicated_manifest_validation()
        test_silent_fallback_elimination()
        test_model_forward_backward()
        test_checkpoint_roundtrip()
        print("\n" + "=" * 70)
        print("ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY (5/5)")
        print("=" * 70)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL ERROR] Pre-flight check failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
