"""
Training Pipeline for Diabetic Retinopathy Detection.
Supports:
1. ResNet50 Baseline and Proposed CNN-Transformer-BiLSTM models.
2. 5-class Severity vs Binary Referable DR classification.
3. AdamW optimizer with Cosine Annealing learning rate schedule.
4. Class-weighted CrossEntropyLoss for handling severe class imbalance.
5. Early stopping based on validation Quadratic Weighted Kappa (QWK) or Macro F1.
6. Comprehensive checkpointing, metric logging, and curve generation.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.utils import (
    set_seed,
    get_device,
    load_config,
    calculate_metrics,
    plot_training_curves,
    FocalLoss
)
from src.dataset import (
    get_idrid_dataloaders,
    DR_CLASSES_5,
    DR_CLASSES_BINARY
)
from src.models import build_model

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str = "five_class",
    clip_norm: float = 1.0,
    grad_accum_steps: int = 1
) -> Tuple[float, float, float]:
    """Train for a single epoch with optional gradient accumulation."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_batches = len(dataloader)

    optimizer.zero_grad()
    for step, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        scaled_loss = loss / grad_accum_steps if grad_accum_steps > 1 else loss
        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_preds, task=task)
    return epoch_loss, metrics["accuracy"], metrics["quadratic_weighted_kappa"]

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    task: str = "five_class"
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate model on validation or test dataloader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs_np = np.array(all_probs)
    metrics = calculate_metrics(all_targets, all_preds, y_probs=all_probs_np, task=task)
    return epoch_loss, metrics

def run_training(
    config: Dict[str, Any],
    resume_checkpoint: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Executes the complete reproducible training workflow."""
    seed = config.get("seed", 42)
    set_seed(seed)

    device_str = config["hardware"].get("device", "auto")
    device = get_device(device_str)
    print(f"\n=======================================================")
    print(f" Starting Experiment: {config.get('experiment_name', 'dr_experiment')}")
    print(f" Model Architecture : {config['model']['name']}")
    print(f" Task Type          : {config['task']['type']} ({config['task']['num_classes']} classes)")
    print(f" Hardware Device    : {device}")
    print(f" Random Seed        : {seed}")
    print(f" Preprocessing      : {config['preprocessing']['method'].upper()} (Size: {config['preprocessing']['image_size']}x{config['preprocessing']['image_size']})")
    print(f"=======================================================\n")

    # 1. Prepare DataLoaders
    train_loader, val_loader, test_loader, class_weights = get_idrid_dataloaders(config, seed=seed)
    print(f"Dataset summary:")
    print(f"  - Training samples  : {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    print(f"  - Test samples      : {len(test_loader.dataset)}")
    print(f"  - Computed class weights: {class_weights.numpy().round(3).tolist()}\n")

    # 2. Build Model
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: Total = {total_params:,} | Trainable = {trainable_params:,}")

    # 3. Criterion with optional class imbalance weighting & Focal Loss
    loss_type = config["training"].get("loss_type", "ce").lower()
    focal_gamma = float(config["training"].get("focal_gamma", 2.0))
    use_cw = config["training"].get("use_class_weights", True)
    if isinstance(use_cw, str):
        use_cw = use_cw.lower() in ["true", "1", "yes"]

    weights = class_weights.to(device) if use_cw else None

    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        print(f"Using Focal Loss (gamma={focal_gamma}, weighted={use_cw}): alpha={weights.cpu().numpy().round(3).tolist() if weights is not None else 'None'}\n")
    elif use_cw:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using Class-Weighted CrossEntropyLoss ({config['training'].get('class_weight_method', 'inverse')}): {class_weights.numpy().round(3).tolist()}\n")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Unweighted CrossEntropyLoss.\n")

    # 4. Optimizer and LR Scheduler
    lr = float(config["training"].get("learning_rate", 1e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    opt_name = config["training"].get("optimizer", "adamw").lower()
    clip_norm = float(config["training"].get("grad_clip_norm", 1.0))

    backbone_lr = config["training"].get("backbone_lr", None)
    head_lr = config["training"].get("head_lr", None)

    if (backbone_lr is not None or head_lr is not None) and hasattr(model, "get_parameter_groups") and config["model"]["name"].lower() != "resnet50":
        bb_lr = float(backbone_lr if backbone_lr is not None else lr)
        hd_lr = float(head_lr if head_lr is not None else lr)
        print(f"Using Differential Learning Rates: Backbone={bb_lr:.2e}, Head={hd_lr:.2e}")
        param_groups = model.get_parameter_groups(backbone_lr=bb_lr, head_lr=hd_lr, weight_decay=weight_decay)
        if opt_name == "adamw":
            optimizer = AdamW(param_groups)
        else:
            optimizer = Adam(param_groups)
    else:
        if opt_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initial backbone freeze if requested
    freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        print(f"Freezing CNN backbone for initial {freeze_epochs} epochs...")
        model.freeze_backbone()

    epochs = config["training"].get("epochs", 25)
    sched_type = config["training"].get("lr_scheduler", "cosine").lower()
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    # Output paths
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    plots_dir = Path(config["paths"]["plots_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    for p in [checkpoints_dir, plots_dir, metrics_dir]:
        p.mkdir(parents=True, exist_ok=True)

    exp_suffix = config.get("exp_suffix", "")
    suffix_part = f"_{exp_suffix}" if exp_suffix else ""
    exp_id = f"{config['model']['name']}_{config['task']['type']}_{config['preprocessing']['method']}{suffix_part}_s{seed}"
    best_ckpt_path = checkpoints_dir / f"{exp_id}_best.pt"
    last_ckpt_path = checkpoints_dir / f"{exp_id}_last.pt"

    # Early stopping config
    es_cfg = config["training"].get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    es_patience = es_cfg.get("patience", 7)
    monitor_metric = es_cfg.get("monitor_metric", "val_qwk")
    mode = es_cfg.get("mode", "max")

    best_metric_val = -float("inf") if mode == "max" else float("inf")
    best_val_metrics = {}
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_qwk": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_qwk": [],
        "lr": []
    }

    start_epoch = 1

    # Resume from checkpoint if supplied
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        best_metric_val = ckpt.get("best_metric_val", best_metric_val)
        best_val_metrics = ckpt.get("val_metrics", {})
        print(f"Resumed at epoch {start_epoch} with best metric: {best_metric_val:.4f}")

    start_time = time.time()
    print("Beginning training loop...")

    for epoch in range(start_epoch, epochs + 1):
        # Gradual unfreezing check
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            print(f"\n>>> [Gradual Unfreezing] Unfreezing CNN backbone at epoch {epoch}!")
            model.unfreeze_backbone()

        grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))

        epoch_start = time.time()

        train_loss, train_acc, train_qwk = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            task=config["task"]["type"],
            clip_norm=clip_norm,
            grad_accum_steps=grad_accum_steps
        )

        val_loss, val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            task=config["task"]["type"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if sched_type == "cosine":
            scheduler.step()
        elif sched_type == "plateau":
            scheduler.step(val_metrics["quadratic_weighted_kappa"])

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_qwk"].append(train_qwk)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])
        history["val_qwk"].append(val_metrics["quadratic_weighted_kappa"])
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% QWK: {train_qwk:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']*100:.1f}% "
            f"Macro-F1: {val_metrics['macro_f1']:.3f} QWK: {val_metrics['quadratic_weighted_kappa']:.3f} | "
            f"LR: {current_lr:.6f} ({epoch_time:.1f}s)",
            flush=True
        )

        # Check for improvement
        if monitor_metric == "val_macro_f1":
            curr_metric = val_metrics["macro_f1"]
        elif monitor_metric == "val_loss":
            curr_metric = val_loss
        else:
            curr_metric = val_metrics["quadratic_weighted_kappa"]

        is_improvement = (curr_metric > best_metric_val) if mode == "max" else (curr_metric < best_metric_val)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "best_metric_val": best_metric_val
        }, last_ckpt_path)

        if is_improvement:
            best_metric_val = curr_metric
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "val_metrics": val_metrics,
                "config": config,
                "best_metric_val": best_metric_val,
                "class_weights": class_weights
            }, best_ckpt_path)
            print(f"  --> Checkpoint saved! New best {monitor_metric}: {best_metric_val:.4f}", flush=True)
        else:
            patience_counter += 1
            if es_enabled and patience_counter >= es_patience:
                print(f"\n[Early Stopping] No improvement for {es_patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Metric ({monitor_metric}): {best_metric_val:.4f}")
    print(f"Best model saved to: {best_ckpt_path}")

    # Save validation metrics JSON
    val_json_path = metrics_dir / f"{exp_id}_val_metrics.json"
    val_metrics_to_save = dict(best_val_metrics)
    val_metrics_to_save["best_validation_metric"] = best_metric_val
    val_metrics_to_save["monitor_metric"] = monitor_metric
    val_metrics_to_save["total_training_time_seconds"] = total_training_time
    with open(val_json_path, "w") as f:
        json.dump(val_metrics_to_save, f, indent=2)

    # Save per-class validation metrics table to CSV if present
    if "per_class_metrics" in best_val_metrics:
        df_per_class = pd.DataFrame.from_dict(best_val_metrics["per_class_metrics"], orient="index")
        val_csv_path = metrics_dir / f"{exp_id}_val_per_class_table.csv"
        df_per_class.to_csv(val_csv_path, index_label="Class")

    # Plot and save curves
    metric_label = "macro_f1" if monitor_metric == "val_macro_f1" else "qwk"
    plot_training_curves(
        history=history,
        save_path=plots_dir / f"{exp_id}_training_curves.png",
        metric_name=metric_label
    )

    return model, {
        "best_checkpoint_path": str(best_ckpt_path),
        "total_training_time_seconds": total_training_time,
        "best_validation_metric": best_metric_val,
        "best_val_metrics": best_val_metrics,
        "history": history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Models on IDRiD.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model name ('resnet50', 'hybrid', 'cnn_transformer', 'cnn_bilstm')")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone_lr", type=float, default=None, help="Override CNN backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=None, help="Override head learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=None, help="Number of warmup epochs to freeze backbone")
    parser.add_argument("--grad_clip_norm", type=float, default=None, help="Max norm for gradient clipping")
    parser.add_argument("--use_class_weights", type=str, default=None, help="Whether to use class-weighted loss ('true' or 'false')")
    parser.add_argument("--class_weight_method", type=str, default=None, help="Class weighting method ('inverse', 'sqrt', 'none')")
    parser.add_argument("--class_weight_power", type=float, default=None, help="Power exponent for class weights")
    parser.add_argument("--loss_type", type=str, default=None, help="Loss function ('ce' or 'focal')")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal loss gamma parameter")
    parser.add_argument("--split_type", type=str, default=None, help="Dataset split type ('standard' or 'leakage_aware')")
    parser.add_argument("--split_manifest", type=str, default=None, help="Path to split manifest JSON file")
    parser.add_argument("--test_protocol", type=str, default=None, help="Test protocol ('official' or 'deduplicated')")
    parser.add_argument("--sampler_type", type=str, default=None, help="Data sampler type ('none' or 'weighted')")
    parser.add_argument("--sampler_power", type=float, default=None, help="Power exponent for weighted sampler")
    parser.add_argument("--monitor_metric", type=str, default=None, help="Early stopping metric ('val_qwk' or 'val_macro_f1')")
    parser.add_argument("--embed_dim", type=int, default=None, help="Hybrid embed_dim")
    parser.add_argument("--transformer_layers", type=int, default=None, help="Hybrid transformer layers")
    parser.add_argument("--transformer_heads", type=int, default=None, help="Hybrid transformer heads")
    parser.add_argument("--bilstm_hidden_dim", type=int, default=None, help="Hybrid bilstm hidden dim")
    parser.add_argument("--pool_type", type=str, default=None, help="Pooling type ('mean_max', 'mean')")
    parser.add_argument("--exp_suffix", type=str, default=None, help="Suffix for checkpoint and experiment naming")
    parser.add_argument("--task", type=str, default=None, help="Override task ('five_class' or 'binary')")
    parser.add_argument("--preprocessing", type=str, default=None, help="Override preprocessing ('clahe' or 'original')")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--image_size", type=int, default=None, help="Override image resolution")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.backbone_lr is not None:
        cfg["training"]["backbone_lr"] = args.backbone_lr
    if args.head_lr is not None:
        cfg["training"]["head_lr"] = args.head_lr
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = args.weight_decay
    if args.freeze_backbone_epochs is not None:
        cfg["training"]["freeze_backbone_epochs"] = args.freeze_backbone_epochs
    if args.grad_clip_norm is not None:
        cfg["training"]["grad_clip_norm"] = args.grad_clip_norm
    if args.use_class_weights is not None:
        cfg["training"]["use_class_weights"] = args.use_class_weights.lower() in ["true", "1", "yes"]
    if args.class_weight_method is not None:
        cfg["training"]["class_weight_method"] = args.class_weight_method
    if args.class_weight_power is not None:
        cfg["training"]["class_weight_power"] = args.class_weight_power
    if args.loss_type is not None:
        cfg["training"]["loss_type"] = args.loss_type
    if args.focal_gamma is not None:
        cfg["training"]["focal_gamma"] = args.focal_gamma
    if args.split_type is not None:
        cfg.setdefault("data", {})["split_type"] = args.split_type
    if args.split_manifest is not None:
        cfg.setdefault("data", {})["split_manifest"] = args.split_manifest
    if args.test_protocol is not None:
        cfg.setdefault("data", {})["test_protocol"] = args.test_protocol
    if args.sampler_type is not None:
        cfg["training"]["sampler_type"] = args.sampler_type
    if args.sampler_power is not None:
        cfg["training"]["sampler_power"] = args.sampler_power
    if args.monitor_metric is not None:
        cfg["training"]["early_stopping"]["monitor_metric"] = args.monitor_metric
    if args.embed_dim is not None:
        cfg["model"].setdefault("hybrid", {})["cnn_embed_dim"] = args.embed_dim
    if args.transformer_layers is not None:
        cfg["model"].setdefault("hybrid", {})["transformer_layers"] = args.transformer_layers
    if args.transformer_heads is not None:
        cfg["model"].setdefault("hybrid", {})["transformer_heads"] = args.transformer_heads
    if args.bilstm_hidden_dim is not None:
        cfg["model"].setdefault("hybrid", {})["bilstm_hidden_dim"] = args.bilstm_hidden_dim
    if args.pool_type is not None:
        cfg["model"].setdefault("hybrid", {})["pool_type"] = args.pool_type
    if args.exp_suffix is not None:
        cfg["exp_suffix"] = args.exp_suffix
    if args.task:
        cfg["task"]["type"] = args.task
        cfg["task"]["num_classes"] = 5 if args.task == "five_class" else 2
    if args.preprocessing:
        cfg["preprocessing"]["method"] = args.preprocessing
    if args.seed:
        cfg["seed"] = args.seed
    if args.grad_accum_steps is not None:
        cfg.setdefault("training", {})["grad_accum_steps"] = args.grad_accum_steps
    if args.image_size is not None:
        cfg.setdefault("preprocessing", {})["image_size"] = args.image_size

    run_training(cfg, resume_checkpoint=args.resume)
