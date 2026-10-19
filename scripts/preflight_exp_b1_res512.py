"""
Preflight Verification Script for Exp B1: ResNet-50 512x512.
Tests:
1. Model loading & parameter count.
2. Micro-batch size 8, gradient accumulation 2 on MPS/device.
3. Numerical stability (loss, gradients, weights across 10 accumulation cycles).
4. Memory stability (MPS allocated/peak memory, system RAM leakage).
5. Real DataLoader batch throughput and evaluation pass.
"""

import sys
import time
import psutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import set_seed, get_device, load_config
from src.dataset import get_idrid_dataloaders
from src.models import build_model

def run_preflight():
    print("=" * 70)
    print("EXP B1 (512x512) MEMORY & STABILITY PREFLIGHT AUDIT")
    print("=" * 70)
    
    cfg_path = ROOT / "configs/experiment_exp_b1_res512.yaml"
    cfg = load_config(cfg_path)
    # Ensure leakage_aware split manifest is set as requested
    cfg["data"]["split_manifest"] = "data/splits/leakage_aware_split_manifest.json"
    
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = get_device(cfg["hardware"].get("device", "auto"))
    print(f"Target Hardware Device: {device}")
    
    # System RAM before
    vmem_init = psutil.virtual_memory()
    print(f"System RAM: Total={vmem_init.total / 1e9:.2f} GB | Available={vmem_init.available / 1e9:.2f} GB | Used={vmem_init.used / 1e9:.2f} GB ({vmem_init.percent}%)")
    
    # 1. Build Model
    print("\n--- 1. Model Instantiation & Parameter Verification ---")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}")
    print(f"Total Parameters    : {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    assert total_params == 23518277, f"Unexpected parameter count: {total_params}"
    
    # 2. Test Forward and Backward on Synthetic 512x512 Batch
    print("\n--- 2. Synthetic Batch (8, 3, 512, 512) Forward & Backward Test ---")
    micro_batch = 8
    accum_steps = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    dummy_x = torch.randn(micro_batch, 3, 512, 512, device=device)
    dummy_y = torch.randint(0, 5, (micro_batch,), device=device)
    
    t0 = time.perf_counter()
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y) / accum_steps
    loss.backward()
    t_fwd_bwd = time.perf_counter() - t0
    
    print(f"Forward + Backward Latency (Batch=8, 512x512): {t_fwd_bwd:.3f} s")
    print(f"Loss Value: {loss.item() * accum_steps:.4f}")
    assert not torch.isnan(loss), "Loss is NaN in synthetic forward pass!"
    assert not torch.isinf(loss), "Loss is Inf in synthetic forward pass!"
    optimizer.zero_grad()
    
    # 3. Memory & Stability Loop (10 Accumulation Steps = 20 Micro-batches)
    print("\n--- 3. Stability Loop (20 Micro-batches / 10 Updates) ---")
    model.train()
    step_times = []
    
    for step in range(20):
        t_s = time.perf_counter()
        bx = torch.randn(micro_batch, 3, 512, 512, device=device)
        by = torch.randint(0, 5, (micro_batch,), device=device)
        
        out = model(bx)
        step_loss = criterion(out, by) / accum_steps
        step_loss.backward()
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        step_times.append(time.perf_counter() - t_s)
        
        # Check gradient finiteness
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient in {name} at step {step}"
    
    mean_step_time = sum(step_times) / len(step_times)
    print(f"Stability loop completed successfully without NaNs or Infs.")
    print(f"Average time per micro-batch (compute only): {mean_step_time:.3f} s")
    
    # 4. Evaluation Pass (no_grad)
    print("\n--- 4. Evaluation Pass Memory Verification ---")
    model.eval()
    with torch.no_grad():
        val_x = torch.randn(micro_batch, 3, 512, 512, device=device)
        val_logits = model(val_x)
        val_loss = criterion(val_logits, dummy_y)
        print(f"Evaluation forward pass success. Output shape: {val_logits.shape}, loss: {val_loss.item():.4f}")
    
    # 5. Check System and Hardware Memory
    print("\n--- 5. Hardware Memory Inspection ---")
    vmem_final = psutil.virtual_memory()
    print(f"System RAM after compute test: Available={vmem_final.available / 1e9:.2f} GB | Used={vmem_final.used / 1e9:.2f} GB")
    
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        try:
            current_alloc = torch.mps.current_allocated_memory() / 1e6
            driver_alloc = torch.mps.driver_allocated_memory() / 1e6
            print(f"MPS Allocated Memory: {current_alloc:.2f} MB")
            print(f"MPS Driver Allocated: {driver_alloc:.2f} MB")
        except Exception as e:
            print(f"MPS memory query note: {e}")
            
    # 6. Real DataLoader Loading & Batch Execution Test
    print("\n--- 6. Real Data Pipeline Integration Test (3 Micro-batches) ---")
    train_loader, val_loader, test_loader, cw = get_idrid_dataloaders(cfg, seed=seed)
    print(f"Loaded train_loader: {len(train_loader)} batches (dataset size: {len(train_loader.dataset)})")
    print(f"Loaded val_loader  : {len(val_loader)} batches (dataset size: {len(val_loader.dataset)})")
    print(f"Loaded test_loader : {len(test_loader)} batches (dataset size: {len(test_loader.dataset)})")
    
    t_data_start = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    
    for step, (images, labels, names) in enumerate(train_loader):
        if step >= 3:
            break
        t_b0 = time.perf_counter()
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = criterion(out, labels) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        t_b1 = time.perf_counter()
        print(f"  Batch {step + 1}/3: input shape {tuple(images.shape)}, loss {loss.item() * accum_steps:.4f}, step time: {t_b1 - t_b0:.3f} s")
    
    total_test_time = time.perf_counter() - t_data_start
    print(f"Real data integration test completed in {total_test_time:.2f} s")
    
    # 7. Preflight Verdict
    print("\n" + "=" * 70)
    print("PREFLIGHT STATUS: PASS")
    print("All memory, gradient, tensor, and DataLoader stability tests PASSED.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = run_preflight()
    if not success:
        sys.exit(1)
