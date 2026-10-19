# IDRiD Dataset Inspection & Integrity Report

**Date**: 2026-10-15
**Hardware Acceleration**: Apple Silicon MPS (Metal Performance Shaders)
**Python**: 3.9.6 | **PyTorch**: 2.2.2

## 1. Executive Summary
- **Total Dataset Size**: 516 retinal fundus images
- **Official Training Set**: 413 images
- **Official Testing Set**: 103 images
- **Native Dimensions**: 4288x2848 RGB JPEG
- **Corrupted / Invalid Files**: 0 found
- **Duplicate Files**: 0 found
- **Missing / Orphan Images**: 0 found

## 2. Diabetic Retinopathy Severity Distribution (5-Class)

| Grade | Severity Description | Train Count (%) | Test Count (%) | Combined Total (%) |
|---|---|---|---|---|
| 0 | No DR | 134 (32.4%) | 34 (33.0%) | 168 (32.6%) |
| 1 | Mild NPDR | 20 (4.8%) | 5 (4.9%) | 25 (4.8%) |
| 2 | Moderate NPDR | 136 (32.9%) | 32 (31.1%) | 168 (32.6%) |
| 3 | Severe NPDR | 74 (17.9%) | 19 (18.4%) | 93 (18.0%) |
| 4 | Proliferative DR | 49 (11.9%) | 13 (12.6%) | 62 (12.0%) |
| **Total** | | **413** | **103** | **516** |

## 3. Binary Referable DR Distribution

- **Threshold**: Grade >= 2 defines Referable DR (Moderate NPDR, Severe NPDR, PDR).
- **Grade 0-1 (Non-Referable)**: Train: 154 (37.3%), Test: 39 (37.9%)
- **Grade 2-4 (Referable)**: Train: 259 (62.7%), Test: 64 (62.1%)

## 4. Key Methodological Observations & Constraints
1. **Severe Imbalance in Mild NPDR (Grade 1)**: Only 20 samples in Train (4.8%) and 5 samples in Test (4.9%). Cost-sensitive cross-entropy (class weighting) and stratified sampling are essential.
2. **High Native Resolution vs Model Input**: Native images are 4288 × 2848. Resizing to 256 × 256 requires careful interpolation, aspect preservation / circular mask cropping, and fundus-specific contrast enhancement (CLAHE).
3. **Separation of Test Set**: The 103 test samples are the official challenge holdout and must remain strictly untouched until final evaluation. Training set (413 images) will be split via Stratified 5-Fold or Stratified 80/20 for validation and model selection.
4. **Clinical Distinction**: `Risk of macular edema` is an independent diabetic macular edema (DME) risk label and is deliberately segregated from the retinopathy grade classification task.
