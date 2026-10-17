# Diabetes_retinopathy_project3

Diabetes Retinopathy Detection is a deep learning project that analyzes retinal fundus images to detect and classify diabetic retinopathy. It uses image preprocessing and CNN-based models to support early screening and help prevent vision loss.

---

## 📌 Project Overview

This repository provides an end-to-end framework for diabetic retinopathy (DR) grading and early screening using color fundus photography. The codebase incorporates:
- **Comprehensive Image Preprocessing**: Ben Graham's circular cropping, CLAHE (Contrast Limited Adaptive Histogram Equalization), noise reduction, and standard retinal normalizations.
- **Deep Learning Architectures**:
  - ResNet-50 baseline and compact backbones.
  - Hybrid spatial-sequential models (CNN + BiLSTM, CNN + Transformer).
- **Class Imbalance Solutions**: Effective handling of rare severe stages via Square-Root / Inverse Class Weights, Focal Loss, and Weighted Random Sampling.
- **Explainability & Trust**: Grad-CAM visual heatmaps highlighting clinically significant lesions (microaneurysms, hemorrhages, hard/soft exudates).
- **Robust Evaluation & Auditing**: Five-class grading and binary referable DR evaluation with metrics including Quadratic Weighted Kappa (QWK), Macro ROC-AUC, Sensitivity, Specificity, and per-class precision/recall.

---

## 📂 Repository Structure

```text
├── configs/
│   └── experiment.yaml             # Hyperparameters, data paths, and training configuration
├── scripts/
│   ├── run_baseline.sh             # Training script for ResNet-50 baseline
│   ├── run_hybrid.sh               # Training script for Hybrid models
│   └── run_imbalance_experiments.py # Script for class imbalance comparison
├── src/
│   ├── dataset.py                  # PyTorch Dataset, dataloaders, and augmentations
│   ├── preprocessing.py            # CLAHE, circular crop, and fundus image cleaning
│   ├── models.py                   # Model architectures (ResNet-50, Hybrid, Compact)
│   ├── train.py                    # Training and validation loops with checkpointing
│   ├── evaluate.py                 # Multi-metric evaluation (AUC, QWK, confusion matrices)
│   ├── explainability.py           # Grad-CAM heatmap generation and overlays
│   ├── cross_validation.py         # Stratified K-fold cross-validation routines
│   ├── inspect_dataset.py          # Data distribution audit and duplicate checks
│   ├── run_step5_evaluation_audit.py
│   ├── run_step6_early_detection.py
│   ├── run_step7_preprocessing_comparison.py
│   ├── run_step8_efficiency_reproducibility.py
│   └── utils.py                    # Metric computations, logging, and seed reproducibility
├── tests/                          # End-to-end and unit tests suite
├── requirements.txt                # Python package dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/anshiyathaman-2000/Diabetes_retinopathy_project3.git
cd Diabetes_retinopathy_project3
pip install -r requirements.txt
```

### 2. Dataset Setup

Prepare the dataset (such as IDRiD or Kaggle EyePACS) and organize the directory structure or configure paths in `configs/experiment.yaml`:
- Disease Grading: Images and corresponding CSV labels (Severity levels 0–4).

---

## 🧪 Usage

### Model Training

Run the baseline ResNet-50 training:
```bash
bash scripts/run_baseline.sh
```

Run hybrid CNN-Transformer / BiLSTM training:
```bash
bash scripts/run_hybrid.sh
```

Run class imbalance ablation studies:
```bash
python scripts/run_imbalance_experiments.py
```

### Evaluation & Auditing

Evaluate checkpoints across test sets:
```bash
python src/evaluate.py --config configs/experiment.yaml
```

Run independent verification audits:
```bash
python src/run_step5_evaluation_audit.py
```

### Explainability (Grad-CAM)

Generate visual explanations for model predictions:
```bash
python src/generate_step4_gradcam.py
```

### Running Tests

Run the full automated test suite:
```bash
pytest tests/
```

---

## 📊 Key Highlights & Results

- **Referable DR Screening**: Binary classification (No DR/Mild vs. Moderate/Severe/Proliferative) achieving high clinical sensitivity.
- **Five-Class Severity Grading**: Detailed classification across No DR (0), Mild (1), Moderate (2), Severe (3), and Proliferative DR (4).
- **Explainability**: Integrated Grad-CAM lesion localization to verify clinical decision-making.

---

## 📄 License & Attribution

This project is developed for diabetic retinopathy detection research and clinical decision support analysis.
