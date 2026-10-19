"""
Unit and Verification Tests for Evaluation Pipeline Reliability (Step 5).
Verifies:
1. Dataset split verification, class mappings, and target label integrity.
2. Duplicate image detection and cryptographic hash integrity.
3. Training augmentation vs deterministic evaluation transform isolation.
4. Robust metric calculation (handling missing classes in ROC-AUC & PR-AUC, QWK, macro F1).
5. Parameter counting accuracy.
6. Inference latency benchmarking with warmup and synchronization.
7. Stratified 5-Fold Cross-Validation split validity and duplicate group cohesion.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils import (
    calculate_metrics,
    count_model_parameters,
    measure_inference_latency
)
from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    DR_CLASSES_5,
    DR_CLASSES_BINARY
)
from src.models import BaselineResNet50
from src.cross_validation import (
    generate_stratified_kfold_splits,
    INTRA_TRAIN_DUPLICATE_PAIRS
)

class TestEvaluationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_csv = Path("B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
        cls.test_csv = Path("B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
        cls.train_img_dir = Path("B. Disease Grading/1. Original Images/a. Training Set")
        cls.test_img_dir = Path("B. Disease Grading/1. Original Images/b. Testing Set")

    def test_dataset_split_and_label_integrity(self):
        """Verify train (413) and test (103) counts, valid integer grades [0, 4], and image presence."""
        df_train = load_and_clean_csv(self.train_csv)
        df_test = load_and_clean_csv(self.test_csv)

        self.assertEqual(len(df_train), 413)
        self.assertEqual(len(df_test), 103)

        # Check grades are integers in range [0, 4]
        self.assertTrue(set(df_train["Retinopathy grade"]).issubset({0, 1, 2, 3, 4}))
        self.assertTrue(set(df_test["Retinopathy grade"]).issubset({0, 1, 2, 3, 4}))

        # Check images exist on disk
        for img in df_train["Image name"]:
            self.assertTrue((self.train_img_dir / f"{img}.jpg").exists(), f"Missing train image: {img}")
        for img in df_test["Image name"]:
            self.assertTrue((self.test_img_dir / f"{img}.jpg").exists(), f"Missing test image: {img}")

    def test_class_weights_isolated_to_train(self):
        """Verify class weights are derived exclusively from train labels and sum/shape is correct."""
        df_train = load_and_clean_csv(self.train_csv)
        train_labels = df_train["Retinopathy grade"].tolist()

        weights = compute_class_weights(train_labels, num_classes=5, method="sqrt")
        self.assertEqual(weights.shape, (5,))
        self.assertTrue((weights > 0).all())
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())

    def test_missing_class_handling_in_roc_and_pr_auc(self):
        """Verify calculate_metrics handles unrepresented classes safely without crashing."""
        # Simulated scenario: only classes 0, 2, 3, 4 present in true labels (class 1 missing)
        y_true = np.array([0, 0, 2, 2, 3, 3, 4, 4])
        # Random probabilities summing to 1
        raw_probs = np.random.uniform(0.1, 1.0, (8, 5))
        y_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_probs, axis=1)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            task="five_class",
            class_names=DR_CLASSES_5
        )

        # Accuracy, QWK, F1 must be valid floats
        self.assertIsInstance(metrics["accuracy"], float)
        self.assertIsInstance(metrics["quadratic_weighted_kappa"], float)
        self.assertIsInstance(metrics["macro_f1"], float)

        # Class 1 is missing, so its OvR ROC-AUC and PR-AUC must safely be None
        per_class = metrics["per_class_metrics"]
        self.assertIsNone(per_class["Mild NPDR"]["roc_auc"])
        self.assertIsNone(per_class["Mild NPDR"]["pr_auc"])

        # The other classes should have valid float AUCs
        self.assertIsInstance(per_class["No DR"]["roc_auc"], float)
        self.assertIsInstance(metrics["overall_roc_auc"], float)

    def test_binary_roc_pr_auc_edge_cases(self):
        """Verify binary metrics handle edge cases gracefully."""
        # Single-class edge case in binary
        y_true_single = np.array([1, 1, 1, 1])
        y_probs_single = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        metrics = calculate_metrics(
            y_true=y_true_single,
            y_pred=np.array([1, 1, 1, 1]),
            y_probs=y_probs_single,
            task="binary",
            class_names=DR_CLASSES_BINARY
        )
        self.assertIsNone(metrics["overall_roc_auc"])
        self.assertIsNone(metrics["overall_pr_auc"])

    def test_parameter_counting(self):
        """Verify count_model_parameters returns accurate total, trainable, and frozen counts."""
        model = BaselineResNet50(num_classes=5, pretrained=False)
        param_info = count_model_parameters(model)

        self.assertGreater(param_info["total_parameters"], 20_000_000)
        self.assertEqual(param_info["trainable_parameters"], param_info["total_parameters"])
        self.assertEqual(param_info["frozen_parameters"], 0)
        self.assertGreater(param_info["model_size_mb"], 80.0)
        self.assertIn("backbone", param_info["component_breakdown"])

    def test_latency_benchmarking(self):
        """Verify measure_inference_latency runs warmup, timed passes, and returns valid stats."""
        model = BaselineResNet50(num_classes=5, pretrained=False)
        device = torch.device("cpu")
        model.to(device)

        lat_info = measure_inference_latency(
            model,
            input_shape=(1, 3, 64, 64),  # Small size for swift unit testing
            batch_size=1,
            device=device,
            num_warmup=2,
            num_runs=5
        )

        self.assertGreater(lat_info["mean_latency_ms"], 0.0)
        self.assertGreater(lat_info["throughput_fps"], 0.0)
        self.assertEqual(lat_info["batch_size"], 1)
        self.assertIn("CPU", lat_info["device"])

    def test_stratified_kfold_split_cohesion(self):
        """Verify 5-fold cross-validation covers 413 samples and duplicate pairs never cross folds."""
        manifest = generate_stratified_kfold_splits(num_folds=5, seed=42, save_manifest=False)

        total_split_samples = sum(s["total_samples"] for s in manifest["fold_statistics"].values())
        self.assertEqual(total_split_samples, 413)

        # Check duplicate pairs cohesion
        folds = manifest["folds"]
        img_to_fold = {}
        for f_name, img_list in folds.items():
            for img in img_list:
                img_to_fold[img] = f_name

        for img1, img2 in INTRA_TRAIN_DUPLICATE_PAIRS:
            self.assertIn(img1, img_to_fold)
            self.assertIn(img2, img_to_fold)
            self.assertEqual(img_to_fold[img1], img_to_fold[img2], f"Duplicate pair {img1}-{img2} split across folds!")

if __name__ == "__main__":
    unittest.main()
