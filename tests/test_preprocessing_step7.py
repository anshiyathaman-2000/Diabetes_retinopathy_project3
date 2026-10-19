"""
Unit and Verification Tests for Step 7: Retinal Image Preprocessing Comparison.

Verifies:
1. Preprocessing method correctness: 'original', 'standard_clahe', and 'clahe' (LAB).
2. Distinct mathematical behaviors of standard RGB CLAHE vs LAB luminance CLAHE vs Original.
3. Strict absence of data leakage: fixed normalization constants, train-only class weights, disjoint splits.
4. Evaluation transforms are strictly deterministic (no stochastic data augmentation on val/test).
5. Metrics calculation consistency (Accuracy, Macro F1, QWK, per-class Recall across Grades 0-4).
6. Offline cache directories completeness for all three preprocessing configurations.
"""

import unittest
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    get_train_transforms,
    get_eval_transforms,
    DR_CLASSES_5
)
from src.preprocessing import (
    preprocess_fundus_image,
    apply_standard_clahe_rgb,
    apply_clahe_lab,
    auto_crop_fundus
)
from src.utils import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestPreprocessingStep7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_manifest_path = PROJECT_ROOT / "data/splits/leakage_aware_split_manifest.json"
        cls.train_csv = PROJECT_ROOT / "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        cls.test_csv = PROJECT_ROOT / "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        cls.cache_root = PROJECT_ROOT / "data/cache"

    def test_preprocessing_output_format_and_range(self):
        """Verifies that all three preprocessing pipelines produce valid RGB PIL Images of specified dimension."""
        dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Create a circle to mimic fundus FOV
        cv2 = __import__("cv2")
        cv2.circle(dummy_img, (200, 200), 150, (180, 80, 50), -1)

        for method in ["original", "standard_clahe", "clahe"]:
            out_img = preprocess_fundus_image(
                image_input=dummy_img,
                image_size=256,
                method=method,
                crop_border=True
            )
            self.assertIsInstance(out_img, Image.Image)
            self.assertEqual(out_img.size, (256, 256))
            self.assertEqual(out_img.mode, "RGB")

            arr = np.array(out_img)
            self.assertEqual(arr.shape, (256, 256, 3))
            self.assertGreaterEqual(arr.min(), 0)
            self.assertLessEqual(arr.max(), 255)

    def test_mathematical_contrast_enhancement_difference(self):
        """Verifies mathematical differences between Original, Standard RGB CLAHE, and LAB Luminance CLAHE."""
        # Create synthetic image with varying luminance gradient
        x = np.linspace(30, 220, 200, dtype=np.uint8)
        y = np.linspace(30, 220, 200, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        test_img = np.stack([xx, yy // 2, (xx + yy) // 3], axis=-1)

        orig = test_img.copy()
        rgb_clahe = apply_standard_clahe_rgb(test_img.copy(), clip_limit=2.0)
        lab_clahe = apply_clahe_lab(test_img.copy(), clip_limit=2.0)

        # 1. CLAHE outputs should differ from original
        self.assertFalse(np.array_equal(orig, rgb_clahe))
        self.assertFalse(np.array_equal(orig, lab_clahe))

        # 2. Standard RGB CLAHE modifies channels independently, while LAB modifies L channel
        # Therefore RGB CLAHE and LAB CLAHE must produce distinct numeric outputs
        self.assertFalse(np.array_equal(rgb_clahe, lab_clahe))

        # Check mean absolute differences
        diff_orig_vs_rgb = np.mean(np.abs(orig.astype(float) - rgb_clahe.astype(float)))
        diff_orig_vs_lab = np.mean(np.abs(orig.astype(float) - lab_clahe.astype(float)))
        diff_rgb_vs_lab = np.mean(np.abs(rgb_clahe.astype(float) - lab_clahe.astype(float)))

        self.assertGreater(diff_orig_vs_rgb, 1.0)
        self.assertGreater(diff_orig_vs_lab, 1.0)
        self.assertGreater(diff_rgb_vs_lab, 1.0)

    def test_data_partitions_leakage_safety(self):
        """Verifies strictly disjoint partitions and absence of data leakage into train statistics."""
        self.assertTrue(self.split_manifest_path.exists())
        with open(self.split_manifest_path, "r") as f:
            manifest = json.load(f)

        train_ids = set(manifest["train"])
        val_ids = set(manifest["val"])
        self.assertEqual(len(train_ids), 330)
        self.assertEqual(len(val_ids), 83)
        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Train and Val partitions must be mutually disjoint")

        # Test official
        df_test = load_and_clean_csv(self.test_csv)
        self.assertEqual(len(df_test), 103)

        # Clean test
        df_test_clean = df_test[df_test["Image name"] != "IDRiD_064"]
        self.assertEqual(len(df_test_clean), 102)

        # Class weights must be computed strictly on train set
        df_train = load_and_clean_csv(self.train_csv)
        train_labels = df_train[df_train["Image name"].isin(train_ids)]["Retinopathy grade"].tolist()
        self.assertEqual(len(train_labels), 330)

        weights = compute_class_weights(train_labels, num_classes=5, method="sqrt")
        self.assertEqual(weights.shape[0], 5)
        self.assertTrue(torch.all(weights > 0))

    def test_eval_transforms_are_strictly_deterministic(self):
        """Verifies validation and test transforms have no stochastic augmentation."""
        eval_tf = get_eval_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        test_pil = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        tensor1 = eval_tf(test_pil)
        tensor2 = eval_tf(test_pil)
        self.assertTrue(torch.equal(tensor1, tensor2), "Evaluation transform must be bitwise deterministic")

    def test_metrics_calculation_consistency(self):
        """Verifies accuracy, macro F1, QWK, and per-class recall computation."""
        y_true = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4, 0, 0, 2, 3, 4] # One error on class 1
        probs = np.eye(5)[[0, 1, 2, 3, 4, 0, 0, 2, 3, 4]]

        metrics = calculate_metrics(y_true, y_pred, y_probs=probs, task="five_class", class_names=DR_CLASSES_5)

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("quadratic_weighted_kappa", metrics)
        self.assertIn("per_class_metrics", metrics)

        self.assertEqual(metrics["accuracy"], 0.9)
        self.assertAlmostEqual(metrics["per_class_metrics"]["No DR"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_class_metrics"]["Mild NPDR"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class_metrics"]["Moderate NPDR"]["recall"], 1.0)

    def test_cache_directories_completeness(self):
        """Verifies offline caches for all three configurations exist and contain 413 train, 103 test images."""
        for method in ["original", "standard_clahe", "clahe"]:
            c_dir = self.cache_root / f"cache_256_{method}"
            self.assertTrue(c_dir.exists(), f"Cache directory missing: {c_dir}")
            train_imgs = list((c_dir / "train").glob("*.png"))
            test_imgs = list((c_dir / "test").glob("*.png"))
            self.assertEqual(len(train_imgs), 413, f"Incomplete train cache for {method}: got {len(train_imgs)}")
            self.assertEqual(len(test_imgs), 103, f"Incomplete test cache for {method}: got {len(test_imgs)}")

if __name__ == "__main__":
    unittest.main()
