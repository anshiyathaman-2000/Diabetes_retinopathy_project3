"""
Unit tests for duplicate audit, leakage-aware splitting, focal loss, and evaluation protocols.
Ensures rigorous verification of research deliverables and data integrity.
"""

import unittest
import json
import hashlib
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from src.utils import FocalLoss, calculate_metrics
from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    get_idrid_dataloaders
)

class TestAuditAndImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_csv = "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        cls.test_csv = "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        cls.manifest_path = "data/splits/leakage_aware_split_manifest.json"
        cls.audit_csv = "outputs/metrics/duplicate_audit_report.csv"

    def test_focal_loss_numerics_and_gradient(self):
        """Verify Focal Loss forward loss computation, focal weighting, and backward gradients."""
        batch_size, num_classes = 4, 5
        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
        alpha = torch.tensor([0.5, 4.0, 0.6, 1.1, 1.7])

        focal_criterion = FocalLoss(alpha=alpha, gamma=2.0)
        loss = focal_criterion(logits, targets)

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

        # Verify gradient flow
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_focal_loss_reductions(self):
        """Test 'mean', 'sum', and 'none' reductions in FocalLoss."""
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        criterion_mean = FocalLoss(gamma=2.0, reduction="mean")
        criterion_sum = FocalLoss(gamma=2.0, reduction="sum")
        criterion_none = FocalLoss(gamma=2.0, reduction="none")

        loss_mean = criterion_mean(logits, targets)
        loss_sum = criterion_sum(logits, targets)
        loss_none = criterion_none(logits, targets)

        self.assertEqual(loss_mean.ndim, 0)
        self.assertEqual(loss_sum.ndim, 0)
        self.assertEqual(loss_none.shape, (4,))
        self.assertAlmostEqual(loss_mean.item(), loss_none.mean().item(), places=5)
        self.assertAlmostEqual(loss_sum.item(), loss_none.sum().item(), places=5)

    def test_duplicate_audit_report_consistency(self):
        """Verify that the duplicate audit report correctly catalogs all clusters."""
        audit_file = Path(self.audit_csv)
        self.assertTrue(audit_file.exists(), "Duplicate audit report CSV must exist.")
        df_audit = pd.read_csv(audit_file)
        self.assertIn("Duplicate_Cluster_ID", df_audit.columns)
        self.assertIn("Image_Name_1", df_audit.columns)
        self.assertIn("Image_Name_2", df_audit.columns)
        self.assertIn("Split_1", df_audit.columns)
        self.assertIn("Split_2", df_audit.columns)
        self.assertIn("Grade_1", df_audit.columns)
        self.assertIn("Grade_2", df_audit.columns)
        self.assertIn("Label_Status", df_audit.columns)

        # Check known duplicate pair: IDRiD_118 (train) and IDRiD_064 (test)
        names_1 = df_audit["Image_Name_1"].tolist()
        names_2 = df_audit["Image_Name_2"].tolist()
        all_names = set(names_1 + names_2)
        self.assertIn("IDRiD_118", all_names)
        self.assertIn("IDRiD_064", all_names)

        # Check conflicting grades recorded for IDRiD_118 vs IDRiD_064
        row = df_audit[(df_audit["Image_Name_1"] == "IDRiD_118") | (df_audit["Image_Name_2"] == "IDRiD_118")].iloc[0]
        if row["Image_Name_1"] == "IDRiD_118":
            self.assertEqual(row["Grade_1"], 0)
            self.assertEqual(row["Grade_2"], 3)
        else:
            self.assertEqual(row["Grade_2"], 0)
            self.assertEqual(row["Grade_1"], 3)
        self.assertEqual(row["Label_Status"], "CONFLICTING_LABELS")

    def test_leakage_aware_split_manifest(self):
        """Verify the leakage-aware split manifest has no overlap and partitions all 413 training images."""
        manifest_file = Path(self.manifest_path)
        self.assertTrue(manifest_file.exists(), "Leakage-aware manifest must exist.")
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        train_imgs = manifest["train"]
        val_imgs = manifest["val"]

        self.assertEqual(len(train_imgs), 330)
        self.assertEqual(len(val_imgs), 83)
        self.assertEqual(len(set(train_imgs)), 330)
        self.assertEqual(len(set(val_imgs)), 83)

        # Disjointness check
        intersection = set(train_imgs).intersection(set(val_imgs))
        self.assertEqual(len(intersection), 0, f"Overlap between train and val: {intersection}")

        # Total count check
        self.assertEqual(len(train_imgs) + len(val_imgs), 413)

        # Check that duplicate pairs (IDRiD_375, IDRiD_073) and (IDRiD_333, IDRiD_045) are not split across train and val
        # In our manifest, both members of each pair were assigned to train
        self.assertIn("IDRiD_375", train_imgs)
        self.assertIn("IDRiD_073", train_imgs)
        self.assertIn("IDRiD_333", train_imgs)
        self.assertIn("IDRiD_045", train_imgs)

    def test_test_protocol_filtering(self):
        """Verify that 'deduplicated' test protocol filters IDRiD_064 (N=102) while 'official' keeps N=103."""
        cfg_official = {
            "task": {"type": "five_class", "num_classes": 5},
            "data": {
                "train_csv": self.train_csv,
                "test_csv": self.test_csv,
                "train_images_dir": "B. Disease Grading/1. Original Images/a. Training Set",
                "test_images_dir": "B. Disease Grading/1. Original Images/b. Testing Set",
                "val_split_ratio": 0.20,
                "split_type": "standard",
                "test_protocol": "official",
                "cache_dir": "data/cache",
                "num_workers": 0
            },
            "preprocessing": {
                "image_size": 256,
                "method": "clahe",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "augmentation": {}
            },
            "training": {"batch_size": 16, "class_weight_method": "inverse"}
        }

        cfg_dedup = dict(cfg_official)
        cfg_dedup["data"] = dict(cfg_official["data"])
        cfg_dedup["data"]["test_protocol"] = "deduplicated"

        _, _, test_loader_official, _ = get_idrid_dataloaders(cfg_official)
        _, _, test_loader_dedup, _ = get_idrid_dataloaders(cfg_dedup)

        self.assertEqual(len(test_loader_official.dataset), 103)
        self.assertEqual(len(test_loader_dedup.dataset), 102)

        dedup_names = test_loader_dedup.dataset.image_names
        self.assertNotIn("IDRiD_064", dedup_names)

    def test_binary_referable_dr_mapping(self):
        """Verify binary task maps Grade 0-1 to 0 (Non-Referable) and Grade 2-4 to 1 (Referable)."""
        cfg_binary = {
            "task": {"type": "binary", "num_classes": 2, "binary_threshold": 2},
            "data": {
                "train_csv": self.train_csv,
                "test_csv": self.test_csv,
                "train_images_dir": "B. Disease Grading/1. Original Images/a. Training Set",
                "test_images_dir": "B. Disease Grading/1. Original Images/b. Testing Set",
                "val_split_ratio": 0.20,
                "split_type": "standard",
                "test_protocol": "official",
                "cache_dir": "data/cache",
                "num_workers": 0
            },
            "preprocessing": {
                "image_size": 256,
                "method": "clahe",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "augmentation": {}
            },
            "training": {"batch_size": 16, "class_weight_method": "inverse"}
        }

        train_loader, _, _, class_weights = get_idrid_dataloaders(cfg_binary)
        dataset = train_loader.dataset

        self.assertEqual(dataset.num_classes, 2)
        for raw_g, mapped_l in zip(dataset.raw_grades, dataset.labels):
            if raw_g < 2:
                self.assertEqual(mapped_l, 0)
            else:
                self.assertEqual(mapped_l, 1)

if __name__ == "__main__":
    unittest.main()
