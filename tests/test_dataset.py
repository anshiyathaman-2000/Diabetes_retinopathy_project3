"""
Unit tests for IDRiD Dataset, Preprocessing, and DataLoader pipelines.
"""

import unittest
from pathlib import Path
import torch
from src.utils import load_config
from src.dataset import (
    load_and_clean_csv,
    compute_class_weights,
    get_idrid_dataloaders,
    IDRiDDataset
)

class TestIDRiDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/experiment.yaml")
        # Ensure fast test execution with small batch and 0 workers
        cls.config["training"]["batch_size"] = 4
        cls.config["data"]["num_workers"] = 0

    def test_csv_loading(self):
        train_csv = self.config["data"]["train_csv"]
        df = load_and_clean_csv(train_csv)
        self.assertEqual(len(df), 413)
        self.assertIn("Image name", df.columns)
        self.assertIn("Retinopathy grade", df.columns)
        # Check no unnamed columns
        self.assertFalse(any("Unnamed" in col for col in df.columns))

    def test_class_weights_calculation(self):
        labels = [0, 0, 0, 1, 2, 2, 3, 4]
        weights = compute_class_weights(labels, num_classes=5)
        self.assertEqual(len(weights), 5)
        self.assertTrue(torch.all(weights > 0))
        # Minority class (1) must have higher weight than majority class (0)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_dataloader_batch_shapes(self):
        train_loader, val_loader, test_loader, class_weights = get_idrid_dataloaders(self.config)
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        self.assertGreater(len(test_loader), 0)

        # Fetch first batch from train loader
        images, labels, names = next(iter(train_loader))
        self.assertEqual(images.ndim, 4)
        self.assertEqual(images.shape[1], 3)
        self.assertEqual(images.shape[2], 256)
        self.assertEqual(images.shape[3], 256)
        self.assertEqual(len(labels), images.shape[0])
        self.assertEqual(len(names), images.shape[0])
        self.assertTrue(all(0 <= l.item() <= 4 for l in labels))

if __name__ == "__main__":
    unittest.main()
