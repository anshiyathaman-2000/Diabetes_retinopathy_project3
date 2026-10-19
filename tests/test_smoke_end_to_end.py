"""
End-to-End Smoke Test for Step 3 Final Pipeline.
Verifies data loading, forward passes, loss calculation, checkpoint loading, and evaluation flow.
"""

import unittest
from pathlib import Path
import torch
import torch.nn as nn
from src.utils import load_config, get_device, FocalLoss, calculate_metrics
from src.models import build_model
from src.dataset import get_idrid_dataloaders

class TestSmokeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/experiment.yaml")
        cls.config["training"]["batch_size"] = 2
        cls.config["data"]["num_workers"] = 0
        cls.device = torch.device("cpu")

    def test_end_to_end_forward_loss_and_metrics(self):
        # 1. DataLoader
        train_loader, val_loader, test_loader, class_weights = get_idrid_dataloaders(self.config)
        images, labels, names = next(iter(val_loader))
        self.assertEqual(images.shape, (2, 3, 256, 256))
        self.assertEqual(len(labels), 2)

        # 2. Model Forward
        model = build_model(self.config).to(self.device)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        self.assertEqual(logits.shape, (2, 5))

        # 3. Loss Calculation
        criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
        criterion_focal = FocalLoss(alpha=class_weights, gamma=2.0)
        loss_ce = criterion_ce(logits, labels)
        loss_focal = criterion_focal(logits, labels)
        self.assertFalse(torch.isnan(loss_ce))
        self.assertFalse(torch.isnan(loss_focal))

        # 4. Metric Computation
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        metrics = calculate_metrics(labels.numpy(), preds, y_probs=probs, task="five_class")
        self.assertIn("accuracy", metrics)
        self.assertIn("quadratic_weighted_kappa", metrics)
        self.assertIn("macro_f1", metrics)

    def test_checkpoint_loading_smoke(self):
        ckpt_path = Path("checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt")
        self.assertTrue(ckpt_path.exists())
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("config", ckpt)
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
