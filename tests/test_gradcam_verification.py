"""
Unit and verification tests for Grad-CAM Explainability Implementation (Step 4).
Verifies hook registration and cleanup, raw logit targeting, numerical safety,
spatial alignment, and checkpoint compatibility on IDRiD models.
"""

import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.models import BaselineResNet50, CNNTransformerBiLSTM
from src.explainability import (
    GradCAM,
    overlay_cam_on_image,
    generate_explanation_figure
)

class TestGradCAMVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.dummy_input = torch.randn(1, 3, 256, 256, device=cls.device)
        cls.num_classes = 5

        # Checkpoints for verification
        cls.resnet_focal_ckpt = Path("checkpoints/resnet50_five_class_clahe_focal_s42_s42_best.pt")
        cls.hybrid_compact_ckpt = Path("checkpoints/hybrid_five_class_clahe_compact_sqrt_cw_s42_best.pt")
        cls.resnet_baseline_ckpt = Path("checkpoints/resnet50_five_class_clahe_sqrt_cw_s42_best.pt")

    def test_target_layer_identification_resnet50(self):
        """Verify target layer for ResNet-50 is the final Bottleneck block in layer4."""
        model = BaselineResNet50(num_classes=self.num_classes, pretrained=False)
        target_layer = model.get_cam_target_layer()
        
        self.assertIsNotNone(target_layer)
        self.assertEqual(target_layer, model.backbone.layer4[-1])

    def test_target_layer_identification_hybrid(self):
        """Verify target layer for Hybrid model is layer4[-1] before transformer/BiLSTM."""
        model = CNNTransformerBiLSTM(
            num_classes=self.num_classes,
            pretrained=False,
            embed_dim=128,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ff_dim=256,
            bilstm_hidden_dim=64
        )
        target_layer = model.get_cam_target_layer()
        
        self.assertIsNotNone(target_layer)
        self.assertEqual(target_layer, model.layer4[-1])

    def test_hook_registration_and_cleanup(self):
        """Confirm hooks attach correctly and are cleanly removed after context exit."""
        model = BaselineResNet50(num_classes=self.num_classes, pretrained=False)
        target_layer = model.get_cam_target_layer()

        # Initial hook count should be 0
        self.assertEqual(len(target_layer._forward_hooks), 0)
        self.assertEqual(len(target_layer._backward_hooks), 0)

        with GradCAM(model, target_layer) as cam:
            # Hooks should now be attached
            self.assertEqual(len(target_layer._forward_hooks), 1)
            self.assertEqual(len(target_layer._backward_hooks), 1)
            
            # Run Grad-CAM
            heatmap, pred_class, conf, probs, info = cam.generate_heatmap(self.dummy_input, target_class=2)
            self.assertFalse(info["has_nan"])
            self.assertFalse(info["has_inf"])

        # Hooks must be completely unregistered after exiting context
        self.assertEqual(len(target_layer._forward_hooks), 0)
        self.assertEqual(len(target_layer._backward_hooks), 0)

    def test_raw_logit_targeting_and_gradient_flow(self):
        """Confirm Grad-CAM computes gradients using pre-softmax logits for the target class."""
        model = BaselineResNet50(num_classes=self.num_classes, pretrained=False)
        target_layer = model.get_cam_target_layer()

        with GradCAM(model, target_layer) as cam:
            # Test targeting specific class
            for target_class in [0, 3]:
                heatmap, pred_class, conf, probs, info = cam.generate_heatmap(self.dummy_input, target_class=target_class)
                self.assertEqual(info["target_class"], target_class)
                self.assertIsNotNone(cam.gradients)
                self.assertIsNotNone(cam.activations)
                self.assertEqual(cam.activations.shape[0], 1)
                self.assertEqual(cam.activations.shape[1], 2048)  # ResNet layer4 channels
                self.assertEqual(cam.gradients.shape, cam.activations.shape)

    def test_numerical_validity_and_bounds(self):
        """Verify heatmaps are finite, non-negative, and properly normalized to [0, 1]."""
        torch.manual_seed(42)
        model = BaselineResNet50(num_classes=self.num_classes, pretrained=False)
        target_layer = model.get_cam_target_layer()

        with GradCAM(model, target_layer) as cam:
            heatmap, pred_class, conf, probs, info = cam.generate_heatmap(self.dummy_input, target_class=None)
            
            # Check shape is 2D numpy array
            self.assertEqual(heatmap.ndim, 2)
            self.assertIsInstance(heatmap, np.ndarray)

            # Check validity flags
            self.assertFalse(info["has_nan"])
            self.assertFalse(info["has_inf"])
            self.assertFalse(info["is_empty"])
            self.assertFalse(info["is_constant"])
            
            # Heatmap range
            self.assertGreaterEqual(heatmap.min(), 0.0)
            self.assertLessEqual(heatmap.max(), 1.0)

    def test_overlay_generation_and_spatial_alignment(self):
        """Verify overlay has exact dimensions and valid uint8 RGB color range."""
        dummy_rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        dummy_cam = np.random.uniform(0.0, 1.0, (8, 8)).astype(np.float32)

        overlay = overlay_cam_on_image(dummy_rgb, dummy_cam, alpha=0.45)
        self.assertEqual(overlay.shape, (256, 256, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreaterEqual(overlay.min(), 0)
        self.assertLessEqual(overlay.max(), 255)

    def test_checkpoint_loading_and_compatibility(self):
        """Verify supported checkpoints exist and load successfully without architectural mismatch."""
        from src.models import build_model
        checkpoints = [
            self.resnet_focal_ckpt,
            self.hybrid_compact_ckpt,
            self.resnet_baseline_ckpt,
        ]

        for ckpt_path in checkpoints:
            self.assertTrue(ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("config", ckpt)
            self.assertIn("model_state_dict", ckpt)

            config = ckpt["config"]
            model = build_model(config)
            
            # Load weights strictly to ensure perfect architecture match
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()

            # Confirm Grad-CAM attaches and generates valid output with loaded weights
            target_layer = model.get_cam_target_layer()
            with GradCAM(model, target_layer) as cam:
                heatmap, pred_class, conf, probs, info = cam.generate_heatmap(self.dummy_input, target_class=0)
                self.assertEqual(heatmap.ndim, 2)
                self.assertFalse(info["has_nan"])
                self.assertFalse(info["has_inf"])

if __name__ == "__main__":
    unittest.main()
