"""
Unit tests for Baseline ResNet-50 and Proposed CNN-Transformer-BiLSTM models.
Verifies tensor dimensions, forward passes, and backward loss gradient propagation.
"""

import unittest
import torch
import torch.nn as nn
from src.models import (
    BaselineResNet50,
    CNNTransformerBiLSTM,
    build_model
)

class TestModels(unittest.TestCase):
    def setUp(self):
        self.dummy_input = torch.randn(2, 3, 256, 256)
        self.dummy_labels_5 = torch.tensor([0, 3], dtype=torch.long)
        self.dummy_labels_2 = torch.tensor([0, 1], dtype=torch.long)

    def test_baseline_resnet50_five_class(self):
        # Use pretrained=False for swift unit testing without network calls
        model = BaselineResNet50(num_classes=5, pretrained=False)
        logits = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 5))

        # Check loss and backward pass
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, self.dummy_labels_5)
        loss.backward()
        self.assertFalse(torch.isnan(loss))

    def test_baseline_resnet50_binary(self):
        model = BaselineResNet50(num_classes=2, pretrained=False)
        logits = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 2))

    def test_hybrid_model_forward_and_backward(self):
        model = CNNTransformerBiLSTM(
            num_classes=5,
            pretrained=False,
            embed_dim=128,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ff_dim=256,
            bilstm_hidden_dim=64,
            use_transformer=True,
            use_bilstm=True
        )
        logits = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 5))

        # Backward gradient flow check
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, self.dummy_labels_5)
        loss.backward()
        self.assertFalse(torch.isnan(loss))

    def test_hybrid_ablation_variants(self):
        # Variant 1: No BiLSTM
        model_no_lstm = CNNTransformerBiLSTM(
            num_classes=5,
            pretrained=False,
            embed_dim=128,
            use_transformer=True,
            use_bilstm=False
        )
        logits1 = model_no_lstm(self.dummy_input)
        self.assertEqual(logits1.shape, (2, 5))

        # Variant 2: No Transformer
        model_no_trans = CNNTransformerBiLSTM(
            num_classes=5,
            pretrained=False,
            embed_dim=128,
            use_transformer=False,
            use_bilstm=True
        )
        logits2 = model_no_trans(self.dummy_input)
        self.assertEqual(logits2.shape, (2, 5))

    def test_parameter_groups_and_freezing(self):
        model = CNNTransformerBiLSTM(num_classes=5, pretrained=False, embed_dim=128)
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["lr"], 1e-5)
        self.assertEqual(groups[1]["lr"], 1e-3)

        # Freezing
        model.freeze_backbone()
        for p in model.get_backbone_params():
            self.assertFalse(p.requires_grad)
        for p in model.get_head_params():
            self.assertTrue(p.requires_grad)

        # Unfreezing
        model.unfreeze_backbone()
        for p in model.get_backbone_params():
            self.assertTrue(p.requires_grad)

    def test_2d_sincos_and_mean_pooling(self):
        from src.models import build_2d_sincos_position_embedding
        pos = build_2d_sincos_position_embedding(8, 8, 128)
        self.assertEqual(pos.shape, (1, 64, 128))

        model_mean = CNNTransformerBiLSTM(
            num_classes=5,
            pretrained=False,
            embed_dim=128,
            pool_type="mean"
        )
        logits = model_mean(self.dummy_input)
        self.assertEqual(logits.shape, (2, 5))

    def test_grad_cam_target_layers(self):
        model_base = BaselineResNet50(num_classes=5, pretrained=False)
        self.assertIsNotNone(model_base.get_cam_target_layer())

        model_hybrid = CNNTransformerBiLSTM(num_classes=5, pretrained=False)
        self.assertIsNotNone(model_hybrid.get_cam_target_layer())

if __name__ == "__main__":
    unittest.main()
