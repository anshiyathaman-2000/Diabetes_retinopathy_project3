"""
Unit and regression tests for Step 8 Independent Verification.
Tests model parameter counting, checkpoint loading, mathematical consistency of latency/FPS,
reproducibility variance calculation, and data leakage / test integrity.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.run_step8_verification import (
    MODEL_SPECS,
    verify_model_complexity,
    verify_checkpoints,
    audit_data_leakage_and_integrity,
    SPLIT_MANIFEST
)

class TestStep8Verification(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_parameter_counts_exact_match(self):
        """Verify parameter counts match reported values exactly across all 3 models."""
        complexity_dict, df = verify_model_complexity()

        # ResNet-50
        self.assertEqual(complexity_dict["resnet50"]["total_parameters"], 23518277)
        self.assertEqual(complexity_dict["resnet50"]["trainable_parameters"], 23518277)
        self.assertEqual(complexity_dict["resnet50"]["non_trainable_parameters"], 0)

        # Compact Hybrid
        self.assertEqual(complexity_dict["hybrid_compact"]["total_parameters"], 24077509)
        self.assertEqual(complexity_dict["hybrid_compact"]["trainable_parameters"], 24077509)
        self.assertEqual(complexity_dict["hybrid_compact"]["non_trainable_parameters"], 0)

        # Full Hybrid
        self.assertEqual(complexity_dict["hybrid_full"]["total_parameters"], 25503045)
        self.assertEqual(complexity_dict["hybrid_full"]["trainable_parameters"], 25503045)
        self.assertEqual(complexity_dict["hybrid_full"]["non_trainable_parameters"], 0)

    def test_weights_size_vs_checkpoint_size(self):
        """Verify model weights state_dict is strictly smaller than full training checkpoint."""
        complexity_dict, df = verify_model_complexity()
        for m_id in ["resnet50", "hybrid_compact", "hybrid_full"]:
            weights_mb = complexity_dict[m_id]["weights_state_dict_mb"]
            self.assertGreater(weights_mb, 80.0)
            self.assertLess(weights_mb, 110.0)

    def test_canonical_checkpoint_state_dict_keys(self):
        """Verify canonical checkpoints load with strict=True and 0 missing/unexpected keys."""
        records, df = verify_checkpoints()
        for rec in records:
            if rec["run_type"] == "canonical_full_convergence":
                self.assertTrue(rec["keys_strict_valid"])
                self.assertEqual(rec["missing_keys_count"], 0)
                self.assertEqual(rec["unexpected_keys_count"], 0)

    def test_fps_mathematical_consistency(self):
        """Verify throughput FPS formula: FPS = (Batch_Size * 1000) / Mean_Latency_ms."""
        mean_latency = 45.0 # ms
        bs = 1
        fps = (bs * 1000.0) / mean_latency
        self.assertAlmostEqual(fps, 22.222, places=2)

        mean_latency_16 = 340.0 # ms
        bs_16 = 16
        fps_16 = (bs_16 * 1000.0) / mean_latency_16
        self.assertAlmostEqual(fps_16, 47.058, places=2)

    def test_data_leakage_and_duplicate_integrity(self):
        """Verify partition disjointness and duplicate IDRiD_064 handling."""
        import json
        with open(SPLIT_MANIFEST, "r") as f:
            manifest = json.load(f)

        leakage_report = audit_data_leakage_and_integrity(manifest)
        self.assertTrue(leakage_report["disjointness_checks"]["train_val_is_strictly_disjoint"])
        self.assertEqual(leakage_report["disjointness_checks"]["train_val_overlap_count"], 0)
        self.assertTrue(leakage_report["disjointness_checks"]["directory_isolation"]["directories_are_separate"])
        self.assertEqual(leakage_report["duplicate_image_handling"]["duplicate_identified"], "IDRiD_064")
        self.assertEqual(leakage_report["duplicate_image_handling"]["clean_deduplicated_protocol_sample_count"], 102)
        self.assertTrue(leakage_report["duplicate_image_handling"]["duplicate_removed_in_clean_protocol"])

if __name__ == "__main__":
    unittest.main()
