"""
Explainability Module using Grad-CAM (Gradient-Weighted Class Activation Mapping).
Generates interpretable visual heatmaps highlighting retinal regions influencing
the CNN component's diabetic retinopathy grade predictions.

IMPORTANT SCIENTIFIC DISCLAIMER:
Grad-CAM heatmaps represent computational feature attributions (where the network
focused its convolutional activation). They are diagnostic decision aids for
research and education only and DO NOT constitute proof of clinically valid pathology
nor replace formal ophthalmological evaluation by a licensed specialist.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_fundus_image, get_eval_transforms
from src.dataset import DR_CLASSES_5, DR_CLASSES_BINARY
from src.models import build_model
from src.utils import get_device

class GradCAM:
    """
    Robust Grad-CAM implementation for retinal fundus models.
    Hooks the final convolutional layer of the CNN backbone to capture
    activation feature maps and corresponding backward gradients.
    
    Includes:
    - Clean hook registration and unregistration (prevents hook leakage/memory build-up)
    - Context manager support (`with GradCAM(model, target_layer) as cam:`)
    - Vectorized linear combination (standard Grad-CAM formula: ReLU(sum_k alpha_k * A_k))
    - Comprehensive numerical health checks (NaN/Inf detection, empty map, constant map)
    - Support for raw pre-softmax logit targeting
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.forward_hook_handle = None
        self.backward_hook_handle = None
        self._register_hooks()

    def _register_hooks(self):
        self.forward_hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # Detach to avoid retaining computational graph of activations
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # Detach to avoid graph retention
        if len(grad_output) > 0 and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = None

    def remove_hooks(self):
        """Unregisters hooks to ensure no memory leaks or dangling hook side effects."""
        if self.forward_hook_handle is not None:
            self.forward_hook_handle.remove()
            self.forward_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray, Dict[str, Any]]:
        """
        Computes Grad-CAM heatmap for the input image tensor.
        
        Args:
            input_tensor: [1, C, H, W] normalized image tensor on device.
            target_class: Optional integer class to explain. If None, uses argmax(probs).
            
        Returns:
            - normalized_heatmap: [H_act, W_act] numpy array normalized to [0.0, 1.0]
            - predicted_class: integer index of predicted class
            - confidence: softmax probability of the target class
            - full_probabilities: [num_classes] numpy array
            - diagnostic_metadata: dict containing validity checks, stats, and gradient health
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass: obtain raw pre-softmax logits
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

        if target_class is None:
            target_class = pred_class

        num_classes = logits.size(1)
        if target_class < 0 or target_class >= num_classes:
            raise ValueError(f"Target class {target_class} is out of bounds for {num_classes}-class model.")

        # Target score before softmax/sigmoid transformation
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        diagnostics = {
            "target_class": target_class,
            "predicted_class": pred_class,
            "target_score_logit": float(target_score.item()),
            "confidence_prob": float(probs[target_class]),
            "activation_shape": list(self.activations.shape) if self.activations is not None else None,
            "gradient_shape": list(self.gradients.shape) if self.gradients is not None else None,
            "has_nan": False,
            "has_inf": False,
            "is_empty": False,
            "is_constant": False,
            "raw_min": 0.0,
            "raw_max": 0.0,
            "raw_mean": 0.0,
            "raw_std": 0.0
        }

        # Validate gradients
        if self.gradients is None:
            raise RuntimeError("GradCAM failed: No gradients were captured at the target convolutional layer.")

        if torch.isnan(self.gradients).any() or torch.isnan(self.activations).any():
            diagnostics["has_nan"] = True
            raise ValueError("GradCAM encountered NaN values in activations or backward gradients.")

        if torch.isinf(self.gradients).any() or torch.isinf(self.activations).any():
            diagnostics["has_inf"] = True
            raise ValueError("GradCAM encountered Inf values in activations or backward gradients.")

        # Standard Grad-CAM: Global Average Pooling of gradients per channel:
        # alpha_k = (1 / Z) * sum_{i,j} (dY^c / dA_{i,j}^k)
        # pooled_gradients shape: [B, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps: sum_k alpha_k * A^k
        # activations shape: [1, C, H, W]
        weighted_activations = torch.sum(self.activations * pooled_gradients, dim=1) # [1, H, W]

        # ReLU on weighted sum to focus strictly on features that have positive impact on the class
        cam = F.relu(weighted_activations).squeeze(0).cpu().numpy()

        diagnostics["raw_min"] = float(np.min(cam))
        diagnostics["raw_max"] = float(np.max(cam))
        diagnostics["raw_mean"] = float(np.mean(cam))
        diagnostics["raw_std"] = float(np.std(cam))

        max_val = np.max(cam)
        if max_val > 1e-8:
            heatmap = cam / max_val
        else:
            # Heatmap is all zeros (empty map)
            heatmap = np.zeros_like(cam)
            diagnostics["is_empty"] = True

        if np.std(heatmap) < 1e-6:
            diagnostics["is_constant"] = True

        confidence = float(probs[target_class])
        return heatmap, pred_class, confidence, probs, diagnostics

def overlay_cam_on_image(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Overlays normalized [0, 1] heatmap on top of retinal RGB image.
    Uses bicubic interpolation for smooth, artifact-free resolution scaling.
    """
    h, w, _ = original_rgb.shape
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=interpolation)
    # Clamp to [0, 1] after bicubic overshoot
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap_colored) * alpha + np.float32(original_rgb) * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def generate_explanation_figure(
    image_path: Union[str, Path],
    model: nn.Module,
    config: Dict[str, Any],
    ground_truth_label: Optional[int] = None,
    target_class: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    model_display_name: str = "Model"
) -> Dict[str, Any]:
    """
    Generates a 3-panel publication-grade explainability figure:
    1. Retinal Fundus Image (exact 256x256 geometry fed to network)
    2. Normalized Grad-CAM Heatmap (bicubic upsampled)
    3. Heatmap Overlay on the Retinal Image
    """
    image_path = Path(image_path)
    if device is None:
        device = get_device(config["hardware"].get("device", "auto"))

    task = config["task"]["type"]
    class_names = DR_CLASSES_5 if task == "five_class" else DR_CLASSES_BINARY

    # Target convolutional layer
    target_layer = model.get_cam_target_layer()

    # Preprocess fundus image with exact evaluation parameters
    img_size = config["preprocessing"]["image_size"]
    prep_method = config["preprocessing"].get("method", "clahe")
    pil_preprocessed = preprocess_fundus_image(
        str(image_path),
        image_size=img_size,
        method=prep_method,
        crop_border=True
    )
    rgb_preprocessed = np.array(pil_preprocessed)

    norm_mean = config["preprocessing"]["normalization"]["mean"]
    norm_std = config["preprocessing"]["normalization"]["std"]
    eval_transform = get_eval_transforms(mean=tuple(norm_mean), std=tuple(norm_std))
    tensor_input = eval_transform(pil_preprocessed).unsqueeze(0).to(device)

    with GradCAM(model, target_layer) as grad_cam:
        heatmap, pred_class, confidence, probs, diag = grad_cam.generate_heatmap(
            tensor_input, target_class=target_class
        )

    explained_class = target_class if target_class is not None else pred_class
    overlay = overlay_cam_on_image(rgb_preprocessed, heatmap, alpha=0.45)

    is_correct = (ground_truth_label == pred_class) if ground_truth_label is not None else None
    correctness_str = "Correct" if is_correct else ("Misclassified" if is_correct is False else "Unknown")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # 1. Retinal Fundus Image
    axes[0].imshow(rgb_preprocessed)
    gt_name = class_names[ground_truth_label] if ground_truth_label is not None else "Unknown"
    axes[0].set_title(
        f"Retinal Image: {image_path.stem}\nGround Truth: {gt_name}",
        fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im1 = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    explained_name = class_names[explained_class]
    axes[1].set_title(
        f"Grad-CAM Heatmap\nAttribution for: {explained_name}",
        fontsize=11, fontweight="bold"
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Heatmap Overlay
    axes[2].imshow(overlay)
    pred_name = class_names[pred_class]
    header_color = "#1b5e20" if is_correct else "#b71c1c"
    axes[2].set_title(
        f"{model_display_name} ({correctness_str})\nPred: {pred_name} (Conf: {confidence*100:.1f}%)",
        fontsize=11, fontweight="bold", color=header_color
    )
    axes[2].axis("off")

    # Scientific disclaimer footer
    disclaimer = (
        "RESEARCH DISCLAIMER: Grad-CAM visualizes model-focused convolutional feature attribution. "
        "It provides explanatory insight into network attention and does NOT constitute definitive clinical proof of lesions."
    )
    fig.text(0.5, 0.02, disclaimer, ha="center", fontsize=8.5, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "ground_truth": ground_truth_label,
        "ground_truth_name": gt_name,
        "predicted_class": pred_class,
        "predicted_class_name": pred_name,
        "is_correct": is_correct,
        "explained_target_class": explained_class,
        "explained_target_name": explained_name,
        "confidence": confidence,
        "target_layer": str(target_layer.__class__.__name__),
        "heatmap_shape": list(heatmap.shape),
        "diagnostics": diag,
        "output_path": str(save_path) if save_path else None
    }
