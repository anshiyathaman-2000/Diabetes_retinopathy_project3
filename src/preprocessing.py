"""
Retinal Fundus Image Preprocessing and Augmentation Pipeline.
Supports:
1. Standard RGB resizing.
2. Circular fundus masking / black border auto-cropping.
3. LAB-space CLAHE (Contrast Limited Adaptive Histogram Equalization)
   preserving retinal chromatic fidelity while enhancing microvascular lesions.
4. Clinically safe data augmentations for training.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

def auto_crop_fundus(image_np: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """
    Crops away non-informative black borders around retinal fundus image.
    Uses luminance thresholding to locate the active retinal field of view.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    if not np.any(mask):
        return image_np

    # Find bounding box of fundus region
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Ensure valid bounding box
    if y_max > y_min and x_max > x_min:
        return image_np[y_min:y_max+1, x_min:x_max+1]
    return image_np

def apply_clahe_lab(
    image_np: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE on the Luminance (L) channel in LAB color space.
    This enhances contrast of microaneurysms, hemorrhages, and exudates
    without causing unnatural chromatic shifts in the A/B channels.
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def apply_standard_clahe_rgb(
    image_np: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies standard CLAHE independently to each channel in RGB color space.
    This represents common baseline CLAHE usage without perceptual color-space separation.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r, g, b = cv2.split(image_np)
    r_enh = clahe.apply(r)
    g_enh = clahe.apply(g)
    b_enh = clahe.apply(b)
    return cv2.merge([r_enh, g_enh, b_enh])

def preprocess_fundus_image(
    image_input: Union[str, Image.Image, np.ndarray],
    image_size: int = 256,
    method: str = "clahe",
    crop_border: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Image.Image:
    """
    Executes fundus-specific preprocessing:
    1. Read and convert to RGB numpy array.
    2. Auto-crop non-informative black margins if requested.
    3. Apply contrast enhancement (LAB CLAHE, Standard RGB CLAHE, or None/Original).
    4. Resize to target dimension using bilinear interpolation.
    5. Return PIL Image ready for torchvision transforms.
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input if image_input.shape[2] == 3 else cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Step 1: Crop black border
    if crop_border:
        img_rgb = auto_crop_fundus(img_rgb)

    # Step 2: Contrast enhancement
    if method in ["clahe", "clahe_lab"]:
        img_rgb = apply_clahe_lab(img_rgb, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method in ["standard_clahe", "clahe_rgb"]:
        img_rgb = apply_standard_clahe_rgb(img_rgb, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method in ["original", "none"]:
        pass  # Keep original raw color distributions
    else:
        raise ValueError(f"Unknown preprocessing method '{method}'. Choose 'clahe', 'standard_clahe', or 'original'.")

    # Step 3: Resize with anti-aliasing
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return pil_img

def get_train_transforms(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_degrees: float = 15.0,
    brightness_jitter: float = 0.1,
    contrast_jitter: float = 0.1
) -> transforms.Compose:
    """
    Builds training augmentation pipeline.
    Applies only clinically safe transformations:
    - Random horizontal and vertical flips (preserves retinal structure symmetry).
    - Mild rotation (angles up to 15 degrees).
    - Mild brightness and contrast adjustment (simulates varying clinic camera exposures).
    Does NOT use shearing, drastic color hue shifts, or aggressive erasing that could mimic lesions.
    """
    transform_list = [
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=vertical_flip_prob),
        transforms.RandomRotation(degrees=rotation_degrees, fill=0),
        transforms.ColorJitter(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=0.05,
            hue=0.0
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    return transforms.Compose(transform_list)

def get_eval_transforms(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Deterministic evaluation transformation (Validation and Testing).
    Strictly converts PIL image to Tensor and normalizes.
    No stochastic augmentations applied.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
