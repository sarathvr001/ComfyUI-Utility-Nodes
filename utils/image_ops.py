"""
Image and mask processing operations for ComfyUI Utility Nodes.
Provides fast tensor conversions, Gaussian edge softening, morphological filtering,
and defringing using NumPy, SciPy, and Pillow.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, grey_erosion, grey_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI image tensor [B, H, W, C] or [H, W, C] in range [0, 1] to NumPy float32 array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    return arr


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy array [B, H, W, C] or [H, W, C] or [H, W] to ComfyUI torch.Tensor float32.
    """
    if isinstance(arr, torch.Tensor):
        return arr.float()
    
    if arr.ndim == 2:  # [H, W] -> [1, H, W] for mask
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):  # [H, W, C] -> [1, H, W, C]
        arr = np.expand_dims(arr, axis=0)

    tensor = torch.from_numpy(arr.astype(np.float32))
    return tensor


def apply_mask_morphology(mask: np.ndarray, size: int) -> np.ndarray:
    """
    Erode (size < 0) or Dilate (size > 0) a 2D float mask in [0, 1].
    Positive size expands the mask (more foreground), negative shrinks it.
    """
    if size == 0:
        return mask

    abs_size = abs(size)
    radius = max(1, abs_size)

    if HAS_SCIPY:
        # Structure element for circular/smooth morphology
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        struct = (x*x + y*y) <= radius*radius

        if size > 0:
            return grey_dilation(mask, footprint=struct)
        else:
            return grey_erosion(mask, footprint=struct)
    else:
        # Pillow fallback
        pil_mask = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        filter_size = radius * 2 + 1
        if size > 0:
            processed = pil_mask.filter(ImageFilter.MaxFilter(filter_size))
        else:
            processed = pil_mask.filter(ImageFilter.MinFilter(filter_size))
        return np.array(processed, dtype=np.float32) / 255.0


def apply_feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply Gaussian feathering / blur to mask edges.
    """
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0)

    if HAS_SCIPY:
        sigma = max(0.1, float(radius) / 2.0)
        blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        return np.clip(blurred, 0.0, 1.0)
    else:
        pil_mask = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred, dtype=np.float32) / 255.0


def suppress_color_spill(
    rgb: np.ndarray,
    mask: np.ndarray,
    key_color: Tuple[float, float, float],
    strength: float = 0.8
) -> np.ndarray:
    """
    Suppress color bleeding / halos on foreground edges matching the removed background key color.
    """
    if strength <= 0.0:
        return rgb

    result = rgb.copy()
    kr, kg, kb = key_color

    # Determine dominant channel of key color
    if kg > kr and kg > kb:
        # Green screen spill suppression: clamp Green to max(Red, Blue)
        max_rb = np.maximum(result[..., 0], result[..., 2])
        excess_g = np.maximum(0.0, result[..., 1] - max_rb)
        # Apply more on semi-transparent edge regions
        edge_weight = (1.0 - mask) * strength
        result[..., 1] -= excess_g * edge_weight
    elif kb > kr and kb > kg:
        # Blue screen spill suppression: clamp Blue to max(Red, Green)
        max_rg = np.maximum(result[..., 0], result[..., 1])
        excess_b = np.maximum(0.0, result[..., 2] - max_rg)
        edge_weight = (1.0 - mask) * strength
        result[..., 2] -= excess_b * edge_weight
    elif kr > kg and kr > kb:
        # Red spill suppression
        max_gb = np.maximum(result[..., 1], result[..., 2])
        excess_r = np.maximum(0.0, result[..., 0] - max_gb)
        edge_weight = (1.0 - mask) * strength
        result[..., 0] -= excess_r * edge_weight

    return np.clip(result, 0.0, 1.0)


def composite_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Combine [H, W, 3] RGB and [H, W] Alpha into [H, W, 4] RGBA array.
    """
    h, w = rgb.shape[:2]
    alpha_expanded = alpha.reshape((h, w, 1))
    return np.concatenate([rgb, alpha_expanded], axis=-1)
