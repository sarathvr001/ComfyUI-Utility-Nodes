"""
Background Remover Processing Core for ComfyUI Utility Nodes.
100% offline, pure Python/NumPy/SciPy algorithms without any AI models.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image

from ...utils.color_utils import (
    hex_to_rgb,
    rgb_to_hsv,
    color_distance_perceptual,
    sample_border_colors,
    PRESET_COLORS,
)
from ...utils.image_ops import (
    apply_mask_morphology,
    apply_feather,
    suppress_color_spill,
    composite_rgba,
)

try:
    from scipy.ndimage import label, binary_fill_holes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep interpolation."""
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _flood_fill_connected_border(img_rgb: np.ndarray, target_color: Tuple[float, float, float], tolerance: float) -> np.ndarray:
    """
    Find background regions connected to the image boundary within the given color tolerance.
    Returns a float mask in [0, 1] where 1 is foreground, 0 is background.
    """
    h, w = img_rgb.shape[:2]
    dist = color_distance_perceptual(img_rgb, target_color)
    is_bg_candidate = dist <= tolerance

    if HAS_SCIPY:
        # Connected component analysis: only keep background regions touching the borders
        labeled, num_features = label(is_bg_candidate)
        if num_features == 0:
            return np.ones((h, w), dtype=np.float32)

        # Border labels
        border_labels = set()
        border_labels.update(np.unique(labeled[0, :]))
        border_labels.update(np.unique(labeled[-1, :]))
        border_labels.update(np.unique(labeled[:, 0]))
        border_labels.update(np.unique(labeled[:, -1]))
        border_labels.discard(0)  # 0 is non-candidate

        # Create background mask from connected border candidates
        bg_mask = np.isin(labeled, list(border_labels))
        fg_mask = (~bg_mask).astype(np.float32)
        return fg_mask
    else:
        # Fallback to direct distance mask
        return np.clip(dist / max(tolerance, 1e-5), 0.0, 1.0)


def _process_color_key(
    img_rgb: np.ndarray,
    target_color: Tuple[float, float, float],
    tolerance: float,
    softness: float = 0.1
) -> np.ndarray:
    """
    Compute foreground mask using color distance and smooth transition band.
    """
    dist = color_distance_perceptual(img_rgb, target_color)
    low_thresh = max(0.0, tolerance - softness / 2.0)
    high_thresh = min(1.0, tolerance + softness / 2.0)
    fg_mask = _smoothstep(low_thresh, high_thresh, dist)
    return fg_mask


def _process_luminance_key(img_rgb: np.ndarray, target_type: str, tolerance: float, softness: float = 0.1) -> np.ndarray:
    """
    Compute foreground mask by keying high or low luminance.
    """
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    
    if "Bright" in target_type or "White" in target_type:
        # High brightness is background
        threshold = 1.0 - tolerance
        low_thresh = max(0.0, threshold - softness / 2.0)
        high_thresh = min(1.0, threshold + softness / 2.0)
        # Foreground is darker than threshold
        fg_mask = 1.0 - _smoothstep(low_thresh, high_thresh, lum)
    else:
        # Low brightness (Black) is background
        low_thresh = max(0.0, tolerance - softness / 2.0)
        high_thresh = min(1.0, tolerance + softness / 2.0)
        # Foreground is brighter than tolerance
        fg_mask = _smoothstep(low_thresh, high_thresh, lum)

    return fg_mask


def _process_grabcut(img_rgb: np.ndarray, border_margin: int = 2) -> np.ndarray:
    """
    Run OpenCV GrabCut foreground extraction if available.
    """
    if not HAS_CV2:
        # Fallback to auto border flood
        border_color = sample_border_colors(img_rgb)
        return _flood_fill_connected_border(img_rgb, border_color, tolerance=0.18)

    h, w = img_rgb.shape[:2]
    img_bgr = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin = max(1, min(border_margin, h // 10, w // 10))
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        return fg_mask
    except Exception:
        border_color = sample_border_colors(img_rgb)
        return _flood_fill_connected_border(img_rgb, border_color, tolerance=0.18)


def remove_background(
    image_rgb: np.ndarray,
    method: str = "Auto Border Flood",
    key_color_preset: str = "Auto Detect",
    custom_hex: str = "#00FF00",
    tolerance: float = 0.15,
    edge_feather: int = 2,
    shrink_expand: int = 0,
    spill_suppress: bool = True,
    invert_mask: bool = False,
    background_fill: str = "Transparent",
    bg_custom_hex: str = "#000000"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute background removal on a single [H, W, 3] RGB float32 image.
    
    Returns:
        rgba_image: [H, W, 4] float32 image with alpha channel.
        mask: [H, W] float32 mask (1 = foreground, 0 = background).
        rgb_image: [H, W, 3] float32 composited or masked RGB image.
    """
    # 1. Determine key color
    if key_color_preset == "Auto Detect":
        key_color = sample_border_colors(image_rgb)
    elif key_color_preset == "Custom Hex":
        key_color = hex_to_rgb(custom_hex)
    elif key_color_preset in PRESET_COLORS:
        key_color = PRESET_COLORS[key_color_preset]
    else:
        key_color = (1.0, 1.0, 1.0)

    # 2. Generate Initial Foreground Mask
    if method == "Auto Border Flood":
        raw_mask = _flood_fill_connected_border(image_rgb, key_color, tolerance)
    elif method == "Color Key / Chroma":
        raw_mask = _process_color_key(image_rgb, key_color, tolerance, softness=max(0.05, tolerance * 0.5))
    elif method == "Luminance (Bright BG)":
        raw_mask = _process_luminance_key(image_rgb, "Bright", tolerance)
    elif method == "Luminance (Dark BG)":
        raw_mask = _process_luminance_key(image_rgb, "Dark", tolerance)
    elif method == "GrabCut (Subject Extraction)":
        raw_mask = _process_grabcut(image_rgb)
    else:
        raw_mask = _flood_fill_connected_border(image_rgb, key_color, tolerance)

    # 3. Invert mask if requested
    if invert_mask:
        raw_mask = 1.0 - raw_mask

    # 4. Refine mask edges
    processed_mask = raw_mask
    if shrink_expand != 0:
        processed_mask = apply_mask_morphology(processed_mask, shrink_expand)
    
    if edge_feather > 0:
        processed_mask = apply_feather(processed_mask, float(edge_feather))

    processed_mask = np.clip(processed_mask, 0.0, 1.0)

    # 5. Clean color spill from foreground edges
    cleaned_rgb = image_rgb.copy()
    if spill_suppress:
        cleaned_rgb = suppress_color_spill(cleaned_rgb, processed_mask, key_color, strength=0.75)

    # 6. Create RGBA output
    rgba_output = composite_rgba(cleaned_rgb, processed_mask)

    # 7. Create RGB output with specified background fill
    if background_fill == "White":
        bg_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif background_fill == "Black":
        bg_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elif background_fill == "Custom Hex":
        bg_rgb = np.array(hex_to_rgb(bg_custom_hex), dtype=np.float32)
    else:  # "Transparent" or direct masked
        bg_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    m = processed_mask[..., np.newaxis]
    rgb_output = cleaned_rgb * m + bg_rgb * (1.0 - m)
    rgb_output = np.clip(rgb_output, 0.0, 1.0)

    return rgba_output, processed_mask, rgb_output
