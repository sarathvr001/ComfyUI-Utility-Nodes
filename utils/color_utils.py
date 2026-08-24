"""
Color utilities for ComfyUI Utility Nodes.
Pure Python and NumPy implementations for color conversions, distance calculations,
and background color estimation without external AI dependencies.
"""

from typing import Tuple, Optional
import numpy as np


PRESET_COLORS = {
    "White": (1.0, 1.0, 1.0),
    "Black": (0.0, 0.0, 0.0),
    "Green Screen": (0.0, 0.85, 0.0),
    "Blue Screen": (0.0, 0.2, 0.9),
    "Magenta": (1.0, 0.0, 1.0),
    "Neutral Gray": (0.5, 0.5, 0.5),
}


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Convert a hex color code (#RRGGBB or #RGB) to normalized float RGB tuple in [0.0, 1.0]."""
    if not hex_str:
        return (1.0, 1.0, 1.0)
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join([c * 2 for c in cleaned])
    if len(cleaned) != 6:
        return (1.0, 1.0, 1.0)
    try:
        r = int(cleaned[0:2], 16) / 255.0
        g = int(cleaned[2:4], 16) / 255.0
        b = int(cleaned[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return (1.0, 1.0, 1.0)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image [H, W, 3] in [0, 1] to HSV [H, W, 3] in [0, 1].
    Pure NumPy vectorised implementation.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Value
    v = max_c

    # Saturation
    s = np.zeros_like(max_c)
    nonzero_max = max_c > 1e-6
    s[nonzero_max] = delta[nonzero_max] / max_c[nonzero_max]

    # Hue
    h = np.zeros_like(max_c)
    nonzero_delta = delta > 1e-6

    idx_r = nonzero_delta & (max_c == r)
    idx_g = nonzero_delta & (max_c == g)
    idx_b = nonzero_delta & (max_c == b)

    h[idx_r] = ((g[idx_r] - b[idx_r]) / delta[idx_r]) % 6.0
    h[idx_g] = ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 2.0
    h[idx_b] = ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 4.0
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def color_distance_perceptual(img_rgb: np.ndarray, target_rgb: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate weighted perceptual Euclidean distance between an image and a target RGB color.
    Weights standard human eye sensitivity: R: 0.299, G: 0.587, B: 0.114.
    Returns array [H, W] normalized in [0, 1].
    """
    target = np.array(target_rgb, dtype=np.float32).reshape((1, 1, 3))
    diff = img_rgb.astype(np.float32) - target
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape((1, 1, 3))
    weighted_diff_sq = (diff ** 2) * weights
    dist = np.sqrt(np.sum(weighted_diff_sq, axis=-1))
    return dist


def sample_border_colors(img_rgb: np.ndarray, border_width: int = 4) -> Tuple[float, float, float]:
    """
    Sample border pixels (top, bottom, left, right edges) of an image to estimate
    the dominant background color.
    """
    h, w = img_rgb.shape[:2]
    bw = min(border_width, h // 4, w // 4, 10)
    if bw < 1:
        bw = 1

    top = img_rgb[:bw, :, :].reshape(-1, 3)
    bottom = img_rgb[-bw:, :, :].reshape(-1, 3)
    left = img_rgb[:, :bw, :].reshape(-1, 3)
    right = img_rgb[:, -bw:, :].reshape(-1, 3)

    borders = np.concatenate([top, bottom, left, right], axis=0)
    # Median is robust against foreground touching border slightly
    median_color = np.median(borders, axis=0)
    return tuple(float(c) for c in median_color)
