import numpy as np
from PIL import Image

from shared.image_utils import as_rgba_array, smooth_mask

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class BackgroundRemoverNode:
    """A utility node that removes a solid-color background without using an AI model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "smoothing": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            }
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)

    @classmethod
    def RETURN_NAMES(cls):
        return ("image",)

    @classmethod
    def FUNCTION(cls):
        return "remove_background"

    @classmethod
    def CATEGORY(cls):
        return "Utility"

    def remove_background(self, image, tolerance: float = 18.0, smoothing: int = 0, smooth=None):
        if image is None:
            raise ValueError("image is required")

        if smooth is not None:
            smoothing = smooth

        if torch is not None and isinstance(image, torch.Tensor):
            tensor = image.detach().cpu().numpy()
            result = self._process_batch(tensor, tolerance=tolerance, smoothing=smoothing)
            return (torch.from_numpy(result.astype(np.float32) / 255.0),)

        pixels = as_rgba_array(image)
        processed = self._compute_background_mask(pixels, tolerance=tolerance, smoothing=smoothing)
        return Image.fromarray(processed, mode="RGBA")

    def _process_batch(self, batch, tolerance: float, smoothing: int):
        if batch.ndim == 3:
            batch = batch[None, ...]

        if batch.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported image shape: {batch.shape}")

        processed = []
        for sample in batch:
            sample_pixels = as_rgba_array(sample)
            processed.append(self._compute_background_mask(sample_pixels, tolerance=tolerance, smoothing=smoothing))

        combined = np.stack(processed)
        return combined.astype(np.uint8)

    def _compute_background_mask(self, pixels: np.ndarray, tolerance: float = 18.0, smoothing: int = 0):
        rgba = pixels.astype(np.float32)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        sample_size = max(2, min(10, min(rgb.shape[0], rgb.shape[1]) // 10))
        border_y = np.concatenate([
            np.arange(0, rgb.shape[0], sample_size),
            np.arange(rgb.shape[0] - 1, -1, -sample_size),
        ])
        border_x = np.concatenate([
            np.arange(0, rgb.shape[1], sample_size),
            np.arange(rgb.shape[1] - 1, -1, -sample_size),
        ])

        border_pixels = []
        for y in border_y:
            for x in border_x:
                border_pixels.append(rgb[y, x])
        border_pixels = np.asarray(border_pixels, dtype=np.float32)
        background_color = (
            rgb.mean(axis=(0, 1)) if border_pixels.size == 0 else np.median(border_pixels, axis=0)
        )

        distance_to_background = np.linalg.norm(rgb - background_color, axis=2)
        background_mask = (distance_to_background <= max(8.0, tolerance * 1.3)).astype(np.float32)

        if smoothing > 0:
            background_mask = smooth_mask(background_mask, smoothing)

        background_mask = (background_mask > 0.5).astype(np.float32)
        new_alpha = np.clip(alpha * (1.0 - background_mask), 0, 255)
        return np.dstack([rgb.astype(np.uint8), new_alpha.astype(np.uint8)]).astype(np.uint8)


NODE_CLASS_MAPPINGS = {"BackgroundRemoverNode": BackgroundRemoverNode}
NODE_DISPLAY_NAME_MAPPINGS = {"BackgroundRemoverNode": "Background Remover"}
