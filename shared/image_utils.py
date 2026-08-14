import numpy as np


def as_rgba_array(image):
    if hasattr(image, "mode"):
        return np.array(image.convert("RGBA"), dtype=np.uint8)

    if isinstance(image, list):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise TypeError("Unsupported image type")

    arr = image
    if arr.dtype != np.uint8:
        if arr.max() <= 1.1:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError("Only single images are supported in direct calls")

    if arr.ndim == 3 and arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.concatenate([
            np.repeat(arr, 3, axis=2),
            np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8),
        ], axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr.astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask

    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="edge")
    smoothed = np.zeros_like(mask, dtype=np.float32)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            window = padded[y:y + radius * 2 + 1, x:x + radius * 2 + 1]
            smoothed[y, x] = np.sum(window * kernel)

    return smoothed
