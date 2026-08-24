"""
ComfyUI Node: Background Remover (Offline / Pure Python)
Removes image background without any AI weights or network connections.
Outputs RGBA image (with transparent PNG alpha), alpha mask, and RGB image.
"""

from typing import Tuple
import torch
import numpy as np

from ...utils.image_ops import tensor_to_numpy, numpy_to_tensor
from .processor import remove_background


class BGRemoverNode:
    """
    Offline Background Remover Node for ComfyUI.
    Uses classical color difference, flood fill, edge feathering, and morphology.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (
                    [
                        "Auto Border Flood",
                        "Color Key / Chroma",
                        "Luminance (Bright BG)",
                        "Luminance (Dark BG)",
                        "GrabCut (Subject Extraction)",
                    ],
                    {"default": "Auto Border Flood"},
                ),
                "key_color": (
                    [
                        "Auto Detect",
                        "White",
                        "Black",
                        "Green Screen",
                        "Blue Screen",
                        "Magenta",
                        "Neutral Gray",
                        "Custom Hex",
                    ],
                    {"default": "Auto Detect"},
                ),
                "custom_hex": ("STRING", {"default": "#FFFFFF"}),
                "tolerance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_feather": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "shrink_expand": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "spill_suppress": ("BOOLEAN", {"default": True}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "background_fill": (
                    ["Transparent", "Black", "White", "Custom Hex"],
                    {"default": "Transparent"},
                ),
                "bg_custom_hex": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("rgba_image", "mask", "rgb_image")
    FUNCTION = "process"
    CATEGORY = "Utility/Image"

    def process(
        self,
        image: torch.Tensor,
        method: str,
        key_color: str,
        custom_hex: str,
        tolerance: float,
        edge_feather: int,
        shrink_expand: int,
        spill_suppress: bool,
        invert_mask: bool,
        background_fill: str,
        bg_custom_hex: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process incoming image tensor batch.
        """
        np_images = tensor_to_numpy(image)
        batch_size = np_images.shape[0]

        rgba_list = []
        mask_list = []
        rgb_list = []

        for i in range(batch_size):
            img_rgb = np_images[i][..., :3]  # Ensure 3-channel RGB
            
            rgba_out, mask_out, rgb_out = remove_background(
                image_rgb=img_rgb,
                method=method,
                key_color_preset=key_color,
                custom_hex=custom_hex,
                tolerance=tolerance,
                edge_feather=edge_feather,
                shrink_expand=shrink_expand,
                spill_suppress=spill_suppress,
                invert_mask=invert_mask,
                background_fill=background_fill,
                bg_custom_hex=bg_custom_hex,
            )

            rgba_list.append(rgba_out)
            mask_list.append(mask_out)
            rgb_list.append(rgb_out)

        out_rgba_np = np.stack(rgba_list, axis=0)
        out_mask_np = np.stack(mask_list, axis=0)
        out_rgb_np = np.stack(rgb_list, axis=0)

        out_rgba_tensor = numpy_to_tensor(out_rgba_np)
        out_mask_tensor = numpy_to_tensor(out_mask_np)
        out_rgb_tensor = numpy_to_tensor(out_rgb_np)

        return (out_rgba_tensor, out_mask_tensor, out_rgb_tensor)
