import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.background_remover.background_remover_node import BackgroundRemoverNode


def test_remove_background_returns_rgba_with_transparency():
    width, height = 32, 32
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    image[:, :, 0] = 200
    image[:, :, 2] = 200
    for y in range(8, 24):
        for x in range(8, 24):
            image[y, x] = [60, 180, 60]

    pil_image = Image.fromarray(image, mode='RGB')
    result = BackgroundRemoverNode().remove_background(pil_image, tolerance=25.0, smooth=0)

    assert result.mode == 'RGBA'
    alpha = np.array(result)[:, :, 3]
    assert alpha.min() < 255
    assert alpha.max() > 0
