# ComfyUI-Utility-Nodes

[![GitHub stars](https://img.shields.io/github/stars/sarathvr001/ComfyUI-Utility-Nodes?style=social)](https://github.com/sarathvr001/ComfyUI-Utility-Nodes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, extensible collection of **offline, pure Python utility nodes** for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

No heavy AI model checkpoints, no external network calls, and zero model downloading required. Everything runs locally, instantly, and efficiently.

---

## 📂 Architecture

`ComfyUI-Utility-Nodes` is built with a clean, modular folder architecture where each node has its own directory and decoupled logic:

```
ComfyUI-Utility-Nodes/
├── __init__.py                # Auto-discovers and registers all nodes dynamically
├── requirements.txt           # Minimal core dependencies (numpy, pillow, scipy)
├── README.md
├── utils/                     # Shared utilities
│   ├── __init__.py
│   ├── color_utils.py         # Color space, hex parser, perceptual delta, auto-border sampling
│   └── image_ops.py           # Tensor conversions, Gaussian feathering, morphology, defringing
└── nodes/                     # Dedicated folders for each node
    ├── __init__.py            # Node discovery aggregator
    └── bg_remover/            # Background Remover Node
        ├── __init__.py        # Node export mappings
        ├── node.py            # ComfyUI Node definition (inputs, outputs, execution)
        └── processor.py       # Offline image processing algorithms
```

### Adding a New Node
1. Create a folder under `nodes/<your_node_name>/`.
2. Implement your logic in `node.py` and `processor.py`.
3. Export `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `nodes/<your_node_name>/__init__.py`.
4. The root loader automatically discovers and registers your node!

---

## 🛠 Available Nodes

| Node Name | Display Name | Category | Description |
| :--- | :--- | :--- | :--- |
| `BGRemoverUtilityNode` | **Background Remover (Offline)** | `Utility/Image` | Fast, 100% offline background remover with flood fill, chroma key, luminance thresholding, defringing, and edge feathering. |

---

## 🌟 Node Guide: Background Remover (Offline)

### Features
- **Auto Border Flood Fill**: Automatically detects background color from the 4 image borders/corners and flood-fills connected background regions without erasing matching colors inside the subject.
- **Color Key / Chroma Key**: Removes background by target color (Green Screen, Blue Screen, White, Black, Magenta, or Custom Hex).
- **Luminance Keying**: Threshold-based separation for bright/white or dark/black studio backgrounds.
- **GrabCut Segmentation**: Iterative foreground extraction.
- **Mask Refinement**:
  - `edge_feather`: Soft Gaussian blending along edges.
  - `shrink_expand`: Morphological erosion/dilation to trim or expand borders.
  - `spill_suppress`: Cleans background color fringe (e.g. green cast) from foreground edges.
  - `invert_mask`: Easily invert foreground and background.
  - `background_fill`: Choose between Transparent (RGBA PNG), Black, White, or Custom Hex background fill.

### Inputs
- `image`: Image tensor (`IMAGE`).
- `method`: Extraction algorithm (`Auto Border Flood`, `Color Key / Chroma`, `Luminance (Bright BG)`, `Luminance (Dark BG)`, `GrabCut (Subject Extraction)`).
- `key_color`: Color preset or `Auto Detect` / `Custom Hex`.
- `custom_hex`: Custom hex color code (e.g. `#00FF00` or `#FFFFFF`).
- `tolerance`: Color sensitivity threshold (`0.0` to `1.0`).
- `edge_feather`: Edge softening radius.
- `shrink_expand`: Mask contraction/expansion (`-50` to `50`).
- `spill_suppress`: Remove color cast from edge pixels.
- `invert_mask`: Flip mask output.
- `background_fill`: Output background replacement (`Transparent`, `Black`, `White`, `Custom Hex`).

### Outputs
- `rgba_image`: 4-channel RGBA image tensor with transparent alpha channel (ready for PNG export).
- `mask`: 1-channel alpha mask tensor (`MASK`).
- `rgb_image`: 3-channel RGB image tensor composited with chosen background fill.

---

## 🚀 Installation

### Method 1: ComfyUI Manager
Search for `ComfyUI-Utility-Nodes` and click **Install**.

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sarathvr001/ComfyUI-Utility-Nodes.git
```

Dependencies (standard with ComfyUI):
```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## 📜 License

This project is licensed under the **MIT License**.
