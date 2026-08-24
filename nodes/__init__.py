"""
Dynamic registry for all utility sub-nodes.
Scans each subfolder under 'nodes' and aggregates NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.
"""

import os
import importlib
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = Path(__file__).parent

for item in current_dir.iterdir():
    if item.is_dir() and not item.name.startswith((".", "_")):
        init_file = item / "__init__.py"
        if init_file.exists():
            module_name = f".{item.name}"
            try:
                mod = importlib.import_module(module_name, package=__name__)
                if hasattr(mod, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
                if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"[ComfyUI-Utility-Nodes] Warning: Failed to load node module '{item.name}': {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
