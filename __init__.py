"""
ComfyUI Utility Nodes - Modular, Offline, Python Functional Node Suite.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[ComfyUI-Utility-Nodes] Loaded {len(NODE_CLASS_MAPPINGS)} utility node(s):")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
