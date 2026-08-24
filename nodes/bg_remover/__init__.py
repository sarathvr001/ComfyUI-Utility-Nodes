from .node import BGRemoverNode

NODE_CLASS_MAPPINGS = {
    "BGRemoverUtilityNode": BGRemoverNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BGRemoverUtilityNode": "Background Remover (Offline)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "BGRemoverNode"]
