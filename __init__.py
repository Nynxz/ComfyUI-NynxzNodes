from comfy_api.latest import ComfyExtension, io

from .nodes.StringTemplateParser import StringTemplateParser
from .nodes.compare_video import CompareVideoOptionsNode, CreateWipeCompareVideoNode, CreateFadeCompareVideoNode


class NynxzCustomNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CompareVideoOptionsNode,
            CreateFadeCompareVideoNode,
            CreateWipeCompareVideoNode,
            StringTemplateParser,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return NynxzCustomNodesExtension()

WEB_DIRECTORY = "./js"
__all__ = ["NynxzCustomNodesExtension", "comfy_entrypoint", "WEB_DIRECTORY"]
