from comfy_api.latest import ComfyExtension, io
from .nodes.StringTemplateParser import StringTemplateParser
from .nodes.CreateFadeCompareVideo import CreateFadeCompareVideo
from .nodes.CreateWipeCompareVideo import CreateWipeCompareVideo


class NynxzCustomNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CreateFadeCompareVideo, CreateWipeCompareVideo, StringTemplateParser]


async def comfy_entrypoint() -> ComfyExtension:
    return NynxzCustomNodesExtension()
