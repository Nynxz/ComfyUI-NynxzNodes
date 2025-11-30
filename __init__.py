from comfy_api.latest import ComfyExtension, io

from .nodes.StringTemplateParser import StringTemplateParser
from .nodes.preview_maker import GalleryLoaderNode, ImagesPreviewMakerNode, PreviewMakerThemePickerNode, VideoPreviewMakerNode
from .nodes.compare_video import CompareVideoOptionsNode, CreateWipeCompareVideoNode, CreateFadeCompareVideoNode


class NynxzCustomNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CompareVideoOptionsNode,
            CreateFadeCompareVideoNode,
            CreateWipeCompareVideoNode,
            StringTemplateParser,
            VideoPreviewMakerNode,
            GalleryLoaderNode,
            ImagesPreviewMakerNode,
            PreviewMakerThemePickerNode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return NynxzCustomNodesExtension()
