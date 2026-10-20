"""The image an editor node's run sends back to its editor as the trace image: saved to temp, with
the node it is for and its size, under `<key>_source`, `<key>_node` and `<key>_size` in the run's UI
output (read by frontend/lib/sentBack.ts)."""

from __future__ import annotations

from comfy_api.latest import io, ui


def sent_back(key: str, node: str, image, cls, name: str) -> dict:
    """The UI keys that hand `image`'s first frame back to editor node `node`; saved as `name`."""
    return {
        f"{key}_node": [node],
        f"{key}_source": ui.ImageSaveHelper.save_images(
            image[:1], name, io.FolderType.temp, cls, compress_level=1
        ),
        f"{key}_size": [int(image.shape[2]), int(image.shape[1])],
    }
