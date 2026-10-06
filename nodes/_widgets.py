"""Pack-specific widget inputs, built on the vendored io_types."""

from __future__ import annotations

from comfy_api.latest import io

from ._zenkit.io_types import inline_widget


def color_input(id: str, default: str, tooltip: str | None = None, **kwargs):
    """A colour as a plain STRING: a picker on the node, any CSS colour string when wired."""
    return inline_widget(
        io.String.Input(id, default=default, tooltip=tooltip, **kwargs), "ColorField"
    )
