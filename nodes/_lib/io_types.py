"""Custom IO types.

A widget type's `io_type` must equal the frontend widget key, which nodekit derives from the
component filename (`frontend/widgets/TestWidget.vue` -> `NYNXZ_TEST_WIDGET`).
"""

from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io


def widget_input(socketless: bool = True):
    """Build a WidgetInput subclass that renders as an on-node widget (no socket)."""

    class _WidgetInput(io.WidgetInput):
        def __init__(
            self,
            id: str,
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: Any = None,
        ):
            # Keyword args: comfy_api's positional signature has changed before.
            super().__init__(
                id,
                display_name=display_name,
                optional=optional,
                # The frontend shows an input's tooltip on hover anywhere over its widget; for a
                # socketless panel that is the whole node body. `tooltip` stays accepted as docs.
                tooltip=tooltip if not socketless else None,
                default=default,
                socketless=socketless,
            )

    return _WidgetInput


WIDGET_IO_PREFIX = "NYNXZ_"


def widget_io_type(component_name: str) -> str:
    """Mirror of nodekit's `typeId()`: "LoraStack" -> "NYNXZ_LORA_STACK"."""
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", component_name)
    snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", snake)
    return WIDGET_IO_PREFIX + snake.upper()


def widget_type(component_name: str, value_type: type, doc: str | None = None):
    """Declare a widget IO type bound to `frontend/widgets/<component_name>.vue`.

    Usage:
        LoraStackType = widget_type("LoraStack", list)
        ...
        inputs=[LoraStackType.Input("stack", default=[])]
    """
    io_type = widget_io_type(component_name)

    @io.comfytype(io_type=io_type)
    class _Type:
        Type = value_type
        Input = widget_input()

    _Type.__name__ = f"{component_name}Type"
    _Type.__qualname__ = _Type.__name__
    if doc:
        _Type.__doc__ = doc
    return _Type


def advanced(inp):
    """Mark an input advanced. Not a constructor kwarg — the concrete Input subclasses
    do not forward `extra_dict`."""
    inp.extra_dict = {**(getattr(inp, "extra_dict", None) or {}), "advanced": True}
    return inp


def inline_widget(inp, component_name: str):
    """Draw a standard input with ``frontend/widgets/<component_name>.vue``.

    The socket keeps the input's own type, so anything of that type can still be wired in; the
    frontend only swaps which widget draws the value. Not ``io.WidgetInput(widget_type=...)``,
    which would replace the socket type as well.
    """
    inp.extra_dict = {
        **(getattr(inp, "extra_dict", None) or {}),
        "widgetType": widget_io_type(component_name),
    }
    return inp


def color_input(id: str, default: str, tooltip: str | None = None, **kwargs):
    """A colour as a plain STRING: a picker on the node, any CSS colour string when wired."""
    return inline_widget(
        io.String.Input(id, default=default, tooltip=tooltip, **kwargs), "ColorField"
    )
