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
                tooltip=tooltip,
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
