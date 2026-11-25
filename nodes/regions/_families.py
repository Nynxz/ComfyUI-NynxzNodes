"""The model families regions run on, one row each. Adding a family is a row here and a module
with its install, not another branch in the node."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import _krea, _moss, _zimage


@dataclass(frozen=True)
class Family:
    name: str
    #: Whether a ModelPatcher wraps this family's diffusion model.
    detect: Callable[[object], bool]
    #: Whether a ModelPatcher already carries this pack's region patches.
    already_regioned: Callable[[object], bool]
    #: `(patched clone, regions, {region: encoded prompt}, RegionSettings, sources) -> [log line]`.
    install: Callable[..., list[str]]
    #: Drop the tokens an encoded prompt's attention mask leaves out (see `_apply.encode`).
    trim_prompts: bool = False
    #: Settings this family doesn't do yet, by name: `track`, `layout_lock`.
    unsupported: tuple[str, ...] = ()


FAMILIES = (
    Family(
        "Z-Image",
        _zimage.is_zimage,
        _zimage.already_regioned,
        _zimage.install_regions,
        trim_prompts=True,
        unsupported=("track", "layout_lock"),
    ),
    Family(
        "MOSS-Image",
        _moss.is_moss,
        _moss.already_regioned,
        _moss.install_regions,
        trim_prompts=True,
        unsupported=("layout_lock",),
    ),
    Family("Krea 2", _krea.is_krea2, _krea.already_regioned, _krea.install_regions),
)


def family_of(model, node_name: str) -> Family:
    """The family `model` belongs to; raises naming them all when it is none of them.

    On a model none of them knows, no gate is ever built and every LoRA silently does nothing,
    which looks exactly like "the LoRA is weak" — so it fails loudly here instead.
    """
    for family in FAMILIES:
        if family.detect(model):
            return family
    names = ", ".join(family.name for family in FAMILIES)
    raise ValueError(
        f"{node_name} works on {names}, and the wired model is none of them. Load one of those "
        "with its own text encoder."
    )
