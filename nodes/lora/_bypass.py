"""A bypass LoRA that adds into the layer's output in place.

ComfyUI's bypass mode runs a LoRA as extra maths beside each layer instead of merging it into the
weights: `out = layer(x) + up(down(x)) * scale`. Written that way, three tensors the size of the
layer's output exist at once (the layer's, the LoRA's, and the LoRA's times `scale`). On a long
video sequence that output is gigabytes: MiniMax H3's MLP at ~126k tokens is 6.75 GiB, so the
stock bypass asks for ~20 GiB in that one layer and runs out of memory.

Here the scale is folded into the small `up` matrix and the product is added straight into the
layer's output with one `addmm_`, so only that output and a rank-wide intermediate exist. Same
arithmetic, one copy instead of three.

Only plain LoRA on a linear layer takes this path; anything else (conv layers, LoCon's mid
weights, DoRA, other adapter types) keeps ComfyUI's own bypass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def add_lora_(base: torch.Tensor, x: torch.Tensor, up, down, scale: float) -> torch.Tensor:
    """`base += scale * up(down(x))`, changing `base` itself, for a linear layer's `up`
    [out, rank] and `down` [rank, in]. Returns `base`."""
    hidden = F.linear(x, down.to(dtype=x.dtype))
    # Scaling `up` (out x rank) is megabytes; scaling the product would be another full output.
    scaled = (up.to(dtype=torch.float32) * float(scale)).to(dtype=base.dtype)
    if base.is_contiguous():
        base.view(-1, base.shape[-1]).addmm_(hidden.reshape(-1, hidden.shape[-1]), scaled.t())
    else:  # can't be viewed flat: one temporary, still short of the stock three
        base.add_(F.linear(hidden, scaled))
    return base


def make_lean(adapter) -> bool:
    """Switch a loaded LoRA adapter to the in-place bypass. False (and untouched) when it isn't a
    plain LoRA; whether its layer is linear is only known at run time, and checked there."""
    from comfy.weight_adapter.lora import LoRAAdapter

    if type(adapter) is not LoRAAdapter:
        return False
    weights = adapter.weights
    if weights[3] is not None or weights[4] is not None:  # LoCon mid, DoRA scale
        return False
    adapter.__class__ = _lean_class()
    return True


_LEAN = None


def _lean_class():
    global _LEAN
    if _LEAN is None:
        from comfy.weight_adapter.lora import LoRAAdapter

        class LeanLoRAAdapter(LoRAAdapter):
            """ComfyUI's bypass hook calls an adapter's own `bypass_forward` when it has one."""

            def bypass_forward(self, org_forward, x, *args, **kwargs):
                base = org_forward(x, *args, **kwargs)
                plain = (
                    not getattr(self, "is_conv", False)
                    and type(base) is torch.Tensor
                    and type(x) is torch.Tensor
                    and not base.requires_grad
                    and base is not x
                )
                if not plain:
                    return self.g(base + self.h(x, base))
                up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
                scale = (alpha / down.shape[0]) if alpha is not None else 1.0
                return add_lora_(base, x, up, down, scale * getattr(self, "multiplier", 1.0))

        _LEAN = LeanLoRAAdapter
    return _LEAN
