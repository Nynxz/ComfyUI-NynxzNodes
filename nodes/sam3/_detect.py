"""Detection and subtraction for a signed prompt. Everything model-facing lives here.

The loop is stock `SAM3 Detect`'s, with two changes: each term is encoded on its own so that its
sign survives as far as the detector, and the negatives are removed from the positives before the
masks leave the frame.

**Subtraction is union-then-remove.** Every negative found in a frame merges into one keep-out
mask, which is then taken off each positive — negatives are not matched back to the positive they
came from. For the ordinary case the two agree, because a head lies inside its own person and
nowhere else. They part company under occlusion, where person B's head overlaps person A's body
and the shared keep-out punches a hole in A. Assigning each negative to the positive that most
contains it is the fix, and is not done here.

Cost is the same shape as stock's: the vision backbone runs once per term per frame, because
`SAM3Detector.forward` re-encodes the image on every call. `forward_from_trunk` exists upstream and
takes a precomputed trunk, so the terms could share one encode; that is an optimisation this module
has not taken, in favour of calling the detector exactly the way the stock node calls it.

Masks are binary the whole way through — the detector's are thresholded, dilation is a max pool,
and removal is a multiply by `1 - keep_out`.
"""

from __future__ import annotations

import logging

import comfy.model_management
import comfy.utils
import torch
import torch.nn.functional as F

# Core's own helpers, imported rather than copied so that mask refinement cannot drift from the
# stock node's. They are private names: a rename upstream fails this module's import, loudly.
from comfy_extras.nodes_sam3 import _extract_text_prompts, _refine_mask

#: The square SAM3's image encoder wants. Stock resizes to it without preserving aspect.
INPUT_SIZE = 1008


def encode(clip, terms):
    """Each term's own conditioning, in prompt order.

    Encoded one term at a time rather than as a single `a:1, b:2` string. Core's multi-prompt path
    tokenizes each part separately anyway, so the embeddings come out identical, and this way the
    sign and the cap stay attached to their term instead of being recovered by position.
    """
    return [clip.encode_from_tokens_scheduled(clip.tokenize(term.text)) for term in terms]


def detect(
    model,
    clip,
    image,
    terms,
    threshold=0.5,
    refine_iterations=2,
    individual_masks=False,
    negative_grow=4,
    drop_empty=False,
):
    """Run every term over every frame. Returns `(masks, bboxes)` in stock's output shapes."""
    conditionings = encode(clip, terms)

    batch, height, width, _ = image.shape
    frames = comfy.utils.common_upscale(
        image[..., :3].movedim(-1, 1), INPUT_SIZE, INPUT_SIZE, "bilinear", crop="disabled"
    )

    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3 = model.model.diffusion_model

    prompts = []
    for conditioning, term in zip(conditionings, terms, strict=True):
        # The third field is core's own parse of a single-term string and is always 1; the term
        # carries the cap the user actually wrote, negatives included.
        embedding, attention, _ = _extract_text_prompts(conditioning, device, dtype)[0]
        prompts.append((embedding, attention, term))

    frame_masks = []
    frame_bboxes = []
    emptied = 0
    pbar = comfy.utils.ProgressBar(batch)

    for index in range(batch):
        frame = frames[index : index + 1].to(device=device, dtype=dtype)
        found = []  # [(mask, score)] — the positives
        keep_out = None  # running union of the negatives

        for embedding, attention, term in prompts:
            for mask, score in _run_term(
                sam3,
                frame,
                image[index],
                embedding,
                attention,
                term,
                threshold,
                refine_iterations,
                (height, width),
                device,
                dtype,
            ):
                if term.negative:
                    keep_out = mask if keep_out is None else torch.maximum(keep_out, mask)
                else:
                    found.append((mask, score))

        if keep_out is not None:
            keep = 1.0 - _dilate(keep_out, negative_grow)
            found = [(mask * keep, score) for mask, score in found]
            emptied += sum(1 for mask, _ in found if not bool(mask.any()))
        if drop_empty:
            found = [(mask, score) for mask, score in found if bool(mask.any())]

        frame_bboxes.append([_bbox(mask, score) for mask, score in found])
        frame_masks.append(_shape(found, individual_masks, height, width))
        pbar.update(1)

    if emptied:
        # Invisible from the graph otherwise: the node ran, masks came out, some are just blank.
        logging.warning(
            f"SAM3 Detect (Signed): the negatives consumed {emptied} positive detection(s) "
            f"entirely. Lower `negative_grow`, cap the negative with `:N`, or raise `threshold` "
            f"if a negative term is matching more than it should."
        )

    out_device = comfy.model_management.intermediate_device()
    frame_masks = [mask.to(out_device) for mask in frame_masks]
    masks = torch.cat(frame_masks, dim=0) if individual_masks else torch.stack(frame_masks)
    return masks, frame_bboxes


def _run_term(
    sam3, frame, source, embedding, attention, term, threshold, refine, size, device, dtype
):
    """One term's detections on one frame — best-scoring first, capped at the term's `:N`."""
    height, width = size
    results = sam3(
        frame,
        text_embeddings=embedding,
        text_mask=attention,
        threshold=threshold,
        orig_size=size,
    )
    scores = results["scores"][0].sigmoid()
    keep = scores > threshold
    boxes = results["boxes"][0][keep].cpu()
    scores = scores[keep].cpu()
    masks = results["masks"][0][keep]

    order = scores.argsort(descending=True)[: term.max_detections]
    return [
        (
            _refine_mask(sam3, source, masks[i], boxes[i], height, width, device, dtype, refine),
            float(scores[i]),
        )
        for i in order
    ]


def _dilate(mask, radius):
    """Grow a `[1, H, W]` mask by `radius` pixels.

    A negative traces its own silhouette, which rarely lands on the same pixels as the positive's
    interior. Removing it exactly leaves a ring of the negative's own edge behind — a collar of
    neck where a head was. A few pixels of growth is the difference between a clean hole and a
    halo, so this defaults on rather than to zero.
    """
    if radius <= 0:
        return mask
    return F.max_pool2d(mask.unsqueeze(0), 2 * radius + 1, stride=1, padding=radius)[0]


def _bbox(mask, score):
    """The tight box around a `[1, H, W]` mask, in the dict shape stock SAM3 emits.

    Recomputed from the residual rather than passed through from the detection, so that a box and
    its mask always describe the same pixels. An emptied positive gets a zero-size box, which keeps
    the two outputs index-aligned.
    """
    rows, cols = torch.nonzero(mask[0] > 0, as_tuple=True)
    if rows.numel() == 0:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0, "score": score}
    x0, x1 = int(cols.min()), int(cols.max())
    y0, y1 = int(rows.min()), int(rows.max())
    return {
        "x": float(x0),
        "y": float(y0),
        "width": float(x1 - x0 + 1),
        "height": float(y1 - y0 + 1),
        "score": score,
    }


def _shape(found, individual_masks, height, width):
    """One frame's residuals in the shape stock emits: `[objects, H, W]`, or one `[H, W]` union."""
    if not found:
        return torch.zeros(0, height, width) if individual_masks else torch.zeros(height, width)
    stacked = torch.cat([mask for mask, _ in found], dim=0)
    return stacked if individual_masks else (stacked > 0).any(dim=0).float()
