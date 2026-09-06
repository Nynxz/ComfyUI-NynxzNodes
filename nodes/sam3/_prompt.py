"""The signed prompt: stock SAM3 `category:N` syntax, plus a leading `-` for subtraction.

    person:2, -head:2       two people, minus up to two heads
    person, -head           one person, minus every head found
    woman, man, -sunglasses two categories, both cleared of sunglasses

Core parses `category:N` in the **tokenizer**, not in the node — `_parse_prompts` in
`comfy/text_encoders/sam3_clip.py` — and packs the result into the conditioning as anonymous
`(embedding, max_detections)` pairs. By the time `SAM3 Detect` reads a prompt the term names are
already gone, so there is nowhere in the stock path for a sign to live. This is that same split,
mirrored piece for piece, with the sign read off first: a prompt containing no `-` parses here to
exactly what core would have parsed from it.

**One deliberate divergence — a bare negative means every match, not one.** Core defaults an
unqualified term to `max_detections=1`, which is right for a positive, because a positive is a
request: you are asking for the thing and one is the useful answer. A negative is a removal, and
removing one of the two heads in frame leaves the other sitting in the mask, which reads as a
broken node rather than as a default. So `-head` subtracts every match above threshold, and
`-head:2` caps it. Positives keep core's default of 1, trap and all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Core's own term pattern, `comfy/text_encoders/sam3_clip.py::_parse_prompts`.
_TERM = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")

#: The cap on a negative that carries no `:N`. Past the detector's query count this is "all".
UNCAPPED = 1_000_000

#: The one character this syntax adds to stock's.
SIGN = "-"


@dataclass(frozen=True)
class Term:
    """One comma-separated piece of a signed prompt."""

    text: str
    max_detections: int
    negative: bool

    def __str__(self) -> str:
        cap = "all" if self.max_detections >= UNCAPPED else self.max_detections
        return f"{SIGN if self.negative else ''}{self.text}:{cap}"


def parse(prompt: str) -> list[Term]:
    """A signed prompt as terms, in prompt order.

    The split mirrors core's: parentheses stripped, comma separated, blank pieces dropped. A `:N`
    that is not a number stays part of the term text, which is what core's regex does with
    anything it fails to match.
    """
    prompt = prompt.replace("(", "").replace(")", "")
    terms: list[Term] = []
    for piece in (p.strip() for p in prompt.split(",")):
        if not piece:
            continue
        negative = piece.startswith(SIGN)
        body = piece[len(SIGN) :].strip() if negative else piece
        if body:
            terms.append(_term(body, negative))
    return terms


def _term(body: str, negative: bool) -> Term:
    default = UNCAPPED if negative else 1
    match = _TERM.match(body)
    if match is None:
        return Term(body, default, negative)
    try:
        capped = max(1, round(float(match.group(2))))
    except ValueError:
        return Term(body, default, negative)  # "1.2.3" is a name, not a count
    return Term(match.group(1).strip(), capped, negative)
