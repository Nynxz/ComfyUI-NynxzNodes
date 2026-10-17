"""Find-and-replace from one line of rules: `green:red, orange:black`.

Rules are `find:replace` pairs separated by commas (or new lines); `\\,` and `\\:` put a literal
comma or colon in either side, and an empty replace (`orange:`) deletes. Every rule is applied in
one pass over the original text, so `red:blue, blue:red` swaps the two rather than turning both
blue, and the longest find wins where finds overlap. Pure — checked anywhere.
"""

from __future__ import annotations

import re

_SPLIT_RULES = re.compile(r"(?<!\\)[,\n]")
_SPLIT_PAIR = re.compile(r"(?<!\\):")


def _unescape(text: str) -> str:
    return text.replace("\\,", ",").replace("\\:", ":").strip()


def parse_rules(spec: str) -> list[tuple[str, str]]:
    """`"green:red, orange:black"` → `[("green", "red"), ("orange", "black")]`. A part with no
    colon, or nothing to find, is skipped."""
    rules = []
    for part in _SPLIT_RULES.split(spec or ""):
        pair = _SPLIT_PAIR.split(part, maxsplit=1)
        if len(pair) != 2:
            continue
        find, replace = _unescape(pair[0]), _unescape(pair[1])
        if find:
            rules.append((find, replace))
    return rules


def _cased_like(found: str, replacement: str) -> str:
    """`replacement` in the case `found` was written in: `Green` → `Red`, `GREEN` → `RED`."""
    if not replacement:
        return replacement
    if len(found) > 1 and found.isupper():
        return replacement.upper()
    if found[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _tidy(text: str) -> str:
    """The gaps a deleted word leaves: doubled spaces, and a space before punctuation."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)


def replace(text: str, rules: list[tuple[str, str]], whole_words=True, match_case=False) -> str:
    if not text or not rules:
        return text
    table: dict[str, str] = {}
    for find, to in rules:
        table.setdefault(find if match_case else find.lower(), to)
    finds = sorted(table, key=len, reverse=True)
    body = "|".join(re.escape(f) for f in finds)
    pattern = rf"(?<!\w)(?:{body})(?!\w)" if whole_words else body
    flags = 0 if match_case else re.IGNORECASE

    def swap(match: re.Match) -> str:
        found = match.group(0)
        to = table[found if match_case else found.lower()]
        return to if match_case else _cased_like(found, to)

    out = re.sub(pattern, swap, text, flags=flags)
    return _tidy(out) if any(not to for _, to in rules) else out
