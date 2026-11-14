"""Writing-style check used for generated text (notebook markdown and teaching notes).

Flags em dashes, semicolons, curly quotes and a list of overused words. Code spans are ignored.
"""
from __future__ import annotations

import re

# Words the project's writing-style rules ban (checked as whole words, case-insensitive).
BANNED = [
    "delve", "tapestry", "seamless", "groundbreaking", "pivotal", "underscore",
    "foster", "landscape", "testament", "crucial", "comprehensive", "robust",
    "leverage", "utilize", "moreover", "furthermore", "consequently",
    "overarching", "paradigm", "realm", "serves as", "stands as",
]


def style_check(name: str, text: str) -> list[str]:
    """Warn about the writing-style rules (code spans are ignored)."""
    plain = re.sub(r"`[^`]*`", "", text)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = re.sub(r"&[a-zA-Z#0-9]+;", "", plain)
    warnings = []
    for i, line in enumerate(plain.splitlines(), 1):
        if "\u2014" in line:
            warnings.append(f"{name}:{i}: em dash")
        if ";" in line:
            warnings.append(f"{name}:{i}: semicolon")
        if re.search("[\u201c\u201d\u2018\u2019]", line):
            warnings.append(f"{name}:{i}: curly quote")
        low = line.lower()
        for word in BANNED:
            if re.search(r"\b" + re.escape(word) + r"\w*\b", low):
                warnings.append(f"{name}:{i}: banned word '{word}'")
    return warnings
