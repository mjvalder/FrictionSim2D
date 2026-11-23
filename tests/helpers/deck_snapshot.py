"""Deck normalisation: strip run-specific noise so decks compare and hash stably."""
from __future__ import annotations

import re
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])

_SEED_PATTERNS = [
    re.compile(r"(?m)^(\s*velocity\s+\S+\s+create\s+\S+\s+)\d+"),
    re.compile(r"(?m)^(\s*fix\b.*\blangevin\s+\S+\s+\S+\s+\S+\s+)\d+"),
    re.compile(r"(?mi)^(\s*variable\s+\w*seed\w*\s+\w+\s+)\d+"),
]
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
                   r"|\b[0-9a-f]{32}\b")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_GIT = re.compile(r"\b[0-9a-f]{40}\b")
_TAIL = r"(?:/[^\s\"'),:;]*)?"


def _abs_paths(text: str) -> str:
    text = text.replace(_REPO, "<ABS>")
    for prefix in (r"/(?:var/)?tmp", r"/var/folders", r"/home/[^/\s]+"):
        text = re.sub(prefix + _TAIL, "<ABS>", text)
    return re.sub(r"<ABS>" + _TAIL, "<ABS>", text)


def normalise_deck(text: str) -> str:
    """Replace seeds, uuids, timestamps, commit hashes and absolute paths."""
    for pat in _SEED_PATTERNS:
        text = pat.sub(r"\g<1><SEED>", text)
    text = _TIMESTAMP.sub("<TIMESTAMP>", text)
    text = _UUID.sub("<UUID>", text)
    text = _GIT.sub("<GITHASH>", text)
    return _abs_paths(text)
