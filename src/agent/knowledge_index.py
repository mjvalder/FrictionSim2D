"""BM25 search over the installed knowledge corpus.

The corpus is ``passages.jsonl`` built by ``tools/agent_data/corpus.py``:
attributed passages from Wikipedia and openly licensed papers. It is not
shipped with the package (CC-BY-SA share-alike, size); it is installed to
`knowledge_dir` and the assistant works without it.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CORPUS_NAME = 'passages.jsonl'
#: BM25 parameters (the usual defaults).
K1, B = 1.5, 0.75
_TOKEN = re.compile(r'[a-z0-9]+')
_STOP = frozenset(
    'a an and are as at be by can do does for from has have how in is it its of on or '
    'that the their there these this to was what when which why with'.split())


def knowledge_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/knowledge`` (``~/.local/share/...`` by default)."""
    base = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(base) / 'FrictionSim2D' / 'knowledge'


def tokens(text: str) -> List[str]:
    """Lower-case words and numbers, stop words dropped ('MoS2' -> 'mos2')."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Passage:
    """One attributed passage."""

    id: str
    title: str
    section: str
    url: str
    licence: str
    text: str


class KnowledgeIndex:
    """Okapi BM25 over passages, title and section included in the indexed text."""

    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self._terms = [Counter(tokens(f'{p.title} {p.section} {p.text}')) for p in passages]
        self._lengths = [sum(t.values()) for t in self._terms]
        self._mean = sum(self._lengths) / len(passages) if passages else 0.0
        frequency: Counter = Counter()
        for terms in self._terms:
            frequency.update(terms.keys())
        total = len(passages)
        self._idf: Dict[str, float] = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5)) for term, n in frequency.items()}

    def __len__(self) -> int:
        return len(self.passages)

    @classmethod
    def from_file(cls, path: Path) -> 'KnowledgeIndex':
        """Read a ``passages.jsonl``; unknown keys are ignored."""
        passages = []
        with Path(path).open(encoding='utf-8') as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    passages.append(Passage(
                        id=str(row['id']), title=row.get('title', ''),
                        section=row.get('section', ''), url=row.get('url', ''),
                        licence=row.get('licence', ''), text=row['text']))
        return cls(passages)

    def search(self, query: str, k: int = 3) -> List[Passage]:
        """The ``k`` best passages for ``query``; none when no query term occurs."""
        wanted = [t for t in set(tokens(query)) if t in self._idf]
        scored: List[Tuple[float, int]] = []
        for i, terms in enumerate(self._terms):
            norm = K1 * (1 - B + B * self._lengths[i] / self._mean)
            score = sum(self._idf[t] * terms[t] * (K1 + 1) / (terms[t] + norm)
                        for t in wanted if terms[t])
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.passages[i] for _, i in scored[:k]]


_CACHE: Dict[Path, Tuple[float, KnowledgeIndex]] = {}


def load_knowledge(path: Optional[Path] = None) -> Optional[KnowledgeIndex]:
    """The index of ``path`` (default: the installed corpus), rebuilt only when the
    file changes; None when there is no corpus."""
    path = Path(path) if path is not None else knowledge_dir() / CORPUS_NAME
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _CACHE.get(path)
    if cached is None or cached[0] != mtime:
        cached = (mtime, KnowledgeIndex.from_file(path))
        _CACHE[path] = cached
    return cached[1]
