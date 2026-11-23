"""Search the local knowledge corpus: attributed passages, never the network."""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import compact

HITS = 3
#: Characters of each passage shown, so three fit the result budget with their sources.
EXCERPT = 380
NO_CORPUS = ('No knowledge corpus is installed; answer from the other tools or say '
             'you do not know.')


def excerpt(text: str) -> str:
    """The start of ``text``, cut at a sentence end where one is near."""
    if len(text) <= EXCERPT:
        return text
    cut = text[:EXCERPT]
    end = cut.rfind('. ')
    return cut[:end + 1] if end > EXCERPT // 2 else cut[:cut.rfind(' ')] + ' ...'


def attach_passages(text: str, index: Any, count: int) -> str:
    """``text`` followed by the ``count`` corpus passages that best match it."""
    hits = index.search(text, count) if index is not None and count else []
    if not hits:
        return text
    refs = '\n'.join(f'[{p.title}] {excerpt(p.text)}' for p in hits)
    return f'{text}\n\nReference passages (use them only if they answer this):\n{refs}'


def search_knowledge(deps, query: str) -> str:
    """The best-matching corpus passages for ``query``, each with title and source."""
    factory: Optional[Callable[[], Any]] = getattr(deps, 'knowledge', None)
    index = factory() if factory is not None else None
    if index is None:
        return NO_CORPUS
    hits = index.search(query, HITS)
    if not hits:
        return f"no passage matches '{query}' in the knowledge corpus"
    blocks = []
    for n, passage in enumerate(hits, 1):
        where = f' -- {passage.section}' if passage.section else ''
        blocks.append(f'[{n}] {passage.title}{where} ({passage.licence}; {passage.url})\n'
                      f'{excerpt(passage.text)}')
    return compact('\n'.join(blocks))
