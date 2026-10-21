"""Search the local knowledge corpus: attributed passages, never the network."""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import compact

HITS = 3
#: Characters of each passage shown, so three fit the result budget with their sources.
EXCERPT = 380
NO_CORPUS = ('No knowledge corpus is installed; answer from the other tools or say '
             'you do not know.')


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT:
        return text
    cut = text[:EXCERPT]
    end = cut.rfind('. ')
    return cut[:end + 1] if end > EXCERPT // 2 else cut[:cut.rfind(' ')] + ' ...'


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
                      f'{_excerpt(passage.text)}')
    return compact('\n'.join(blocks))
