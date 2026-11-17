"""Vocabulary shared by the import and the CalcJob parser -- aiida-free.

`src.aiida.ingest` needs `TIER_KEYS` and a content-hash helper at module load
time (for `POINT_KEYS = TIER_KEYS + ('retract',)`), and defers every `aiida`
import into a function body so it stays importable without aiida-core. This
module holds what it and :mod:`src.aiida.calcjob` must agree on, and imports
no `aiida` at all, so either can import it at module level.
"""
import hashlib
from pathlib import Path
from typing import Sequence

#: The three sampling tiers that become stored tier files. `src/aiida/
#: ingest.py` imports this rather than declaring its own, so `POINT_KEYS =
#: TIER_KEYS + ('retract',)` there and the CalcJob parser cannot disagree
#: about what a tier is.
TIER_KEYS = ('window', 'raw', 'peratom')


def is_superseded(node) -> bool:
    """Whether `node` was marked replaced by `src.aiida.ingest.supersede`.

    `src.aiida.compare` reads the flag through this function, so a
    superseded deck or point never stands for its run in a comparison.
    """
    return bool(node.base.extras.get('superseded', False))


def content_hash(paths: Sequence[Path]) -> str:
    """SHA-256 over the given files' bytes, in the given order.

    The ONE definition of a content hash over a set of result files:
    `src.aiida.ingest.load_points` (which hashes every entry of
    `point.tiers`, `'retract'` included) and the CalcJob parser both call
    this, so the two paths can never diverge into hashing a different file
    set.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
