"""Vocabulary shared between `data/results.py` and `ingest.py` -- aiida-free.

`data/results.py` imports `aiida.orm.Data` at module level (it IS a Data node
type), and `data/__init__.py` eagerly imports `results.py` alongside
`simulation.py`/`simulation_set.py` (also `aiida.orm.Data` node types) -- so
importing ANY name from `src.aiida.data`, including a submodule of it, forces
`aiida` to be importable. `src.aiida.ingest` needs `TIER_KEYS` and a
content-hash helper at module load time (for `POINT_KEYS = TIER_KEYS +
('retract',)`), but every other symbol in that module defers its own `aiida`
import into a function body specifically so the module stays importable on a
machine without `aiida` installed. This module lives OUTSIDE `src.aiida.data`
(a sibling of `ingest.py`, not a child of the package with the eager
imports) so importing it triggers none of that.
"""
import hashlib
from pathlib import Path
from typing import Sequence

#: The three sampling tiers that become stored tier files. `src/aiida/
#: ingest.py` imports this rather than declaring its own, so `POINT_KEYS =
#: TIER_KEYS + ('retract',)` there and `FrictionResultsData.add_result_files`'s
#: loop cannot disagree about what a tier is.
TIER_KEYS = ('window', 'raw', 'peratom')


def is_superseded(node) -> bool:
    """Whether `node` was marked replaced by `src.aiida.ingest.supersede`.

    Both `ingest.find_results` (a superseded node must never be matched again)
    and `export._results_of` (a superseded node must never be written back
    out -- it and its replacement share the same run-relative file names, so
    exporting both would make the last one dumped win, possibly the stale
    one) read the same `base.extras` flag; this is the one place that reads
    it, so the two callers cannot drift into checking it differently.
    """
    return bool(node.base.extras.get('superseded', False))


def content_hash(paths: Sequence[Path]) -> str:
    """SHA-256 over the given files' bytes, in the given order.

    The ONE definition of a content hash over a set of result files:
    `FrictionResultsData.add_result_files`'s fallback (when no hash is passed
    in) and `src.aiida.ingest.load_points` (which hashes every entry of
    `point.tiers`, `'retract'` included) both call this, so the two paths
    can never diverge into hashing a different file set.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
