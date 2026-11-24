"""What a finished run holds, as the import and the CalcJob both read it -- aiida-free.

* Shared vocabulary: tier keys, content hashes, Group label prefixes, the
  Syncthing-conflict rule and the superseded flag.
* :data:`RETRIEVE_LIST`: the files the import-only CalcJob retrieves. Trajectories
  (``visuals/``, ``*.lammpstrj``) are excluded on purpose; of ``build/`` only the
  ``*.lmp`` structures are kept.
* Deck discovery: which ``lammps/*.in`` decks a run holds, which load points each
  covers and which log belongs to each. The point filter inverts
  ``core.utils.tokens.format_numeric_token`` as ``hpc/manifest.py`` does: ``m`` is a
  minus sign, ``p`` a decimal point.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..run.tree import RunLocator
    from .ingest import LoadPoint

__all__ = ['ANALYSIS_GROUP_PREFIX', 'CONFLICT', 'DATASET_PREFIX', 'GROUP_PREFIX',
           'LOAD_TOL', 'RETRIEVE_LIST', 'SET_GROUP_FILTER', 'TIER_KEYS', 'Deck',
           'build_dir', 'content_hash', 'deck_log', 'discover_decks', 'group_label',
           'is_conflict', 'is_set_label', 'is_superseded', 'plain_files', 'points_for',
           'reads_build']

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

#: The three sampling tiers that become stored tier files.
TIER_KEYS = ('window', 'raw', 'peratom')

#: Group label prefixes: a set is ``friction2d/<label>``, a dataset
#: ``friction2d/datasets/<name>``, an analysis ``friction2d/analyses/<dataset>``.
#: Stored in every profile and archive: never change them.
GROUP_PREFIX = 'friction2d/'
DATASET_PREFIX = 'friction2d/datasets/'
ANALYSIS_GROUP_PREFIX = 'friction2d/analyses/'

#: QueryBuilder filter on Group labels that narrows to ``friction2d/...``; :func:`is_set_label`
#: then drops datasets and analyses (case-sensitively).
SET_GROUP_FILTER = {'label': {'like': f'{GROUP_PREFIX}%'}}


def is_set_label(label: str) -> bool:
    """Whether a Group label names a set, not a dataset or an analysis."""
    return label.startswith(GROUP_PREFIX) and not label.startswith(
        (DATASET_PREFIX, ANALYSIS_GROUP_PREFIX))


def group_label(label: str) -> str:
    """The AiiDA Group every node of the set ``label`` belongs to."""
    return f'{GROUP_PREFIX}{label}'

#: Marks a Syncthing copy of a file; the original is the record.
CONFLICT = '.sync-conflict-'

#: Tolerance when matching a load, angle or speed read from a file name.
LOAD_TOL = 1e-9


def is_conflict(path: Path) -> bool:
    """Whether ``path`` is a Syncthing conflict copy."""
    return CONFLICT in Path(path).name


def plain_files(directory: Path, pattern: str) -> List[Path]:
    """Files matching ``pattern`` in ``directory``, Syncthing copies left out."""
    if not Path(directory).is_dir():
        return []
    return [p for p in sorted(Path(directory).glob(pattern))
            if p.is_file() and not is_conflict(p)]


def is_superseded(node) -> bool:
    """Whether `node` was marked replaced by `frictionsim2d.aiida.ingest.supersede`.

    `frictionsim2d.aiida.compare` reads the flag through this function, so a
    superseded deck or point never stands for its run in a comparison.
    """
    return bool(node.base.extras.get('superseded', False))


def content_hash(paths: Sequence[Path]) -> str:
    """SHA-256 over the given files' bytes, in the given order.

    The ONE definition of a content hash over a set of result files:
    `frictionsim2d.aiida.ingest.load_points` (which hashes every entry of
    `point.tiers`, `'retract'` included) and the CalcJob parser both call
    this, so the two paths can never diverge into hashing a different file
    set.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

#: ``(remote, local, depth)`` entries for AiiDA's ``retrieve_files_from_list``. Plain
#: strings would be flattened to their basename; wildcard entries use ``depth=None``
#: so the retrieved tree keeps the run's shape, and the two literal entries carry
#: their own path-segment count (``depth=None`` raises ``TypeError`` for a literal).
#: The transport's glob does not recurse, so every ``provenance/`` level is named.
RETRIEVE_LIST: Tuple[Tuple[str, str, Optional[int]], ...] = (
    ('provenance.json', '.', 1),
    ('provenance/*', '.', None),
    ('provenance/cif/*', '.', None),
    ('provenance/potentials/*', '.', None),
    ('lammps/*.in', '.', None),
    ('lammps/system.in.settings', '.', 2),
    ('lammps/run_*.log', '.', None),
    ('results/*', '.', None),
    ('data/hold_*.txt', '.', None),
    ('data/final_*.data', '.', None),
    ('data/load_*.data', '.', None),
    ('build/*.lmp', '.', None),
    ('log*.lammps', '.', None),
    ('*.out', '.', None),
)


# ---------------------------------------------------------------------------
# Deck discovery
# ---------------------------------------------------------------------------

#: What `format_numeric_token` emits for `f"{value:g}"`: optional leading `m`,
#: digits, optional `p<digits>`, optional exponent `em<digits>` / `e<digits>`.
_NUM = r'm?\d+(?:p\d+)?(?:e[m+]?\d+)?'

#: stem pattern -> filter key. The builders emit exactly these:
#: slide_f<load>N (AFM force), slide_p<load>gpa (sheet pressure),
#: slide_a<angle>, slide_<speed>ms.
_STEM_PATTERNS = (
    (re.compile(rf'^slide_f(?P<v>{_NUM})N$'), 'load'),
    (re.compile(rf'^slide_p(?P<v>{_NUM})gpa$'), 'load'),
    (re.compile(rf'^slide_a(?P<v>{_NUM})$'), 'angle'),
    (re.compile(rf'^slide_(?P<v>{_NUM})ms$'), 'speed'),
)


#: Stems that legitimately cover every point of the run, so get `{}` silently.
_UNFILTERED_STEMS = ('slide', 'slide_refine')


@dataclass(frozen=True)
class Deck:
    """One LAMMPS input deck of a run."""
    name: str
    path: Path
    log_path: Optional[Path]
    point_filter: Dict[str, float] = field(default_factory=dict)
    writes_structures: bool = False


def _token_to_float(token: str) -> float:
    """Inverse of `format_numeric_token`."""
    sign = -1.0 if token.startswith('m') else 1.0
    body = token[1:] if token.startswith('m') else token
    body = body.replace('em', 'e-').replace('p', '.')
    return sign * float(body)


def _parse_filter(stem: str) -> Dict[str, float]:
    for pattern, key in _STEM_PATTERNS:
        match = pattern.match(stem)
        if match:
            return {key: _token_to_float(match.group('v'))}
    if stem.startswith('slide_') and stem not in _UNFILTERED_STEMS:
        logger.warning("Deck stem %r is unrecognised; its CalcJob will cover "
                       "every load point of the run.", stem)
    return {}


def _stems(locator: 'RunLocator') -> List[str]:
    """Deck stems: `system*` first, then the rest, each sorted."""
    stems = sorted(p.stem for p in plain_files(locator.lammps_dir, '*.in'))
    return ([s for s in stems if s.startswith('system')]
            + [s for s in stems if not s.startswith('system')])


def deck_log(locator: 'RunLocator', deck_name: str) -> Optional[Path]:
    """The log of one deck, or None.

    Tried in order: `lammps/run_<stem>.log` (array mode); `lammps/run_system.log`
    or `lammps/run_slide.log` (combined mode); `results/log_<stem minus
    'slide_'>.lammps` (the MLIP sheet runs: `slide_p1gpa` ->
    `results/log_p1gpa.lammps`); the single `log*.lammps` in the run root
    (local run) when the run has exactly one deck.
    """
    lammps_dir = locator.lammps_dir
    combined = 'run_system.log' if deck_name.startswith('system') else 'run_slide.log'
    for candidate in (lammps_dir / f'run_{deck_name}.log', lammps_dir / combined,
                      Path(locator.results_dir)
                      / f"log_{deck_name.removeprefix('slide_')}.lammps"):
        if candidate.is_file():
            return candidate
    if len(_stems(locator)) == 1:
        local = plain_files(locator.run_dir, 'log*.lammps')
        if len(local) == 1:
            return local[0]
    return None


def discover_decks(locator: 'RunLocator') -> List[Deck]:
    """Every deck of the run, system decks first, then sorted by name."""
    return [Deck(name=stem,
                 path=locator.lammps_dir / f'{stem}.in',
                 log_path=deck_log(locator, stem),
                 point_filter=_parse_filter(stem),
                 writes_structures=stem.startswith('system'))
            for stem in _stems(locator)]


def build_dir(locator: 'RunLocator') -> Path:
    """``build/`` of the run, or of the level above ``L{n}`` (AFM)."""
    here = Path(locator.run_dir) / 'build'
    return here if here.is_dir() else Path(locator.run_dir).parent / 'build'


def reads_build(deck: Deck, decks: Sequence[Deck]) -> bool:
    """Whether ``deck`` is handed the run's ``build/*.lmp``: the system deck, or the
    only deck when the run has no system deck."""
    return deck.writes_structures or not any(d.writes_structures for d in decks)


def points_for(deck: Deck, points: List['LoadPoint']) -> List['LoadPoint']:
    """The load points this deck covers (all of them for an empty filter)."""
    return [p for p in points
            if all(abs(getattr(p, key) - value) <= LOAD_TOL
                   for key, value in deck.point_filter.items())]
