"""What a finished run holds, as the import and the CalcJob both read it -- aiida-free.

Three parts, none of which imports `aiida` (so `frictionsim2d.aiida.ingest`,
`frictionsim2d.aiida.compare` and :mod:`frictionsim2d.aiida.calcjob` can all import this at
module level, and `ingest` stays importable without aiida-core):

* **Shared vocabulary** -- what `ingest` and the CalcJob parser must agree
  on: :data:`TIER_KEYS` (`ingest` needs it at module load time for
  `POINT_KEYS = TIER_KEYS + ('retract',)`), :func:`content_hash` and
  :func:`is_superseded`.

* **Retention** -- what a finished run keeps. One rule, two consumers.
  `ingest.load_points` derives each point's aux files from
  :func:`retained_paths`. The import-only `LammpsFrictionCalcJob` retrieves
  from :func:`retrieve_patterns`, narrowed per deck kind
  (`calcjob._temporary_list`): results, end snapshots, loaded structures and
  hold traces are retrieved and stored as outputs, while the run's inputs
  (decks, `system.in.settings`, `provenance/`, `build/*.lmp`, potentials)
  are stored as content-addressed CalcJob inputs read straight from the run
  directory. Root-level `*.out` and `log*.lammps` of a multi-deck local run
  are retained here but not yet stored by the CalcJob path. Trajectories
  (`visuals/`, `*.lammpstrj`) are deliberately excluded: spec U3 keeps
  visual files on disk for verification only. Of `build/`, only the `*.lmp`
  structures are kept.

* **Deck discovery** -- which `lammps/*.in` decks a run holds. Each deck
  becomes one import-only CalcJob, so this part decides which decks exist,
  which load points each covers and which log belongs to each. `LoadPoint`
  is only named for typing (under `TYPE_CHECKING`) and :func:`points_for`
  duck-types on `.load`/`.angle`/`.speed`, so any object carrying those
  attributes works. The point filter inverts
  `core.utils.tokens.format_numeric_token` exactly as `hpc/manifest.py`
  does: `m` is a minus sign, `p` a decimal point.
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

__all__ = ['TIER_KEYS', 'content_hash', 'is_superseded',
           'RETAINED_GLOBS', 'RETRIEVE_LIST', 'retained_paths', 'retrieve_patterns',
           'Deck', 'discover_decks', 'deck_log', 'points_for']

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

#: The three sampling tiers that become stored tier files. `frictionsim2d/aiida/
#: ingest.py` imports this rather than declaring its own, so `POINT_KEYS =
#: TIER_KEYS + ('retract',)` there and the CalcJob parser cannot disagree
#: about what a tier is.
TIER_KEYS = ('window', 'raw', 'peratom')


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
# Retention
# ---------------------------------------------------------------------------

#: LOCAL retention, evaluated with `pathlib.Path.glob`, where `**` recurses.
RETAINED_GLOBS: Tuple[str, ...] = (
    'provenance.json',
    'provenance/**/*',
    'lammps/*.in',
    'lammps/system.in.settings',
    'lammps/run_*.log',
    'results/*',
    'data/hold_*.txt',
    'data/final_*.data',
    'data/load_*.data',
    'build/*.lmp',
    'log*.lammps',
    '*.out',
)

#: REMOTE retention, evaluated by AiiDA's transport via
#: `aiida.engine.daemon.execmanager.retrieve_files_from_list`. Verified against
#: that function directly (real `LocalTransport`, no daemon/profile needed --
#: see `test_real_retrieval_preserves_the_tree`), because reasoning about it
#: from the docstring alone is exactly how the first version of this list
#: shipped broken:
#:   1. `(remote, local, depth)` tuples, not plain strings. A plain string is
#:      flattened to its basename (`execmanager.py:703-712`
#:      `local_names = [os.path.split(rem)[1]]`), so `results/friction_*.txt`
#:      would arrive at the top of the FolderData with no `results/` directory
#:      and the parser would find nothing.
#:   2. No `**`: the transport's glob is fnmatch-style and does not recurse, so
#:      every provenance level is named explicitly.
#:   3. Every WILDCARD entry uses `local='.'` with `depth=None`. With
#:      `depth=None`, execmanager builds each match's local name from
#:      `rel_rem` -- the match's path *relative to the remote run root* -- so
#:      the retrieved tree reproduces the run directory's own shape, and
#:      `execmanager` pre-creates each needed subdirectory first (the
#:      `depth is None or depth > 1` branch). A fixed `depth=0` local dir per
#:      category (e.g. `('results/*', 'results', 0)`) looks equivalent but
#:      collapses EVERY match from that one pattern onto the SAME local name --
#:      two files under `results/` would collide, last-writer-wins, onto a
#:      single flat file literally named `results`.
#:   4. The two genuinely LITERAL (non-wildcard) entries -- `provenance.json`
#:      and `lammps/system.in.settings` -- cannot use `depth=None`: execmanager
#:      compares it with `depth > 0` in the non-wildcard branch and raises
#:      `TypeError`. They use `depth` = the pattern's own path-segment count
#:      instead (1 and 2), which reproduces the exact same
#:      whole-path-relative-to-root effect that `None` gives a wildcard, and
#:      still triggers directory pre-creation once `depth > 1`
#:      (`lammps/system.in.settings`). `depth=0` for a nested literal path
#:      (e.g. `lammps/system.in.settings` with `local='.'`) instead drops the
#:      `lammps/` prefix and writes the bare basename to the retrieved root --
#:      confirmed with a real `LocalTransport` run before this fix.
#:
#: `lammps/run_*.log` covers the per-deck HPC job logs (see
#: `frictionsim2d/templates/hpc/_environment.j2` / `pbs_array.j2`, which write
#: `<run>/lammps/run_<deck>.log` and copy them back alongside results/,
#: visuals/ and data/); `log*.lammps` above stays for local (non-HPC) runs.
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


def retained_paths(run_dir: Path) -> List[Path]:
    """Every retained file under `run_dir`, absolute and sorted."""
    run_dir = Path(run_dir)
    found = set()
    for pattern in RETAINED_GLOBS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def retrieve_patterns(prefix: str = '') -> List[Tuple[str, str, Optional[int]]]:
    """`RETRIEVE_LIST`, optionally nested under a remote working-directory prefix.

    Only the REMOTE half of each tuple is prefixed, and the retrieved tree must
    reproduce the SAME shape under that prefix as the unprefixed tree does at
    its root -- `<prefix>/results/x.txt` next to `<prefix>/provenance.json`,
    never one nested under the prefix while the other lands flattened at the
    retrieved root.

    That consistency does not fall out of prefixing the remote string alone.
    A WILDCARD entry (`depth=None`) already reproduces the full path relative
    to the remote working directory (`execmanager`'s `rel_rem` -- see
    `RETRIEVE_LIST`'s own docstring), so prepending the prefix to the pattern
    is enough: `rel_rem` picks it up for free. A LITERAL entry's `depth` is
    instead a fixed path-SEGMENT COUNT (`tmp_rname.split('/')[-depth:]`) sized
    to the pattern's own segments (`provenance.json` -> 1,
    `lammps/system.in.settings` -> 2); left unchanged under a prefix, that
    same count now selects only the pattern's own tail and drops the prefix
    off the front, landing the two literal files at the retrieved ROOT while
    every wildcard entry lands under `<prefix>/` -- exactly the split layout
    `locator_for` cannot make sense of (it looks for `provenance.json`
    alongside `results/`, not four directories above it). Every literal
    entry's depth is therefore widened by the prefix's own segment count, so
    it takes in the whole prefixed path too.
    """
    if not prefix:
        return list(RETRIEVE_LIST)
    head = prefix.rstrip('/')
    extra = len(head.split('/'))
    return [(f'{head}/{remote}', local, depth if depth is None else depth + extra)
            for remote, local, depth in RETRIEVE_LIST]


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

_CONFLICT = '.sync-conflict-'
_TOL = 1e-9


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


def _is_conflict(path: Path) -> bool:
    return _CONFLICT in path.name


def _stems(locator: 'RunLocator') -> List[str]:
    """Deck stems: `system*` first, then the rest, each sorted."""
    stems = sorted(p.stem for p in locator.lammps_dir.glob('*.in')
                   if not _is_conflict(p))
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
        local = [p for p in locator.run_dir.glob('log*.lammps')
                 if not _is_conflict(p)]
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


def points_for(deck: Deck, points: List['LoadPoint']) -> List['LoadPoint']:
    """The load points this deck covers (all of them for an empty filter)."""
    return [p for p in points
            if all(abs(getattr(p, key) - value) <= _TOL
                   for key, value in deck.point_filter.items())]
