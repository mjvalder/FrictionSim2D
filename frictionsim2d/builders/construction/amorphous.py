"""The amorphous block library and the melt-quench simulation type.

A melt-quench takes hours, so amorphous blocks are made once and reused. A block
is reused only when its material, potential bytes and quench conditions match
(:class:`QuenchKey`) and it is large enough; it is then cropped, never tiled.

- Lookup (:func:`find_block`, :func:`nearest_blocks`) reads three roots, in
  order: the packaged ``frictionsim2d/data/materials``, the materials pack
  ``$XDG_DATA_HOME/FrictionSim2D/materials-pack`` and the user library
  :func:`amorphous_library_dir`. Only the user library is ever written, and
  only by :func:`import_block`.
- A friction build that finds no block raises :class:`AmorphousMissing`. No
  path starts a melt-quench on a miss.
- Making a block is a simulation of its own: :func:`write_quench_root` writes a
  root that ``hpc generate`` treats like any other (or that
  :func:`run_quench_root_local` runs serially), and :func:`import_quench_root`
  copies the finished blocks into the user library.

Spec: ``docs/superpowers/specs/TO-DO/2026-11-24-amorphous-quench-design.md``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple,
                    Union)

from ...core.config import GlobalSettings, QuenchRunConfig, QuenchSettings
from ...core.mlip import is_mlip

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Dims = Tuple[float, float, float]

#: The eight fields a block is keyed on, in the order messages show them.
KEY_FIELDS = ('material', 'potential_sha256', 'melt_temp', 'quench_temp', 'quench_rate',
              'timestep', 'melt_steps', 'equilibrate_steps')
#: The six quench conditions (the key minus material and potential).
CONDITION_FIELDS = ('melt_temp', 'quench_temp', 'quench_rate', 'timestep',
                    'melt_steps', 'equilibrate_steps')
_FLOAT_FIELDS = ('melt_temp', 'quench_temp', 'quench_rate', 'timestep')
_INT_FIELDS = ('melt_steps', 'equilibrate_steps')

SIDECAR_SCHEMA = 'amorphous-1'
QUENCH_SCHEMA = 'quench-1'
DECK_NAME = 'quench.in'
LOG_NAME = 'run_quench.log'
_TEMPLATE = 'common/make_amorphous.lmp'
_KEYED_NAME = re.compile(r'^amor_(?P<mat>.+?)(?:__(?P<key8>[0-9a-f]{8}))?\.lmp$')


# ---------------------------------------------------------------- the key

def file_sha256(path: PathLike) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(name: str, value: Any) -> Any:
    if name in _FLOAT_FIELDS:
        return format(float(value), '.6g')
    if name in _INT_FIELDS:
        return int(value)
    return str(value)


@dataclass(frozen=True, eq=False)
class QuenchKey:
    """What makes two amorphous blocks interchangeable.

    Floats compare to 6 significant digits (``1e12 == 1000000000000.0``): the
    canonical form writes them with ``format(x, '.6g')`` and equality is on the
    canonical form, since a hash cannot carry a tolerance window.
    """
    material: str
    potential_sha256: str
    melt_temp: float
    quench_temp: float
    quench_rate: float
    timestep: float
    melt_steps: int
    equilibrate_steps: int

    def as_dict(self) -> Dict[str, Any]:
        """The eight fields, as plain values (for sidecars and provenance)."""
        return {name: getattr(self, name) for name in KEY_FIELDS}

    def canonical(self) -> str:
        """JSON of the eight canonicalised fields, sorted keys, no whitespace."""
        return json.dumps({name: _canonical_value(name, getattr(self, name))
                           for name in KEY_FIELDS}, sort_keys=True, separators=(',', ':'))

    @property
    def sha256(self) -> str:
        """sha256 of :meth:`canonical`."""
        return hashlib.sha256(self.canonical().encode('utf-8')).hexdigest()

    @property
    def key8(self) -> str:
        """The first 8 hex digits of :attr:`sha256`: the block's file-name tag."""
        return self.sha256[:8]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuenchKey) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())

    @classmethod
    def from_settings(cls, material: str, potential: PathLike,
                      quench: QuenchSettings, **conditions: Any) -> 'QuenchKey':
        """The key for ``material`` under ``potential`` (a file) and ``quench``;
        ``conditions`` override single condition fields."""
        values = {name: conditions.get(name, getattr(quench, name))
                  for name in CONDITION_FIELDS}
        return cls(material=str(material), potential_sha256=file_sha256(potential),
                   melt_temp=float(values['melt_temp']),
                   quench_temp=float(values['quench_temp']),
                   quench_rate=float(values['quench_rate']),
                   timestep=float(values['timestep']),
                   melt_steps=int(values['melt_steps']),
                   equilibrate_steps=int(values['equilibrate_steps']))

    @classmethod
    def from_mapping(cls, meta: Mapping[str, Any]
                     ) -> Tuple[Optional['QuenchKey'], Optional[str]]:
        """``(key, None)`` from a sidecar or provenance mapping, or
        ``(None, reason)`` when a field is missing. Missing fields are never
        filled from settings: that is the silent-reuse bug this key exists to stop."""
        missing = [name for name in KEY_FIELDS if meta.get(name) in (None, '')]
        if missing:
            return None, f".json lacks {', '.join(missing)}"
        try:
            return cls(material=str(meta['material']),
                       potential_sha256=str(meta['potential_sha256']),
                       melt_temp=float(meta['melt_temp']),
                       quench_temp=float(meta['quench_temp']),
                       quench_rate=float(meta['quench_rate']),
                       timestep=float(meta['timestep']),
                       melt_steps=int(meta['melt_steps']),
                       equilibrate_steps=int(meta['equilibrate_steps'])), None
        except (TypeError, ValueError) as exc:
            return None, f".json has an unreadable key field ({exc})"

    def differs(self, wanted: 'QuenchKey') -> List[str]:
        """``['melt_temp 3000 (wanted 2500)', ...]``: the fields where this key
        is not ``wanted``."""
        out = []
        for name in KEY_FIELDS:
            mine, theirs = getattr(self, name), getattr(wanted, name)
            if _canonical_value(name, mine) != _canonical_value(name, theirs):
                if name == 'potential_sha256':
                    out.append(f'potential sha256 {mine[:8]} (wanted {theirs[:8]})')
                else:
                    out.append(f'{name} {_fmt(mine)} (wanted {_fmt(theirs)})')
        return out

    def describe(self, potential_name: str = '') -> str:
        """The ``needs:`` text of an :class:`AmorphousMissing` message."""
        pot = f'potential {potential_name} ' if potential_name else 'potential '
        return (f'{self.material}, {pot}(sha256 {self.potential_sha256[:8]}), '
                f'melt {_fmt(self.melt_temp)} K for {self.melt_steps} steps,\n'
                f'           quench to {_fmt(self.quench_temp)} K at '
                f'{self.quench_rate:.0e} K/s, timestep {_fmt(self.timestep)} ps, '
                f'relax {self.equilibrate_steps} steps  [key {self.key8}]')


def _box_text(box: Optional[Sequence[float]]) -> str:
    """``'214.8x214.8x48.3 A'`` (``'no box'`` for None)."""
    return 'x'.join(f'{v:.1f}' for v in box) + ' A' if box else 'no box'


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return format(value, 'g')
    return str(value)


def _refuse_mlip(pot_type: str) -> None:
    """A melt-quench needs a classical potential (documentation/mlip.md §1)."""
    if is_mlip(str(pot_type).lower()):
        raise ValueError(
            f"amorph = 'a' needs a classical potential: '{pot_type}' is an MLIP, and an "
            f"amorphous block is made by a melt-quench MD run that the `pair_style zero` "
            f"build surrogate cannot do. Use amorph = 'c', or quench the block with a "
            f"classical potential. See documentation/mlip.md §1.")


def quench_nsteps(key: QuenchKey) -> int:
    """Quench run length: ``dT / (rate * dt * 1e-12)`` steps (rate K/s, dt ps).

    Doubling the rate halves the run. Derived from the key, so recorded only.

    Raises:
        ValueError: ``quench_rate`` is not positive.
    """
    if key.quench_rate <= 0:
        raise ValueError(f"quench.quench_rate must be positive (K/s), got {key.quench_rate}")
    return max(round((key.melt_temp - key.quench_temp)
                     / (key.quench_rate * key.timestep * 1e-12)), 1)


# ---------------------------------------------------------------- the roots

def _data_home() -> Path:
    # Read the same way as agent/knowledge_index.py.
    return Path(os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share'))


def amorphous_library_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/amorphous``: the only place blocks are written."""
    return _data_home() / 'FrictionSim2D' / 'amorphous'


def materials_pack_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/materials-pack`` (read-only here)."""
    return _data_home() / 'FrictionSim2D' / 'materials-pack'


def packaged_materials_dir() -> Path:
    """The packaged ``frictionsim2d/data/materials`` (read-only, always)."""
    # `data/materials` has no __init__.py: resolve it under the `data` package,
    # whose files() is a plain path (a namespace package's is a MultiplexedPath).
    return Path(str(resources.files('frictionsim2d.data'))) / 'materials'


def library_roots() -> List[Tuple[str, Path]]:
    """``[(name, dir)]`` in lookup order: packaged, pack, user."""
    return [('packaged', packaged_materials_dir()), ('pack', materials_pack_dir()),
            ('user', amorphous_library_dir())]


def read_data_header(path: PathLike) -> Tuple[Optional[Dims], Optional[int]]:
    """``(box, natoms)`` from a LAMMPS data file's header, reading only the header."""
    lo_hi: Dict[str, float] = {}
    natoms = None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in itertools.islice(handle, 60):
                parts = line.split()
                if len(parts) >= 2 and parts[1] == 'atoms' and natoms is None:
                    try:
                        natoms = int(parts[0])
                    except ValueError:
                        pass
                for axis in 'xyz':
                    if len(parts) >= 4 and parts[2:4] == [f'{axis}lo', f'{axis}hi']:
                        lo_hi[axis] = float(parts[1]) - float(parts[0])
                if parts and parts[0] in ('Atoms', 'Masses') and len(lo_hi) == 3:
                    break
    except (OSError, ValueError, UnicodeDecodeError):
        return None, natoms
    box = (lo_hi['x'], lo_hi['y'], lo_hi['z']) if len(lo_hi) == 3 else None
    return box, natoms


@dataclass
class Block:
    """One ``amor_*.lmp`` found in a library root."""
    path: Path
    root: str
    root_index: int
    material: str
    meta: Dict[str, Any]
    key: Optional[QuenchKey]
    box: Optional[Dims]
    unkeyed_reason: Optional[str] = None

    @property
    def volume(self) -> float:
        """Box volume (inf without a readable box)."""
        if not self.box:
            return float('inf')
        return self.box[0] * self.box[1] * self.box[2]

    def fits(self, min_dims: Sequence[float]) -> bool:
        """At least ``min_dims`` in every axis."""
        return bool(self.box) and all(b >= float(m) for b, m in zip(self.box, min_dims))


def _sidecar(path: Path) -> Path:
    return Path(str(path) + '.json')


def _load_block(path: Path, root: str, index: int) -> Optional[Block]:
    match = _KEYED_NAME.match(path.name)
    if not match:
        return None
    material, file_key8 = match.group('mat'), match.group('key8')
    side = _sidecar(path)
    if not side.is_file():
        box, _ = read_data_header(path)
        return Block(path, root, index, material, {}, None, box, 'no .json')
    try:
        meta = json.loads(side.read_text(encoding='utf-8'))
        if not isinstance(meta, dict):
            raise ValueError('not a JSON object')
    except (OSError, ValueError) as exc:
        box, _ = read_data_header(path)
        return Block(path, root, index, material, {}, None, box, f'unreadable .json ({exc})')
    box = meta.get('box')
    box = (tuple(float(v) for v in box) if isinstance(box, (list, tuple)) and len(box) == 3
           else read_data_header(path)[0])
    key, reason = QuenchKey.from_mapping(meta)
    if key is not None and file_key8 is not None and key.key8 != file_key8:
        key, reason = None, f'key mismatch: file says {file_key8}, .json gives {key.key8}'
    return Block(path, root, index, material, meta, key, box, reason)


def scan_blocks(material: Optional[str] = None) -> List[Block]:
    """Every ``amor_*.lmp`` at the top level of each root, in root order."""
    blocks = []
    for index, (name, directory) in enumerate(library_roots()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('amor_*.lmp')):
            block = _load_block(path, name, index)
            if block is not None and (material is None or block.material == material):
                blocks.append(block)
    return blocks


def block_rows(material: Optional[str] = None) -> List[Dict[str, Any]]:
    """One JSON-able row per block of :func:`scan_blocks` (``quench list``)."""
    return [{'file': block.path.name, 'root': block.root, 'path': str(block.path),
             'key8': block.key.key8 if block.key else None,
             'unkeyed': block.unkeyed_reason,
             'box': list(block.box) if block.box else None,
             'natoms': block.meta.get('natoms'),
             **{name: getattr(block.key, name) if block.key else block.meta.get(name)
                for name in CONDITION_FIELDS}}
            for block in scan_blocks(material)]


def block_line(row: Mapping[str, Any]) -> str:
    """One :func:`block_rows` row as a table line."""
    box = 'x'.join(f'{v:.1f}' for v in row['box']) + ' A' if row['box'] else 'no box'
    key = row['key8'] or f"unkeyed ({row['unkeyed']})"
    conditions = ', '.join(f"{name}={row[name]}" for name in CONDITION_FIELDS
                           if row[name] is not None)
    return f"{row['file']}  {row['root']}  {key}  {box}  natoms={row['natoms']}  {conditions}"


def find_block(key: QuenchKey, min_dims: Sequence[float]) -> Optional[Path]:
    """The block keyed ``key`` with a box of at least ``min_dims`` (x, y, z, A).

    Several fits: the smallest box volume wins; ties go to root order
    (packaged, pack, user), then file name, so a rebuild picks the same block.
    """
    fits = [b for b in scan_blocks(key.material) if b.key == key and b.fits(min_dims)]
    if not fits:
        return None
    return min(fits, key=lambda b: (b.volume, b.root_index, b.path.name)).path


def satisfying_block(key: QuenchKey, slab: Sequence[float]) -> Optional[Path]:
    """A block that makes quenching ``slab`` again pointless, or None.

    Its key matches and either its box covers ``slab`` or it was itself
    quenched from a slab at least as large (NPT shrinks the box a little,
    so the shipped 200 x 200 x 50 A request came out 48.3 A thick).
    """
    def requested(block: Block) -> bool:
        slab_req = block.meta.get('slab_requested')
        return (isinstance(slab_req, (list, tuple)) and len(slab_req) == 3
                and all(float(r) >= float(s) for r, s in zip(slab_req, slab)))

    fits = [b for b in scan_blocks(key.material)
            if b.key == key and (b.fits(slab) or requested(b))]
    if not fits:
        return None
    return min(fits, key=lambda b: (b.volume, b.root_index, b.path.name)).path


def _shortfall(block: Block, min_dims: Optional[Sequence[float]]) -> float:
    if not min_dims:
        return 0.0
    if not block.box:
        return float('inf')
    return sum(max(float(m) - b, 0.0) for b, m in zip(block.box, min_dims))


def nearest_blocks(material: str, n: int = 3, *, key: Optional[QuenchKey] = None,
                   min_dims: Optional[Sequence[float]] = None
                   ) -> List[Tuple[Path, Dict[str, Any]]]:
    """``(path, meta)`` for up to ``n`` blocks of ``material``, closest first.

    Ordered by the number of key fields that differ from ``key`` (an unkeyed
    block counts as differing in all of them), then by size shortfall against
    ``min_dims``. ``meta`` is the sidecar plus ``root``, ``box``, ``key8``,
    ``differs`` and ``unkeyed_reason``.
    """
    ranked = []
    for block in scan_blocks(material):
        if block.key is None:
            differs: List[str] = []
            distance = len(KEY_FIELDS) + 1
        else:
            differs = block.key.differs(key) if key is not None else []
            distance = len(differs)
        meta = {**block.meta, 'root': block.root, 'box': list(block.box) if block.box else None,
                'key8': block.key.key8 if block.key else None, 'differs': differs,
                'unkeyed_reason': block.unkeyed_reason}
        ranked.append(((distance, _shortfall(block, min_dims), block.root_index,
                        block.path.name), block.path, meta))
    ranked.sort(key=lambda item: item[0])
    return [(path, meta) for _rank, path, meta in ranked[:n]]


# ---------------------------------------------------------------- members

@dataclass
class QuenchMember:
    """One block to make: one member directory of a quench root."""
    material: str
    potential: Path
    pot_type: str
    cif: Path
    slab: Dims
    key: QuenchKey
    min_dims: Dims = (0.0, 0.0, 0.0)

    @classmethod
    def make(cls, material: str, potential: PathLike, pot_type: str, cif: PathLike,
             slab: Sequence[float], quench: QuenchSettings,
             min_dims: Sequence[float] = (0.0, 0.0, 0.0), **conditions: Any) -> 'QuenchMember':
        """Validate and build a member. ``conditions`` override ``quench`` fields.

        Raises:
            ValueError: missing potential or CIF, a non-positive quench rate or slab.
        """
        _refuse_mlip(pot_type)
        potential, cif = Path(potential), Path(cif)
        if not potential.is_file():
            raise ValueError(f"quench potential '{potential}' not found")
        if not cif.is_file():
            raise ValueError(f"quench CIF '{cif}' not found")
        slab_t = tuple(float(v) for v in slab)
        if len(slab_t) != 3 or any(v <= 0 for v in slab_t):
            raise ValueError(f"quench slab must be three positive lengths (A), got {list(slab)}")
        key = QuenchKey.from_settings(material, potential, quench, **conditions)
        quench_nsteps(key)  # rate > 0
        return cls(material=str(material), potential=potential, pot_type=str(pot_type),
                   cif=cif, slab=slab_t, key=key,
                   min_dims=tuple(float(v) for v in min_dims))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready form (the GUI prefills a quench document from it)."""
        return {'material': self.material, 'potential': str(self.potential),
                'pot_type': self.pot_type, 'cif': str(self.cif), 'slab': list(self.slab),
                'min_dims': list(self.min_dims), 'key8': self.key.key8,
                **{name: getattr(self.key, name) for name in CONDITION_FIELDS}}


def _merge_members(members: Iterable[QuenchMember]) -> List[QuenchMember]:
    """One member per key, its slab and min_dims the axis-wise max."""
    merged: Dict[str, QuenchMember] = {}
    for member in members:
        prior = merged.get(member.key.key8)
        if prior is None:
            merged[member.key.key8] = member
            continue
        prior.slab = tuple(max(a, b) for a, b in zip(prior.slab, member.slab))
        prior.min_dims = tuple(max(a, b) for a, b in zip(prior.min_dims, member.min_dims))
    return list(merged.values())


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _resolve(name: str, base_dir: Optional[Path]) -> Path:
    # pylint: disable-next=import-outside-toplevel
    from ...core.utils.paths import search_paths
    for candidate in search_paths(name, base_dir):
        if candidate.is_file():
            return candidate
    return Path(name)


def expand_quench_config(section: Mapping[str, Any], settings: GlobalSettings,
                         base_dir: Optional[PathLike] = None) -> List[QuenchMember]:
    """The members of a ``[quench]`` section: ``material`` x every listed condition.

    ``{mat}`` in ``potential``/``cif`` is replaced per material; ``cif``
    defaults to ``<mat>.cif`` (then ``materials/<mat>.cif``) next to the
    config, ``slab`` to ``settings.quench.quench_slab_dims``.

    Raises:
        ValueError: an invalid section or member (see :meth:`QuenchMember.make`).
    """
    cfg = QuenchRunConfig.model_validate(dict(section))
    base = Path(base_dir) if base_dir is not None else None
    slab = cfg.slab or settings.quench.quench_slab_dims
    listed = {name: _as_list(getattr(cfg, name)) for name in CONDITION_FIELDS
              if getattr(cfg, name) is not None}
    names = list(listed)
    members = []
    for material in _as_list(cfg.material):
        potential = _resolve(cfg.potential.replace('{mat}', material), base)
        if cfg.cif:
            cif = _resolve(cfg.cif.replace('{mat}', material), base)
        else:
            cif = _resolve(f'{material}.cif', base)
            if not cif.is_file():
                cif = _resolve(f'materials/{material}.cif', base)
        for combo in itertools.product(*(listed[n] for n in names)):
            members.append(QuenchMember.make(material, potential, cfg.pot_type, cif, slab,
                                             settings.quench, **dict(zip(names, combo))))
    return _merge_members(members)


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sheet_sections(run_dict: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The sheet sections of a run: ``[2D]``/``[2D-N]`` raw, or ``sheet``/``sheets``
    as a stored ``config.json`` names them."""
    for key in ('2D', 'sheet'):
        if isinstance(run_dict.get(key), Mapping):
            return [run_dict[key]]
    numbered = sorted(k for k in run_dict if str(k).startswith('2D-'))
    sheets = [run_dict[k] for k in numbered] or list(run_dict.get('sheets') or [])
    return [s for s in sheets if isinstance(s, Mapping)]


def _periodic_sheet_span(run_dict: Mapping[str, Any]) -> Tuple[float, float]:
    """The in-plane size the periodic substrate is cut to: the sheet as built.

    One sheet: its requested size rounded to whole orthogonal cells
    (:func:`~frictionsim2d.builders.construction.components.built_monolayer_span`,
    the same rounding the build does). Two or more sheets (a hetero coincidence
    cell) or anything that cannot be resolved here: the requested ``x, y``.
    """
    sheets = _sheet_sections(run_dict)
    first = sheets[0] if sheets else {}
    requested = (_float(_as_list(first.get('x'))[0]) or 0.0,
                 _float(_as_list(first.get('y'))[0]) or 0.0)
    if len(sheets) != 1:
        return requested
    try:
        # pylint: disable-next=import-outside-toplevel
        from ...core.config import SheetConfig
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .components import built_monolayer_span
        fields = {k: v for k, v in first.items() if v is not None}
        return built_monolayer_span(SheetConfig(**fields))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Sizing the substrate from the requested sheet (%s)", exc)
        return requested


def _component_needs(run_dict: Mapping[str, Any]) -> List[Tuple[Mapping[str, Any], Dims]]:
    """``[(section, min_dims)]`` for the amorphous tip/substrate of one run.

    Tip: ``(2r, 2r, 2r)``. Substrate: ``(sub.x, sub.y, thickness)`` for a
    finite sheet; otherwise the sheet as it will be built
    (:func:`_periodic_sheet_span`) and the thickness. The build keeps its own
    size check.
    """
    needs = []
    tip = run_dict.get('tip')
    if isinstance(tip, Mapping) and str(tip.get('amorph') or 'c').strip() == 'a':
        radius = _float(tip.get('r')) or 0.0
        needs.append((tip, (2 * radius,) * 3))
    sub = run_dict.get('sub')
    if isinstance(sub, Mapping) and str(sub.get('amorph') or 'c').strip() == 'a':
        general = run_dict.get('general') or {}
        finite = str(general.get('finite_sheet', False)).strip().lower() in ('true', '1', 'yes')
        if finite:
            span = (_float(sub.get('x')) or 0.0, _float(sub.get('y')) or 0.0)
        else:
            span = _periodic_sheet_span(run_dict)
        needs.append((sub, (span[0], span[1], _float(sub.get('thickness')) or 0.0)))
    return needs


def amorphous_needs(run_dicts: Iterable[Mapping[str, Any]], settings: GlobalSettings,
                    base_dir: Optional[PathLike] = None) -> List[QuenchMember]:
    """Every amorphous block the expanded friction configs need, one per key.

    A component whose potential or CIF cannot be found, or whose potential is
    an MLIP, is skipped: config validation and the build report those themselves,
    per member.
    """
    base = Path(base_dir) if base_dir is not None else None
    members = []
    quench = settings.quench
    for run_dict in run_dicts:
        for section, dims in _component_needs(run_dict):
            potential = _resolve(str(section.get('pot_path') or ''), base)
            cif = _resolve(str(section.get('cif_path') or ''), base)
            if (not potential.is_file() or not cif.is_file()
                    or is_mlip(str(section.get('pot_type') or '').lower())):
                continue
            slab = tuple(max(float(s), d) for s, d in zip(quench.quench_slab_dims, dims))
            members.append(QuenchMember.make(str(section.get('mat')), potential,
                                             str(section.get('pot_type') or ''), cif, slab,
                                             quench, min_dims=dims))
    return _merge_members(members)


_LEGACY_REBUILD: ContextVar[bool] = ContextVar('amorphous_legacy_rebuild', default=False)


@contextmanager
def legacy_rebuild_lookup() -> Iterator[None]:
    """Within this block, a key miss may fall back to a shipped legacy block.

    For ``aiida rebuild`` only. Trees built before the keyed library record
    whatever ``settings.quench`` said (the publication set: timestep 0.001,
    melt 50000, relax 20000), while the block they were built on was the
    shipped ``amor_Si``: the old lookup checked size only. Rebuilding them must
    find that block again. Live runs, previews and ``run quench --for`` never
    enter this mode and keep strict keyed matching.
    """
    token = _LEGACY_REBUILD.set(True)
    try:
        yield
    finally:
        _LEGACY_REBUILD.reset(token)


def _legacy_fallback(key: QuenchKey, min_dims: Sequence[float]) -> Optional[Path]:
    """The pre-rework rule, in rebuild mode only: a packaged legacy block of the
    same material and potential that is large enough."""
    if not _LEGACY_REBUILD.get():
        return None
    fits = [b for b in scan_blocks(key.material)
            if b.root == 'packaged' and b.meta.get('source') == 'legacy'
            and b.meta.get('material') == key.material
            and b.meta.get('potential_sha256') == key.potential_sha256
            and b.fits(min_dims)]
    if not fits:
        return None
    block = min(fits, key=lambda b: (b.volume, b.path.name))
    differs = block.key.differs(key) if block.key is not None else ['unkeyed']
    logger.warning(
        "Rebuild: no block matches the recorded quench conditions [key %s]; using the "
        "shipped legacy block %s, which older trees were built on whatever their "
        "settings said. Block (recorded): %s", key.key8, block.path.name,
        ', '.join(d.replace('(wanted ', '(recorded ') for d in differs))
    return block.path


def resolve_block(key: QuenchKey, min_dims: Sequence[float]) -> Optional[Path]:
    """:func:`find_block`, plus the rebuild-only legacy fallback
    (:func:`legacy_rebuild_lookup`)."""
    return find_block(key, min_dims) or _legacy_fallback(key, min_dims)


def missing_members(needs: Iterable[QuenchMember]) -> List[QuenchMember]:
    """The members of ``needs`` that no library block satisfies."""
    return [m for m in needs if resolve_block(m.key, m.min_dims) is None]


def quench_members_for(config: Union[PathLike, Mapping[str, Any]], settings: GlobalSettings,
                       model: str = 'afm') -> List[QuenchMember]:
    """The quench members a friction config is missing (empty: nothing to make).

    Shared by ``run quench --for``, :class:`AmorphousMissing` and the GUI's
    "Create quench job". ``config`` is a config file or an already-parsed dict.
    """
    # pylint: disable-next=import-outside-toplevel
    from ...core.config import parse_config
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from ...run.expand import expand_config_sweeps
    base = parse_config(config if isinstance(config, Mapping) else Path(config))
    return missing_members(amorphous_needs(expand_config_sweeps(dict(base), model=model),
                                           settings))


# ---------------------------------------------------------------- the miss

def quench_command(config_file: PathLike, settings_file: Optional[PathLike] = None,
                   output_dir: str = 'DIR') -> str:
    """``FrictionSim2D run quench --for CONFIG [--settings-file F] -o DIR``."""
    parts = ['FrictionSim2D run quench --for', str(config_file)]
    if settings_file:
        parts += ['--settings-file', str(settings_file)]
    parts += ['-o', output_dir]
    return ' '.join(parts)


class AmorphousMissing(RuntimeError):
    """No library block matches what a friction build needs.

    Attributes:
        members: one :class:`QuenchMember` per missing key (what to quench).
        keys: their :class:`QuenchKey`\\ s.
        min_dims: the size each needs, ``(x, y, z)`` A, aligned with ``keys``.
        nearest: per key, :func:`nearest_blocks` output.
        command: the ``run quench --for ...`` line, or None when the friction
            config is not known (a builder-level miss; ``run_simulations``
            fills it in with :meth:`for_config`).
    """

    def __init__(self, members: Sequence[QuenchMember], *,
                 config_file: Optional[PathLike] = None,
                 settings_file: Optional[PathLike] = None) -> None:
        self.members = list(members)
        self.keys = [m.key for m in self.members]
        self.min_dims = [m.min_dims for m in self.members]
        self.nearest = [nearest_blocks(m.material, key=m.key, min_dims=m.min_dims)
                        for m in self.members]
        self.config_file = config_file
        self.settings_file = settings_file
        self.command = (quench_command(config_file, settings_file)
                        if config_file is not None else None)
        super().__init__(self.render())

    def for_config(self, config_file: PathLike,
                   settings_file: Optional[PathLike] = None) -> 'AmorphousMissing':
        """A copy naming ``config_file`` in :attr:`command`."""
        return AmorphousMissing(self.members, config_file=config_file,
                                settings_file=settings_file)

    def render(self) -> str:
        """The user-facing message (spec §5)."""
        lines = []
        for material in dict.fromkeys(m.material for m in self.members):
            lines.append(f'No amorphous {material} block matches this run.')
        for member, nearest in zip(self.members, self.nearest):
            lines.append(f'  needs:   {member.key.describe(member.potential.name)}')
            x, y, z = member.min_dims
            lines.append(f'  size:    at least {x:.1f} x {y:.1f} x {z:.1f} A')
            if not nearest:
                lines.append(f'  nearest: none for {member.material}')
            for path, meta in nearest:
                size = _box_text(meta.get('box'))
                why = (f"unkeyed: {meta['unkeyed_reason']}" if meta.get('unkeyed_reason')
                       else 'differs: ' + (', '.join(meta['differs']) or 'size only'))
                lines.append(f"  nearest: {path.name}  {size}  {meta['root']}  {why}")
        lines.append('  create the quench job (runs for hours; nothing starts until you run it):')
        if self.command:
            lines.append(f'    {self.command}            then HPC, or add --local')
            lines.append('  then: FrictionSim2D quench import DIR/quench_<timestamp>, '
                         'and rerun this command.')
        else:
            lines.append('    FrictionSim2D run quench --for <this config> -o DIR'
                         '            then HPC, or add --local')
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()

    def as_payload(self) -> Dict[str, Any]:
        """``{'quench_command', 'quench_members'}`` for a GUI worker result."""
        return {'quench_command': self.command,
                'quench_members': [m.as_dict() for m in self.members]}


def make_amorphous(mat_name: str, cif_path: PathLike, target_x: float, target_y: float,
                   target_z: float, pot_path: PathLike, pot_type: str,
                   settings: GlobalSettings) -> Path:
    """The library block for a component, by lookup only.

    Returns:
        The block's path (read-only: the caller copies it before cropping).

    Raises:
        AmorphousMissing: no block matches the key and size. Nothing is written.
        ValueError: ``pot_type`` is an MLIP (no block can be quenched with one).
    """
    _refuse_mlip(pot_type)
    key = QuenchKey.from_settings(mat_name, pot_path, settings.quench)
    min_dims = (float(target_x), float(target_y), float(target_z))
    found = resolve_block(key, min_dims)
    if found is not None:
        logger.info("Using amorphous block %s", found)
        return found
    slab = tuple(max(float(s), d) for s, d in zip(settings.quench.quench_slab_dims, min_dims))
    raise AmorphousMissing([QuenchMember(mat_name, Path(pot_path), pot_type, Path(cif_path),
                                         slab, key, min_dims)])


def preflight_amorphous(run_dicts: Iterable[Mapping[str, Any]], settings: GlobalSettings,
                        config_file: PathLike,
                        settings_file: Optional[PathLike] = None) -> None:
    """Raise one :class:`AmorphousMissing` for every block the runs lack.

    Called by ``run_simulations`` before the simulation root is created.
    """
    missing = missing_members(amorphous_needs(run_dicts, settings))
    if missing:
        raise AmorphousMissing(missing, config_file=config_file, settings_file=settings_file)


# ---------------------------------------------------------------- the quench root

def member_rel_dir(member: QuenchMember) -> Path:
    """``quench/<mat>/<key8>``: the member's directory relative to the root."""
    return Path('quench') / member.material / member.key.key8


def block_name(material: str, key8: str) -> str:
    """``amor_<mat>__<key8>.lmp``."""
    return f'amor_{material}__{key8}.lmp'


def _version() -> str:
    try:
        # pylint: disable-next=import-outside-toplevel
        from importlib.metadata import version
        return version('FrictionSim2D')
    except Exception:  # pylint: disable=broad-except
        return 'unknown'


def _git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=10,
                              check=False).stdout.strip() or 'unknown'
    except Exception:  # pylint: disable=broad-except
        return 'unknown'


def _write_member(root: Path, member: QuenchMember, settings: GlobalSettings,
                  tree_settings: Mapping[str, Any]) -> Path:
    # pylint: disable=import-outside-toplevel
    import yaml
    from ...core.config import ComponentConfig
    from ...core.utils.structures import cifread
    from ...interfaces.jinja import render_template
    from ...potentials.manager import PotentialManager
    from .components import create_orthogonal_slab  # pylint: disable=cyclic-import
    # pylint: enable=import-outside-toplevel

    rel = member_rel_dir(member)
    mdir = root / rel
    for sub in ('build', 'lammps', 'data', 'results', 'visuals',
                'provenance/potentials', 'provenance/cif'):
        (mdir / sub).mkdir(parents=True, exist_ok=True)
    (mdir / 'provenance' / 'settings.yaml').write_text(
        yaml.safe_dump(dict(tree_settings), sort_keys=False), encoding='utf-8')
    shutil.copy2(member.potential, mdir / 'provenance' / 'potentials' / member.potential.name)
    shutil.copy2(member.cif, mdir / 'provenance' / 'cif' / member.cif.name)

    base_block = f'{member.material}_crystal_block.lmp'
    create_orthogonal_slab(member.cif, mdir / 'build' / base_block, *member.slab)

    elements = cifread(member.cif)['elements']
    component = ComponentConfig(mat=member.material, pot_type=member.pot_type,
                                pot_path=str(member.potential), cif_path=str(member.cif))
    component.pot_path = (rel / 'provenance' / 'potentials' / member.potential.name).as_posix()
    pot_commands = PotentialManager(settings, use_langevin=False) \
        .get_single_component_commands(component, elements)

    key = member.key
    nsteps = quench_nsteps(key)
    # Drawn here, not in the template: a Jinja `| random` leaves the seed only
    # in the rendered script. Same contract as SimulationBase._run_seed.
    melt_seed = random.randint(1, 900_000_000)
    deck = render_template(_TEMPLATE, {
        'melt_seed': melt_seed,
        'neighbor_list': settings.simulation.neighbor_list,
        'neigh_modify_command': settings.simulation.neigh_modify_command,
        'base_block': (rel / 'build' / base_block).as_posix(),
        'pot_commands': pot_commands,
        'min_style': settings.simulation.min_style,
        'minimization_command': settings.simulation.minimization_command,
        'timestep': key.timestep,
        'melt_temp': key.melt_temp,
        'melt_steps': key.melt_steps,
        'quench_temp': key.quench_temp,
        'quench_nsteps': nsteps,
        'equilibrate_steps': key.equilibrate_steps,
        'amor_file': (rel / 'data' / block_name(member.material, key.key8)).as_posix(),
    })
    (mdir / 'lammps' / DECK_NAME).write_text(deck, encoding='utf-8')

    record = {
        'kind': 'quench', 'schema': QUENCH_SCHEMA, **key.as_dict(),
        'key8': key.key8, 'key_sha256': key.sha256,
        'slab_requested': list(member.slab), 'melt_seed': melt_seed,
        'quench_nsteps': nsteps, 'pot_type': member.pot_type,
        'potential_file': member.potential.name, 'cif_file': member.cif.name,
        'frictionsim2d_version': _version(), 'git_commit': _git_commit(),
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    (mdir / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n',
                                          encoding='utf-8')
    return mdir


def write_quench_root(members: Sequence[QuenchMember], out_dir: PathLike,
                      settings: GlobalSettings, root_name: Optional[str] = None) -> Path:
    """Write a quench simulation root: one member directory per block.

    ``<out_dir>/quench_YYYYMMDD_HHMMSS/quench/<mat>/<key8>/`` holds
    ``provenance.json``, ``provenance/`` (settings, potential, CIF),
    ``build/<mat>_crystal_block.lmp``, ``lammps/quench.in`` and the empty
    ``data/ results/ visuals/`` the HPC job copies back. Deck paths are
    root-relative: the job runs from the root. Nothing is run.

    Returns:
        The root.
    """
    # pylint: disable-next=import-outside-toplevel
    from ...core.config import redact_secrets
    root = Path(out_dir) / (root_name or datetime.now().strftime('quench_%Y%m%d_%H%M%S'))
    root.mkdir(parents=True, exist_ok=True)
    tree_settings = redact_secrets(settings).model_dump()
    for member in _merge_members(members):
        mdir = _write_member(root, member, settings, tree_settings)
        logger.info("Quench member %s: slab %s A, %d quench steps", mdir.relative_to(root),
                    'x'.join(f'{v:g}' for v in member.slab), quench_nsteps(member.key))
    return root


def is_quench_root(root: PathLike) -> bool:
    """Whether ``root`` holds quench members (``quench/*/*/provenance.json``, kind quench)."""
    for prov in Path(root).glob('quench/*/*/provenance.json'):
        try:
            if json.loads(prov.read_text(encoding='utf-8')).get('kind') == 'quench':
                return True
        except (OSError, ValueError):
            continue
    return False


def member_dirs(root: PathLike) -> List[Path]:
    """The member directories of a quench root, sorted."""
    return sorted(p.parent for p in Path(root).glob('quench/*/*/provenance.json'))


# ---------------------------------------------------------------- running locally

#: The ``lmp`` child of :func:`run_lmp_serial` while it runs, so a caller that is being
#: cancelled (the GUI's quench worker) can stop it instead of orphaning hours of MD.
_ACTIVE_LMP: Optional[subprocess.Popen] = None


def run_lmp_serial(deck: str, log: str, cwd: Path) -> None:
    """``lmp -in DECK -log LOG`` from ``cwd``, serially.

    Serial, deliberately: ``lmp`` links no MPI, so ``mpiexec -np N lmp`` runs N
    identical melt-quenches at once (N x runtime and memory for one result).

    Raises:
        subprocess.CalledProcessError: ``lmp`` exited non-zero.
        OSError: ``lmp`` could not be started.
    """
    global _ACTIVE_LMP  # pylint: disable=global-statement
    command = ['lmp', '-in', deck, '-log', log]
    with subprocess.Popen(command, cwd=str(cwd)) as child:
        _ACTIVE_LMP = child
        try:
            code = child.wait()
        except BaseException:
            # An exception or SIGTERM-driven SystemExit must not orphan lmp.
            child.kill()
            child.wait()
            raise
        finally:
            _ACTIVE_LMP = None
    if code:
        raise subprocess.CalledProcessError(code, command)


def stop_active_lmp() -> None:
    """Kill the ``lmp`` that :func:`run_lmp_serial` is waiting on, if any."""
    child = _ACTIVE_LMP
    if child is not None and child.poll() is None:
        child.kill()


def run_quench_root_local(root: PathLike) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Run every member of ``root`` serially, from the root. Hours per member.

    A member whose ``lmp`` fails is logged and the rest still run; import
    then reports it incomplete.

    Returns:
        ``(members run, [(member, error)])``.
    """
    root = Path(root)
    done: List[Path] = []
    failed: List[Tuple[Path, str]] = []
    for mdir in member_dirs(root):
        rel = mdir.relative_to(root).as_posix()
        record = json.loads((mdir / 'provenance.json').read_text(encoding='utf-8'))
        blocks = list((mdir / 'build').glob('*_crystal_block.lmp'))
        natoms = read_data_header(blocks[0])[1] if blocks else None
        total = (int(record['melt_steps']) + int(record['quench_nsteps'])
                 + int(record['equilibrate_steps']))
        logger.info("Melt-quench %s: ~%s atoms, %d MD steps (serial lmp)",
                    rel, natoms if natoms is not None else '?', total)
        try:
            run_lmp_serial(f'{rel}/lammps/{DECK_NAME}', f'{rel}/lammps/{LOG_NAME}', root)
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.error("Melt-quench %s failed: %s", rel, exc)
            failed.append((mdir, str(exc)))
            continue
        done.append(mdir)
    return done, failed


# ---------------------------------------------------------------- import

@dataclass
class QuenchImportReport:
    """What :func:`import_quench_root` did, per member."""
    imported: List[Path] = field(default_factory=list)
    unchanged: List[Path] = field(default_factory=list)
    refused: List[Tuple[Path, str]] = field(default_factory=list)
    incomplete: List[Tuple[Path, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Nothing refused and nothing incomplete."""
        return not (self.refused or self.incomplete)

    def lines(self) -> List[Tuple[str, bool]]:
        """``(text, to_stderr)`` per line of the CLI report."""
        return ([(f"✅ imported {path}", False) for path in self.imported]
                + [(f"= already in the library: {path}", False) for path in self.unchanged]
                + [(f"⚠️  {message}", False) for message in self.warnings]
                + [(f"❌ refused {member}: {reason}", True) for member, reason in self.refused]
                + [(f"❌ incomplete {member}: {reason}", True)
                   for member, reason in self.incomplete])


def _log_finished(log: Path) -> Optional[str]:
    """None when the log is a finished LAMMPS run, else what is wrong."""
    if not log.is_file():
        return f'no lammps/{LOG_NAME}'
    lines = [line.strip() for line in log.read_text(encoding='utf-8', errors='replace')
             .splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith('ERROR')]
    if errors:
        return f'lammps/{LOG_NAME} has an ERROR line: {errors[0]}'
    if not lines or not lines[-1].startswith('Total wall time'):
        return f"lammps/{LOG_NAME} does not end with 'Total wall time' (run unfinished?)"
    return None


def check_member(member_dir: PathLike) -> Tuple[Dict[str, Any], QuenchKey, Path, Dims, int]:
    """``(record, key, block, box, natoms)`` of a complete member.

    Raises:
        ValueError: the member is incomplete; the message names the missing piece.
    """
    mdir = Path(member_dir)
    prov = mdir / 'provenance.json'
    if not prov.is_file():
        raise ValueError(f'{mdir}: no provenance.json')
    try:
        record = json.loads(prov.read_text(encoding='utf-8'))
    except ValueError as exc:
        raise ValueError(f'{mdir}: provenance.json unreadable ({exc})') from exc
    if record.get('kind') != 'quench':
        raise ValueError(f"{mdir}: provenance.json kind is {record.get('kind')!r}, not 'quench'")
    key, reason = QuenchKey.from_mapping(record)
    if key is None:
        raise ValueError(f'{mdir}: provenance.json {reason[len(".json "):]}')
    if key.key8 != mdir.name:
        raise ValueError(f'{mdir}: key {key.key8} from provenance.json does not match the '
                         f'directory name {mdir.name}')
    block = mdir / 'data' / block_name(key.material, key.key8)
    if not block.is_file():
        raise ValueError(f'{mdir}: no data/{block.name} (the job has not copied it back)')
    box, natoms = read_data_header(block)
    if box is None or not natoms:
        raise ValueError(f'{mdir}: data/{block.name} has no readable box or no atoms')
    problem = _log_finished(mdir / 'lammps' / LOG_NAME)
    if problem:
        raise ValueError(f'{mdir}: {problem}')
    return record, key, block, box, natoms


def _atomic_copy(src: Path, dest: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f'.{dest.name}.', dir=dest.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write(text: str, dest: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f'.{dest.name}.', dir=dest.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def import_block(member_dir: PathLike, *, replace: bool = False, source: str = 'hpc',
                 report: Optional[QuenchImportReport] = None) -> Path:
    """Copy one complete member's block into the user library; return its path.

    The block goes in first (temp file + ``os.replace``), the sidecar last, so
    a half-written block is invisible to :func:`find_block`. An existing block
    with the same key is replaced only if the new one is larger in every axis,
    or with ``replace``; an identical block is left as is.

    Raises:
        ValueError: the member is incomplete (nothing is written).
        FileExistsError: a same-key block is not smaller and ``replace`` is False.
    """
    record, key, block, box, natoms = check_member(member_dir)
    slab = record.get('slab_requested') or []
    if len(slab) == 3 and any(b < float(s) for b, s in zip(box, slab)):
        message = (f"{member_dir}: block {_box_text(box)} is smaller "
                   f"than the requested slab {slab} (NPT shrink); the recorded box counts")
        logger.warning("%s", message)
        if report is not None:
            report.warnings.append(message)

    library = amorphous_library_dir()
    library.mkdir(parents=True, exist_ok=True)
    dest = library / block.name
    side = _sidecar(dest)
    block_sha = file_sha256(block)
    if dest.is_file():
        old_box, _ = read_data_header(dest)
        if side.is_file() and file_sha256(dest) == block_sha:
            if report is not None:
                report.unchanged.append(dest)
            return dest
        larger = old_box is not None and all(n > o for n, o in zip(box, old_box))
        if not (larger or replace):
            raise FileExistsError(
                f"{dest} exists with the same key and is not smaller "
                f"({_box_text(old_box)}); pass --replace")
        side.unlink(missing_ok=True)

    sidecar = {
        **key.as_dict(), 'key8': key.key8, 'key_sha256': key.sha256,
        'box': list(box), 'natoms': natoms, 'block_sha256': block_sha,
        'melt_seed': record.get('melt_seed'), 'quench_nsteps': record.get('quench_nsteps'),
        'pot_type': record.get('pot_type'), 'potential_file': record.get('potential_file'),
        'slab_requested': slab, 'source': source,
        'frictionsim2d_version': record.get('frictionsim2d_version') or _version(),
        'created': datetime.now().isoformat(timespec='seconds'), 'schema': SIDECAR_SCHEMA,
    }
    _atomic_copy(block, dest)
    _atomic_write(json.dumps(sidecar, indent=2) + '\n', side)
    if report is not None:
        report.imported.append(dest)
    logger.info("Imported amorphous block %s", dest)
    return dest


def import_quench_root(root: PathLike, *, replace: bool = False,
                       source: str = 'hpc') -> QuenchImportReport:
    """Import every complete member of a quench root (:func:`import_block`).

    Incomplete members are listed in the report, never half-imported.
    """
    report = QuenchImportReport()
    members = member_dirs(root)
    if not members:
        report.incomplete.append((Path(root), 'no quench/<mat>/<key8>/provenance.json found'))
    for mdir in members:
        try:
            import_block(mdir, replace=replace, source=source, report=report)
        except FileExistsError as exc:
            report.refused.append((mdir, str(exc)))
        except ValueError as exc:
            report.incomplete.append((mdir, str(exc)))
    return report
