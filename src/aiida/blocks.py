"""Canonical parameter blocks of one run (spec §5.2, plan *Revision 2026-11-16*).

A run's ``config.json``, ``settings.yaml`` and ``provenance.json`` are split
into named blocks: one ``material`` and one ``potential`` block per entry of
``provenance['materials']``, plus ``interaction``, ``thermostat``,
``integration``, ``drive``, ``geometry``, ``protocol``, ``output`` and
``realization``. Identical blocks hash alike (:func:`block_hash`), so
:mod:`src.aiida.store` stores each one once and every run that shares one links
to the same node.

A block holds **requested** values only, so that "which runs differ only in X"
has an answer (critique C1):

- the swept axes (:data:`SWEPT`: load, angle, speed) are point attributes, not
  block content, and the deck shape (``outer_loop``) is not identity;
- builder-computed values (relaxed ``lat_c``, ``drive_k``, ...) go to the
  unhashed :attr:`RunBlocks.derived`, never into a hash;
- content is flat (dotted field path -> value), coerced to the model's type
  (``300`` and ``300.0`` alike for a float field), and a field equal to the
  family model's default is elided, so an absent field and its default hash
  alike and a field added later changes no existing hash;
- ``kind`` and ``block_schema`` sit inside the content, so the hash is
  recomputable from the stored attributes alone (C2).

The field table, units and defaults live in :mod:`src.aiida.block_schema`;
the names are re-exported here. Files enter a block by sha256, never by path.

Keep this module free of aiida imports: it runs in the fast suite without it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import numbers
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ..data.provenance import MATERIAL_ROLES, sha256_file
from ..run.tree import provenance_dir_for
from .block_schema import (
    ANNOTATIONS, BLOCK_KINDS, BLOCK_SCHEMA_VERSION, DERIVED_PATHS, EXCLUDED,
    FIELD_MAP, INTERLAYER_DERIVED, LEGACY_DEFAULTS, LIST_SECTIONS,
    PROVENANCE_EXCLUDED, PROVENANCE_MAP, PROVENANCE_SOURCED, REALIZATION_FIELDS,
    SWEPT, RunBlocks, coerce, defaults_for, is_sweep_union, leaf_field_paths,
)

__all__ = [
    'BLOCK_KINDS', 'BLOCK_SCHEMA_VERSION', 'EXCLUDED', 'FIELD_MAP', 'NAME_SEP',
    'PROVENANCE_EXCLUDED', 'PROVENANCE_MAP', 'PROVENANCE_SOURCED', 'SWEPT',
    'UNRESOLVED', 'RunBlocks', 'assigned', 'block_hash', 'block_kind',
    'leaf_field_paths', 'split_blocks',
]

logger = logging.getLogger(__name__)

(_MATERIAL, _POTENTIAL, _INTERACTION, _THERMOSTAT, _INTEGRATION, _DRIVE, _GEOMETRY,
 _PROTOCOL, _OUTPUT, _REALIZATION) = BLOCK_KINDS

#: Kinds with one block per run (material/potential are one per material).
_RUN_KINDS = BLOCK_KINDS[2:]

#: Separates a block's kind from its role in a block name (``material__sheet_0``).
NAME_SEP = '__'

#: Prefix of the value recorded for a file that could not be found.
UNRESOLVED = 'unresolved:'

#: Fields holding a file path: a block records the file's sha256 instead, under
#: the content key ``<path>_sha256`` (``cif_path`` -> ``cif_sha256``).
_PATH_FIELDS = {'cif_path': 'cif_sha256', 'interlayer_file': 'interlayer_file_sha256'}

#: Where a copy of each kind of file lands under ``<run>/provenance/``.
_PROVENANCE_SUBDIR = {'cif_path': 'cif', 'interlayer_file': 'potentials'}

#: The content key of the interlayer file in the ``interaction`` block.
_INTERLAYER_KEY = 'general.interlayer_file_sha256'


def assigned(path: str) -> bool:
    """Is the field at ``path`` assigned (a block, a swept axis, or provenance-sourced)?"""
    return path in FIELD_MAP or path in PROVENANCE_SOURCED or path in SWEPT


# --- hashing --------------------------------------------------------------------

def block_kind(name: str) -> str:
    """The kind of a block name: ``material__sheet_0`` -> ``material``."""
    kind = name.split(NAME_SEP, 1)[0]
    if kind not in BLOCK_KINDS:
        raise ValueError(f'unknown block kind {kind!r} (from {name!r}); '
                         f'expected one of {BLOCK_KINDS}')
    return kind


def _normalise(value: Any) -> Any:
    """Floats (numpy included) as ``repr(float(x))``; ints and bools unchanged."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return repr(float(value))
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    raise TypeError(f'cannot hash a {type(value).__name__} in a parameter block')


def block_hash(content: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of one block's content.

    ``content`` is ``{'kind', 'block_schema', 'fields'}`` as
    :func:`split_blocks` builds it; the kind is read from it, so a material in
    two roles hashes alike. Keys are sorted and floats written as
    ``repr(float(x))``, so key order and ``0.001``/``1e-3``/``np.float64(0.001)``
    do not matter. ``1`` and ``1.0`` stay distinct here; :func:`split_blocks`
    has already coerced each value to its field's type.
    """
    block_kind(str(content.get('kind')))
    payload = json.dumps(_normalise(content), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _canonical(kind: str, fields: Mapping[str, Any]) -> Dict[str, Any]:
    return {'kind': kind, 'block_schema': BLOCK_SCHEMA_VERSION, 'fields': dict(fields)}


# --- files ----------------------------------------------------------------------

def _file_sha(path: Optional[str], field: str,
              prov_dir: Optional[Path]) -> Optional[str]:
    """sha256 of the file a run used for ``path``.

    The build-time copy under ``prov_dir`` (``<prov>/cif/``, ``<prov>/potentials/``
    or ``<prov>/`` itself) wins: it is what the run read, while the config path
    may have been edited since or point at another machine. The config path is
    the fallback. A non-empty path found nowhere is logged and recorded as
    ``'unresolved:<basename>'``, so two different missing files with different
    names stay distinct instead of both becoming ``None``.
    """
    if not path:
        return None
    name = Path(path).name
    candidates: List[Path] = []
    if prov_dir is not None:
        candidates += [prov_dir / _PROVENANCE_SUBDIR[field] / name, prov_dir / name]
    candidates.append(Path(path))
    for candidate in candidates:
        if candidate.is_file():
            return sha256_file(candidate)
    logger.warning('%s %r not found (looked in %s); recorded as %s%s',
                   field, path, [str(c) for c in candidates], UNRESOLVED, name)
    return f'{UNRESOLVED}{name}'


# --- requested values -------------------------------------------------------------

def _sheets(config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The sheet stack: ``sheets``, or ``[sheet]`` when it is empty.

    ``sheet`` mirrors ``sheets[0]``, so content always names ``sheets.<i>``:
    a config with only ``sheet`` and one with ``sheets == [sheet]`` hash alike.
    """
    sheets = [s for s in config.get('sheets') or [] if isinstance(s, Mapping)]
    if not sheets and isinstance(config.get('sheet'), Mapping):
        sheets = [config['sheet']]
    return sheets


def _present(config: Mapping[str, Any], settings: Mapping[str, Any]
             ) -> Iterator[Tuple[str, str, Any]]:
    """``(content path, FIELD_MAP path, raw value)`` of every requested field the
    config or settings holds, material fields aside (see _material_block)."""
    sheets = _sheets(config)
    for map_path, kind in FIELD_MAP.items():
        head, _, rest = map_path.partition('.')
        if kind == _MATERIAL or head == 'sheet':
            continue
        if head == 'settings':
            section, name = rest.split('.')
            values = settings.get(section) or {}
            if name in values:
                yield map_path, map_path, values[name]
        elif head in LIST_SECTIONS:
            items = sheets if head == 'sheets' else [
                s for s in config.get(head) or [] if isinstance(s, Mapping)]
            for i, item in enumerate(items):
                if rest in item:
                    yield f'{head}.{i}.{rest}', map_path, item[rest]
        elif not rest:
            if head in config:
                yield map_path, map_path, config[head]
        else:
            section = config.get(head)
            if isinstance(section, Mapping) and rest in section:
                yield map_path, map_path, section[rest]


def _resolve_layers(path: str, value: Any, n_layers: Optional[int],
                    single_sheet: bool) -> Any:
    """A sheet's ``layers`` list as the run's layer count.

    One sheet: the run's ``n_layers`` (which must be listed), or the list's only
    entry. In a stack ``n_layers`` is the total, and every stack builder reads
    each sheet's ``layers[0]`` (``construction/hetero_stack.py:1439``), so that
    is the value.
    """
    layers = list(value) if isinstance(value, (list, tuple)) else [value]
    if not single_sheet and layers:
        return int(layers[0])
    if n_layers is not None:
        if layers and int(n_layers) not in [int(n) for n in layers]:
            raise ValueError(f'{path}: the run has {n_layers} layers, but the config '
                             f'lists {layers}')
        return int(n_layers)
    if len(layers) == 1:
        return int(layers[0])
    raise ValueError(f'{path}: the config lists layers {layers}, and the run-level '
                     'value cannot be resolved; pass n_layers')


def _requested(path: str, map_path: str, value: Any, *, n_layers: Optional[int],
               single_sheet: bool, prov_dir: Optional[Path]) -> Tuple[str, Any]:
    """``(content key, canonical value)`` of one requested field."""
    name = map_path.rsplit('.', 1)[-1]
    if name in _PATH_FIELDS:
        return (path.rsplit('.', 1)[0] + '.' + _PATH_FIELDS[name],
                _file_sha(value, name, prov_dir))
    if map_path == 'sheets.layers':
        value = _resolve_layers(path, value, n_layers, single_sheet)
    elif isinstance(value, (list, tuple)) and is_sweep_union(ANNOTATIONS.get(map_path)):
        if len(value) != 1:
            raise ValueError(f'{path}: a list {list(value)} is left in the run config, '
                             'and the run-level value cannot be resolved')
        value = value[0]
    return path, coerce(map_path, value)


def _is_default(map_path: str, value: Any, defaults: Mapping[str, Any]) -> bool:
    """Is ``value`` the field's model default (or its legacy default)?"""
    candidates = []
    if map_path in defaults:
        candidates.append(defaults[map_path])
    legacy = LEGACY_DEFAULTS.get(BLOCK_SCHEMA_VERSION, {})
    if map_path in legacy:
        candidates.append(coerce(map_path, legacy[map_path]))
    return any(_normalise(value) == _normalise(c) for c in candidates)


def _run_fields(config: Mapping[str, Any], settings: Mapping[str, Any],
                provenance: Mapping[str, Any], *, family: str,
                n_layers: Optional[int], prov_dir: Optional[Path]
                ) -> Dict[str, Dict[str, Any]]:
    """Kind -> flat requested fields, for every run-wide kind."""
    defaults = defaults_for(family)
    single_sheet = len(_sheets(config)) == 1
    out: Dict[str, Dict[str, Any]] = {kind: {} for kind in _RUN_KINDS}
    for path, map_path, raw in _present(config, settings):
        key, value = _requested(path, map_path, raw, n_layers=n_layers,
                                single_sheet=single_sheet, prov_dir=prov_dir)
        if value is None and map_path.rsplit('.', 1)[-1] in _PATH_FIELDS:
            continue
        if not _is_default(map_path, value, defaults):
            out[FIELD_MAP[map_path]][key] = value
    sha = (provenance.get('interlayer') or {}).get('sha256')
    if sha:
        # The builder resolved the file (a bundled default when the config
        # named none); its hash beats a path that may not exist here.
        out[_INTERACTION][_INTERLAYER_KEY] = sha
    for name in REALIZATION_FIELDS:
        value = provenance.get(name)
        if value is not None:
            out[_REALIZATION][name] = coerce(name, value)
    return out


# --- materials ------------------------------------------------------------------

def _role_components(config: Mapping[str, Any]) -> Dict[str, List[Optional[Mapping]]]:
    """The config component behind each provenance entry, per role, in builder order.

    Mirrors ``AFMSimulation._write_provenance`` / the sheet family's
    ``_sheet_provenance``: sheets, then the hetero-flake materials, then a
    ``[flake]`` -- recorded only when it has a ``mat`` -- then tip, then
    substrate. Unless there are two hetero-flake materials, the sheet family
    fills each missing ``mat``/``cif_path``/``pot_path``/``pot_type`` of
    ``[Flake]`` from ``[2D]`` (``FlakeSlideSimulation._filled_flake_config``)
    before recording it, field by field; the same fill is applied here. An
    AFM ``[flake]`` must name all four, so the fill never changes it.
    """
    sheets = list(config.get('sheets') or []) or [config.get('sheet')]
    flake_sheets = list(config.get('flake_sheets') or [])
    flakes = list(flake_sheets)
    flake = config.get('flake')
    if flake is not None and len(flake_sheets) != 2:
        base = config.get('sheet') or sheets[0] or {}
        flake = {**flake, **{k: base.get(k) for k in ('mat', 'cif_path', 'pot_path',
                                                       'pot_type')
                             if flake.get(k) is None}}
    if flake is not None and flake.get('mat'):
        flakes.append(flake)
    return {'sheet': sheets, 'flake': flakes, 'tip': [config.get('tip')],
            'substrate': [config.get('sub')]}


def _without_none(fields: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def _material_block(entry: Mapping[str, Any], component: Optional[Mapping[str, Any]],
                    prov_dir: Optional[Path]) -> Dict[str, Any]:
    """What the material *is*: its name and its structure file's sha256."""
    component = component or {}
    return _canonical(_MATERIAL, _without_none({
        'name': component.get('mat') or entry.get('name'),
        'cif_sha256': _file_sha(component.get('cif_path'), 'cif_path', prov_dir),
    }))


def _potential_block(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The material's potential: its type and its file's sha256 (M6)."""
    return _canonical(_POTENTIAL, _without_none({
        'pot_type': entry.get('pot_type'),
        'pot_file_sha256': entry.get('pot_file_sha256'),
    }))


# --- derived --------------------------------------------------------------------

def _derived(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Builder-computed values, flat (``geometry.lat_c``, ``interlayer.style``)."""
    geometry = provenance.get('geometry') or {}
    interlayer = provenance.get('interlayer') or {}
    out: Dict[str, Any] = {}
    for path in DERIVED_PATHS:
        head, _, rest = path.partition('.')
        if head == 'geometry' and rest:
            out[path] = coerce(path, geometry.get(rest))
        elif head == 'interlayer' and rest in INTERLAYER_DERIVED:
            out[path] = coerce(path, interlayer.get(rest))
        else:
            out[path] = coerce(path, provenance.get(path))
    return out


# --- the split ------------------------------------------------------------------

def split_blocks(config: Mapping[str, Any], settings: Mapping[str, Any],
                 provenance: Mapping[str, Any], *, family: str,
                 n_layers: Optional[int] = None,
                 run_dir: Optional[Union[str, Path]] = None,
                 provenance_dir: Optional[Union[str, Path]] = None,
                 ) -> RunBlocks:
    """Split one run's parameters into named, content-hashable blocks.

    Args:
        config: ``provenance/config.json`` (``model_dump`` of the run config).
        settings: ``provenance/settings.yaml``; empty falls back to
            ``config['settings']``.
        provenance: ``provenance.json`` (``ProvenanceRecord.model_dump``).
        family: The build family (``provenance['family']``); picks the model
            whose defaults are elided.
        n_layers: The run's sheet layer count, to resolve a single sheet's
            ``layers`` list (an AFM run builds each count in its own ``L{n}``).
        run_dir: The run directory (the one holding ``provenance.json``; for
            AFM the ``L{n}`` level). Its ``provenance/`` directory is found
            with :func:`src.run.tree.provenance_dir_for`, and the build-time
            file copies there are hashed in preference to config paths.
        provenance_dir: The ``provenance/`` directory itself; overrides
            ``run_dir``. With neither, config paths are hashed as they are.

    Returns:
        :class:`RunBlocks`: ``material__<role>_<i>`` and ``potential__<role>_<i>``
        per entry of ``provenance['materials']`` (``i`` counts within the
        role), and one block per other kind; plus the derived values.

    Raises:
        ValueError: An unknown family or role, or a list left under a
            non-swept field whose run-level value cannot be resolved.
    """
    settings = settings or config.get('settings') or {}
    if provenance_dir is not None:
        prov_dir: Optional[Path] = Path(provenance_dir)
    elif run_dir is not None:
        prov_dir = provenance_dir_for(Path(run_dir))
    else:
        prov_dir = None
    blocks: Dict[str, Dict[str, Any]] = {}

    components = _role_components(config)
    seen = {role: 0 for role in MATERIAL_ROLES}
    for entry in provenance.get('materials') or []:
        role = entry['role']
        if role not in seen:
            raise ValueError(f'unknown material role {role!r}; expected {MATERIAL_ROLES}')
        i = seen[role]
        seen[role] += 1
        candidates = components[role]
        component = candidates[i] if i < len(candidates) else None
        blocks[f'{_MATERIAL}{NAME_SEP}{role}_{i}'] = _material_block(
            entry, component, prov_dir)
        blocks[f'{_POTENTIAL}{NAME_SEP}{role}_{i}'] = _potential_block(entry)

    fields = _run_fields(config, settings, provenance, family=family,
                         n_layers=n_layers, prov_dir=prov_dir)
    for kind in _RUN_KINDS:
        blocks[kind] = _canonical(kind, fields[kind])
    return RunBlocks(blocks=blocks, derived=_derived(provenance), family=family)
