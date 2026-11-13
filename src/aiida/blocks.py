"""Canonical parameter blocks of one run (spec §5.2), without AiiDA.

A run's ``config.json``, ``settings.yaml`` and ``provenance.json`` are split
into named blocks -- one ``material`` and one ``potential`` block per entry of
``provenance['materials']``, plus ``thermostat``, ``geometry``, ``drive``,
``protocol`` and ``output``. Identical blocks hash alike (:func:`block_hash`),
so :mod:`src.aiida.store` stores each one once and every run that shares a
material, potential or thermostat links to the same node.

:data:`FIELD_MAP` assigns every leaf field of ``AFMSimulationConfig``,
``SheetOnSheetSimulationConfig`` and ``GlobalSettings`` to one block kind, and
:data:`EXCLUDED` names the fields that are not simulation physics.
``tests/aiida/test_blocks.py`` fails when a model grows a field neither
names. :func:`split_blocks` is driven by the same table.

Files enter a block by sha256, never by path: a path is machine-specific, the
content is what the run used.

Keep this module free of aiida imports: it runs in the fast suite without it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import numbers
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from ..data.provenance import MATERIAL_ROLES, sha256_file
from ..run.tree import provenance_dir_for

logger = logging.getLogger(__name__)

BLOCK_KINDS = ('material', 'potential', 'thermostat', 'geometry', 'drive',
               'protocol', 'output')

_MATERIAL, _POTENTIAL, _THERMOSTAT, _GEOMETRY, _DRIVE, _PROTOCOL, _OUTPUT = BLOCK_KINDS

#: Separates a block's kind from its role in a block name (``material__sheet_0``).
NAME_SEP = '__'


def _fields(kind: str, *names: str) -> Dict[str, str]:
    return {name: kind for name in names}


# --- per-section field tables -------------------------------------------------

_COMPONENT = _fields(_MATERIAL, 'mat', 'cif_path')

#: Component fields whose block content comes from the provenance entry, not
#: from config.json: ``pot_type``/``pot_path`` -> ``materials.pot_type``/
#: ``pot_file_sha256`` (what the build resolved, see _potential_block).
_COMPONENT_FROM_PROVENANCE = _fields(_POTENTIAL, 'pot_type', 'pot_path')

_SHEET = {
    **_COMPONENT,
    'lat_c': _MATERIAL,
    **_fields(_GEOMETRY, 'x', 'y', 'layers', 'stack_type', 'interlayer_g'),
}

_TIP = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'r', 'amorph'),
    **_fields(_DRIVE, 'dspring', 'damping_ratio', 'tip_x', 'tip_y',
              'tip_x_offset', 'tip_y_offset'),
}

_SUBSTRATE = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'thickness', 'x', 'y', 'amorph', 'match',
              'match_strain_tol'),
}

_FLAKE = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'shape', 'edge_length', 'rotation_deg', 'x', 'y',
              'corner_radius', 'corner_spring_k', 'corner_hold', 'center_x',
              'center_y'),
}

_GENERAL = {
    'temp': _THERMOSTAT,
    **_fields(_DRIVE, 'force', 'pressure', 'scan_angle', 'scan_angle_force',
              'scan_speed', 'outer_loop', 'bond_spring', 'driving_spring'),
    **_fields(_GEOMETRY, 'finite_sheet', 'hetero_stacking', 'twist_angle',
              'twist_angle_tol', 'twist_strain_tol', 'twist_max_supercell'),
    **_fields(_POTENTIAL, 'interlayer_style', 'interlayer_file'),
    # How friction is measured (reaction force vs group/group), not the
    # interaction itself.
    'interlayer_comparison': _PROTOCOL,
}

_PES = _fields(_PROTOCOL, 'grid_n', 'grid_n_refine', 'z_relax', 'tip_load',
               'scan_cells', 'eval_mode', 'md_steps')
_INDENT = _fields(_PROTOCOL, 'hold_steps', 'z_step', 'n_steps', 'seed')
_LATTICE_MATCH = _fields(_GEOMETRY, 'method', 'strain_tol', 'max_supercell',
                         'area_tol')

_SETTINGS = {
    'geometry': _fields(
        _GEOMETRY, 'tip_reduction_factor', 'rigid_tip', 'tip_start_clearance',
        'lat_c_default', 'finite_sheet_edge_width', 'finite_sheet_afm_edge_mode',
        'finite_sheet_edge_spring_k', 'finite_sheet_sheetonsheet_edge_mode'),
    'thermostat': _fields(_THERMOSTAT, 'type', 'time_integration',
                          'langevin_boundaries'),
    'simulation': {
        **_fields(_DRIVE, 'timestep', 'thermo', 'min_style',
                  'minimization_command', 'neighbor_list', 'neigh_modify_command',
                  'slide_run_steps', 'drive_method', 'constraint_mode',
                  'sheet_hold_steps', 'ramp_max_iterations'),
        'halt_temperature_factor': _PROTOCOL,
    },
    'quench': _fields(_DRIVE, 'quench_slab_dims', 'quench_rate', 'melt_temp',
                      'quench_temp', 'timestep', 'melt_steps', 'equilibrate_steps'),
    'output': _fields(_OUTPUT, 'dump', 'dump_frequency', 'results_frequency',
                      'channels', 'optional_channels'),
    'potential': _fields(_POTENTIAL, 'lj_cutoff', 'lj_type', 'reaxff_safezone',
                         'reaxff_mincap'),
}

#: config.json sections holding one component (or a list of them).
_COMPONENT_SECTIONS: Dict[str, Dict[str, str]] = {
    'sheet': _SHEET, 'sheets': _SHEET, 'flake_sheets': _SHEET,
    'tip': _TIP, 'sub': _SUBSTRATE, 'flake': _FLAKE,
}

#: config.json sections that are not components.
_OTHER_SECTIONS: Dict[str, Dict[str, str]] = {
    'general': _GENERAL, 'pes': _PES, 'indent': _INDENT,
    'lattice_match': _LATTICE_MATCH,
}

#: Field path -> block kind, for every assigned leaf of AFMSimulationConfig,
#: SheetOnSheetSimulationConfig and GlobalSettings (as ``settings.<...>``).
#: A field of a list of models (``sheets``) is named without an index.
FIELD_MAP: Dict[str, str] = {
    **{f'{section}.{name}': kind
       for section, table in {**_COMPONENT_SECTIONS, **_OTHER_SECTIONS}.items()
       for name, kind in table.items()},
    'lj_override': _POTENTIAL,
    **{f'settings.{section}.{name}': kind
       for section, table in _SETTINGS.items() for name, kind in table.items()},
}

#: Config field path -> block kind for the component fields a block takes
#: from provenance.json instead (see _COMPONENT_FROM_PROVENANCE). Assigned,
#: but not read by split_blocks from config.json.
PROVENANCE_SOURCED: Dict[str, str] = {
    f'{section}.{name}': kind
    for section in _COMPONENT_SECTIONS
    for name, kind in _COMPONENT_FROM_PROVENANCE.items()
}

#: provenance.json fields each block reads. ``materials.*`` is per entry,
#: ``geometry.*`` is ``BuiltGeometry`` (sheet-role material blocks only).
PROVENANCE_MAP: Dict[str, str] = {
    'materials.pot_type': _POTENTIAL,
    'materials.pot_file_sha256': _POTENTIAL,
    'materials.pot_resolution_route': _POTENTIAL,
    'geometry.lat_c': _MATERIAL,
    'geometry.cell_a': _MATERIAL,
    'geometry.cell_b': _MATERIAL,
    'damping_ratio': _THERMOSTAT,
    'damp_ev': _THERMOSTAT,
    'thermostat_scheme': _THERMOSTAT,
    'drive_k': _DRIVE,
}

_HPC = ('scheduler_type', 'queue', 'partition', 'account', 'hpc_host', 'hpc_home',
        'log_dir', 'scratch_dir', 'num_nodes', 'num_cpus', 'memory_gb',
        'walltime_hours', 'max_array_size', 'modules', 'conda_env', 'mpi_command',
        'use_tmpdir', 'preflight', 'lammps_scripts')
_AIIDA = ('profile', 'computer_label', 'max_stored_file_mb', 'skip_fraction',
          'machine')
_DATABASE = ('host', 'port', 'dbname', 'user', 'password', 'api_key', 'api_url',
             'skip_fraction')
# Analysis thresholds of the post-run integrity check, not simulation physics:
# a new threshold set judges the same run differently, it is not a new run.
_INTEGRITY = ('hole_bonds', 'isolated_radius_bonds', 'isolated_atoms',
              'ejected_atoms', 'ejected_bonds', 'heating_sigma', 'gap_rise',
              'collapse', 'frozen_std', 'gap_start_samples',
              'sheet_heating_skip_samples', 'sheet_gap_end_samples', 'margins',
              'version')

#: Leaf fields that are not simulation physics, each named. Job submission
#: (``hpc``), the import itself (``aiida``), the central DB (``database``,
#: until Task 11 removes it), integrity thresholds (``integrity``), how and
#: where the amorphous quench runs (not what it produces), and
#: ``materials_list``, which sweep expansion consumes before the model exists.
EXCLUDED = frozenset(
    [f'settings.hpc.{n}' for n in _HPC]
    + [f'settings.aiida.{n}' for n in _AIIDA]
    + [f'settings.database.{n}' for n in _DATABASE]
    + [f'settings.integrity.{n}' for n in _INTEGRITY]
    + ['settings.quench.run_local', 'settings.quench.n_procs',
       'settings.quench.allow_generate']
    + [f'{s}.materials_list' for s in ('sheet', 'sheets', 'flake_sheets')]
)

#: Fields holding a file path: a block records the file's sha256 instead.
_PATH_FIELDS = {'cif_path': 'cif_sha256', 'interlayer_file': 'interlayer_file_sha256'}

#: Where a copy of each kind of file lands under ``<run>/provenance/``.
_PROVENANCE_SUBDIR = {'cif_path': 'cif', 'interlayer_file': 'potentials'}


# --- model introspection --------------------------------------------------------

def _model_in(annotation: Any) -> Optional[type]:
    """The pydantic model inside ``annotation`` (bare, Optional or List), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (dict, Mapping):
        return None
    for arg in get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None


def leaf_field_paths(model: type, prefix: str = '') -> Iterator[str]:
    """Every leaf field path of ``model``, recursing into nested models.

    A field whose type is (an Optional or a List of) a model is descended into;
    everything else -- scalars, lists of scalars, dicts -- is a leaf.
    """
    for name, field in model.model_fields.items():
        sub = _model_in(field.annotation)
        if sub is None:
            yield f'{prefix}{name}'
        else:
            yield from leaf_field_paths(sub, f'{prefix}{name}.')


def assigned(path: str) -> bool:
    """Is the field at ``path`` part of a block (:data:`FIELD_MAP` or
    :data:`PROVENANCE_SOURCED`)?"""
    return path in FIELD_MAP or path in PROVENANCE_SOURCED


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


def block_hash(kind: str, content: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``{'kind': kind, 'content': content}``.

    ``kind`` may be a block name; its role suffix is dropped, so the Si of a
    tip and the Si of a substrate hash alike. Keys are sorted and floats
    normalised by ``repr(float(x))``, so key order and ``0.001``/``1e-3``/
    ``np.float64(0.001)`` do not matter; ``1`` and ``1.0`` stay distinct.
    """
    payload = json.dumps({'kind': block_kind(kind), 'content': _normalise(content)},
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# --- the split ------------------------------------------------------------------

#: Prefix of the value recorded for a file that could not be found.
UNRESOLVED = 'unresolved:'


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


def _pick(section: Mapping[str, Any], table: Mapping[str, str], kind: str,
          prov_dir: Optional[Path]) -> Dict[str, Any]:
    """The fields of ``section`` that ``table`` assigns to ``kind``, paths hashed."""
    out: Dict[str, Any] = {}
    for name, field_kind in table.items():
        if field_kind != kind:
            continue
        value = section.get(name)
        if name in _PATH_FIELDS:
            out[_PATH_FIELDS[name]] = _file_sha(value, name, prov_dir)
        else:
            out[name] = value
    return out


def _sections(config: Mapping[str, Any]) -> Dict[str, Any]:
    """config.json sections to read; ``sheet`` only when ``sheets`` is empty."""
    sections = {name: config.get(name) for name in (*_COMPONENT_SECTIONS, *_OTHER_SECTIONS)}
    if sections.get('sheets'):
        sections.pop('sheet')
    return sections


def _generic_block(kind: str, config: Mapping[str, Any], settings: Mapping[str, Any],
                   prov_dir: Optional[Path], components: bool = True) -> Dict[str, Any]:
    """Every config/settings field of ``kind``, nested as section -> field."""
    content: Dict[str, Any] = {}
    tables = {**(_COMPONENT_SECTIONS if components else {}), **_OTHER_SECTIONS}
    for name, value in _sections(config).items():
        if name not in tables or value is None:
            continue
        table = tables[name]
        if not any(k == kind for k in table.values()):
            continue
        if isinstance(value, list):
            content[name] = [_pick(item, table, kind, prov_dir) for item in value]
        else:
            content[name] = _pick(value, table, kind, prov_dir)
    picked = {section: _pick(settings.get(section) or {}, table, kind, prov_dir)
              for section, table in _SETTINGS.items()
              if any(k == kind for k in table.values())}
    if picked:
        content['settings'] = picked
    return content


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


def _material_block(entry: Mapping[str, Any], component: Optional[Mapping[str, Any]],
                    provenance: Mapping[str, Any], prov_dir: Optional[Path]) -> Dict:
    component = component or {}
    role = entry['role']
    # A hetero-flake material is a SheetConfig too, and has a lat_c.
    table = _SHEET if 'lat_c' in component else _COMPONENT
    block = _pick(component, table, _MATERIAL, prov_dir)
    block.pop('mat', None)
    block = {'name': component.get('mat') or entry.get('name'), **block}
    if role == 'sheet':
        geometry = provenance.get('geometry') or {}
        block['built'] = {k: geometry.get(k) for k in ('lat_c', 'cell_a', 'cell_b')}
    return block


def _potential_block(entry: Mapping[str, Any], shared: Mapping[str, Any]) -> Dict:
    return {
        'pot_type': entry.get('pot_type'),
        'pot_file_sha256': entry.get('pot_file_sha256'),
        'pot_resolution_route': entry.get('pot_resolution_route'),
        **shared,
    }


def _shared_potential(config: Mapping[str, Any], settings: Mapping[str, Any],
                      provenance: Mapping[str, Any], prov_dir: Optional[Path]) -> Dict:
    """Run-wide potential fields: settings.potential, interlayer, lj_override."""
    shared = _generic_block(_POTENTIAL, config, settings, prov_dir, components=False)
    interlayer = provenance.get('interlayer') or {}
    if interlayer.get('sha256'):
        # The builder resolved the file (a bundled default when the config
        # named none); its hash beats a path that may not exist here.
        shared.setdefault('general', {})['interlayer_file_sha256'] = interlayer['sha256']
    shared['lj_override'] = dict(config.get('lj_override') or {})
    return shared


def split_blocks(config: Mapping[str, Any], settings: Mapping[str, Any],
                 provenance: Mapping[str, Any], *,
                 run_dir: Optional[Union[str, Path]] = None,
                 provenance_dir: Optional[Union[str, Path]] = None,
                 ) -> Dict[str, Dict[str, Any]]:
    """Split one run's parameters into named, content-hashable blocks.

    Args:
        config: ``provenance/config.json`` (``model_dump`` of the run config).
        settings: ``provenance/settings.yaml``; empty falls back to
            ``config['settings']``.
        provenance: ``provenance.json`` (``ProvenanceRecord.model_dump``).
        run_dir: The run directory (the one holding ``provenance.json``; for
            AFM the ``L{n}`` level). Its ``provenance/`` directory is found
            with :func:`src.run.tree.provenance_dir_for`, and the build-time
            file copies there are hashed in preference to config paths.
        provenance_dir: The ``provenance/`` directory itself; overrides
            ``run_dir``. With neither, config paths are hashed as they are.

    Returns:
        ``material__<role>_<i>`` and ``potential__<role>_<i>`` per entry of
        ``provenance['materials']`` (``i`` counts within the role), plus
        ``thermostat``, ``geometry``, ``drive``, ``protocol`` and ``output``.
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
    shared_potential = _shared_potential(config, settings, provenance, prov_dir)
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
            entry, component, provenance, prov_dir)
        blocks[f'{_POTENTIAL}{NAME_SEP}{role}_{i}'] = _potential_block(
            entry, shared_potential)

    for kind in (_THERMOSTAT, _GEOMETRY, _DRIVE, _PROTOCOL, _OUTPUT):
        blocks[kind] = _generic_block(kind, config, settings, prov_dir)
    blocks[_THERMOSTAT]['provenance'] = {
        k: provenance.get(k) for k in ('damping_ratio', 'damp_ev', 'thermostat_scheme')}
    blocks[_DRIVE]['provenance'] = {'drive_k': provenance.get('drive_k')}
    return blocks
