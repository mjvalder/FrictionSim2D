"""The per-run `provenance` block of a `runs[]` entry.

Everything here is the same for every load point of a run, so the reader builds
it once per run directory (`run_provenance`) and each point adds only its own
ramp diagnostics (`point_provenance`). Never raises: a record that cannot be
read yields `{'error': ...}` so one bad file cannot sink the whole export.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from frictionsim2d.core.observables import AFM_HOLD_STEPS, tier_frequencies
from frictionsim2d.data.provenance import load_key, read_provenance, sha256_file

logger = logging.getLogger(__name__)

#: Record fields that describe the execution, not the physics; nested under
#: `provenance['execution']`.
EXECUTION_FIELDS = ('wall_time_s', 'n_procs', 'atoms_per_proc')

#: Families whose system phase renders `afm/system_init.lmp`, and so runs its
#: constant-load hold (`AFM_HOLD_STEPS`): AFM slides and indents alike.
_AFM_STACK_FAMILIES = ('afm', 'indent')

_TIP_KEYS = ('r', 'amorph', 'dspring', 'damping_ratio')
_GENERAL_KEYS = ('force', 'pressure', 'scan_angle', 'scan_speed',
                 'scan_angle_force', 'outer_loop', 'driving_spring',
                 'bond_spring')
_SIM_KEYS = ('timestep', 'slide_run_steps', 'drive_method', 'constraint_mode',
             'sheet_hold_steps', 'ramp_max_iterations',
             'halt_temperature_factor')


def _section(parent: Any, key: str) -> Optional[Mapping]:
    value = parent.get(key) if isinstance(parent, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _pick(source: Optional[Mapping], keys) -> Dict[str, Any]:
    return {key: (source.get(key) if source is not None else None)
            for key in keys}


def config_settings(config: Optional[Mapping],
                    family: Optional[str] = None) -> Optional[dict]:
    """The physics-relevant slice of a run's `provenance/config.json`.

    Missing keys give None, never KeyError; None when `config` is None.
    `afm_hold_steps` is the template constant, present for the AFM-stack
    families only (`family` None counts as not AFM).
    """
    if config is None:
        return None
    settings = _section(config, 'settings')
    sim = _section(settings, 'simulation')
    output = _section(settings, 'output')
    thermostat = _section(settings, 'thermostat')
    tip = _section(config, 'tip')
    simulation = _pick(sim, _SIM_KEYS)
    if family in _AFM_STACK_FAMILIES:
        simulation['afm_hold_steps'] = AFM_HOLD_STEPS
    channels = output.get('channels') if output is not None else None
    return {
        'simulation': simulation,
        'output': {
            'results_frequency': (output.get('results_frequency')
                                  if output is not None else None),
            'tier_frequency': tier_frequencies(config) or None,
            'channels': copy.deepcopy(channels),
        },
        'thermostat': copy.deepcopy(dict(thermostat))
        if thermostat is not None else None,
        'tip': _pick(tip, _TIP_KEYS) if tip is not None else None,
        'general': _pick(_section(config, 'general'), _GENERAL_KEYS),
    }


def run_provenance(locator, config: Optional[Mapping]) -> dict:
    """The per-RUN part (same for every load point of the run)."""
    path: Path = locator.provenance_file
    tokens = locator.tokens
    family = tokens.family
    settings = config_settings(config, family)
    try:
        sha = sha256_file(path)
    except OSError as exc:
        return {'error': str(exc), 'provenance_sha256': None,
                'config': settings}
    try:
        record = read_provenance(path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Unreadable provenance %s: %s", path, exc)
        return {'error': str(exc), 'provenance_sha256': sha,
                'config': settings}
    block = record.model_dump(mode='json')
    # Absent from an lj record's file (write_provenance), so absent here too:
    # an lj export is unchanged, a non-lj one carries the interlayer choice.
    if block.get('interlayer') is None:
        block.pop('interlayer', None)
    block['execution'] = {name: block.pop(name) for name in EXECUTION_FIELDS}
    for material in block['materials']:
        if material.get('pot_path'):
            material['pot_path'] = Path(material['pot_path']).name
    block['config'] = settings
    block['provenance_sha256'] = sha
    block['enriched'] = (record.lammps_version is not None
                         or record.ramp_converged is not None)
    return block


def point_provenance(run_block: dict, load: float) -> dict:
    """`run_block` plus this load's `ramp` and `load_achieved`."""
    point = dict(run_block)
    if 'error' in run_block:
        return point
    key = load_key(load)
    point['ramp'] = (run_block.get('ramp_by_load') or {}).get(key)
    point['load_achieved'] = (run_block.get('load_achieved_by_load')
                              or {}).get(key)
    return point


#: Per-run or per-load values that differ by construction; never pooled.
_UNPOOLED = frozenset({
    'provenance_sha256', 'execution', 'ramp', 'load_achieved',
    'ramp_by_load', 'load_achieved_by_load', 'unloaded_reference_z', 'error'})


def _leaves(block: Mapping, prefix: str = ''):
    """`(dotted.path, value)` for every pooled leaf; lists are leaves."""
    for name, value in block.items():
        if name in _UNPOOLED or name.startswith('seed_'):
            continue
        path = f'{prefix}{name}'
        if isinstance(value, Mapping):
            yield from _leaves(value, f'{path}.')
        else:
            yield path, value


def _value_key(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value,
                                                           sort_keys=True)


def pool_provenance(blocks: Sequence[Tuple[str, dict]]) -> dict:
    """Pool per-run provenance blocks: one entry per leaf, counted over runs.

    `blocks` is `(rel_path, run_block)`, deduplicated by `rel_path` (a run is
    counted once however many load points it has). A leaf that agrees across
    runs is `{'value': v}`, otherwise `{'values': {key: n_runs}}`. Seeds are
    unique per run by design: reported as `seeds_unique` and
    `seeds_shared_across_runs`, never pooled. Tolerates blocks of unreadable
    records (missing keys).
    """
    unique = dict(blocks)
    counts: Dict[str, Dict[str, int]] = {}
    originals: Dict[str, Dict[str, Any]] = {}
    for block in unique.values():
        for path, value in _leaves(block):
            key = _value_key(value)
            by_key = counts.setdefault(path, {})
            by_key[key] = by_key.get(key, 0) + 1
            originals.setdefault(path, {})[key] = value
    pooled: Dict[str, Any] = {}
    for path, by_key in sorted(counts.items()):
        if len(by_key) == 1:
            pooled[path] = {'value': originals[path][next(iter(by_key))]}
        else:
            pooled[path] = {'values': by_key}
    hashes: Dict[str, int] = {}
    seed_runs: Dict[Tuple[str, Any], int] = {}
    for block in unique.values():
        digest = block.get('config_hash')
        if digest is not None:
            hashes[digest] = hashes.get(digest, 0) + 1
        seen = {(name, _value_key(value)) for name, value in block.items()
                if name.startswith('seed_') and value is not None}
        for item in seen:
            seed_runs[item] = seed_runs.get(item, 0) + 1
    shared = sum(1 for n in seed_runs.values() if n > 1)
    pooled['config_hashes'] = hashes
    pooled['seeds_unique'] = shared == 0
    pooled['seeds_shared_across_runs'] = shared
    return pooled
