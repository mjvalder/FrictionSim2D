"""The per-run `provenance` block of a `runs[]` entry.

Everything here is the same for every load point of a run, so the reader builds
it once per run directory (`run_provenance`) and each point adds only its own
ramp diagnostics (`point_provenance`). Never raises: a record that cannot be
read yields `{'error': ...}` so one bad file cannot sink the whole export.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.core.observables import tier_frequencies
from src.data.provenance import load_key, read_provenance, sha256_file

logger = logging.getLogger(__name__)

#: Record fields that describe the execution, not the physics; nested under
#: `provenance['execution']`.
EXECUTION_FIELDS = ('wall_time_s', 'n_procs', 'atoms_per_proc')

#: Families whose slide runs the AFM `system_init` hold (`AFM_HOLD_STEPS`).
_AFM_STACK_FAMILIES = ('afm',)

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
    # pylint: disable-next=import-outside-toplevel
    from src.builders.afm.base import AFM_HOLD_STEPS
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
        record = read_provenance(
            path, family=family, model=tokens.model,
            material_name=tokens.materials[0] if tokens.materials else '')
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Unreadable provenance %s: %s", path, exc)
        return {'error': str(exc), 'provenance_sha256': sha,
                'config': settings}
    block = record.model_dump(mode='json')
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
