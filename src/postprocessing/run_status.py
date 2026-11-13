"""Which load points a run was configured to produce, and how each one ended.

The reader used to know only the points that left a results file; a load that
crashed, or one never reached because an earlier load of the same script died,
vanished without trace. Here the expected grid comes from the run's own
`provenance/config.json`, so every configured point gets a status:

* `complete`    -- every tier present has its expected rows (one row of slack).
* `truncated`   -- the files parse but are short; their frames are kept.
* `unreadable`  -- the window file (or the only file) exists but fails to parse.
* `no_output`   -- configured, the run exists, no results file.
* `halted`      -- stopped by the temperature guard (`common/halt.lmp`);
                   its frames are kept (they record the blow-up), but no
                   summary statistic is computed from them.
* `failed_integrity` -- complete, but a stage-3 integrity rule fired
                   (`postprocessing/integrity.py`).
* `quarantined` -- complete, but stage 3 could not judge it (missing
                   snapshot, unresolved layers) or a value is within a margin.
* `not_reached` -- `no_output` after a non-complete point of the same chain.
                   Set from the file-level statuses only: stage 3 runs after
                   it, so a failed integrity check never makes a later point
                   `not_reached`.

A chain is one in-script loop. The slide templates nest `speed` (outer) ->
load -> `angle` (inner) and ramp each load from the previous one, so a crash
stops everything after it. A guard halt does not: it costs only its own point
(or, in the AFM system ramp, its load and the loads after it).
`general.outer_loop` splits one axis into separate scripts (separate HPC
jobs): each value of that axis is its own chain.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.observables import is_truncated

logger = logging.getLogger(__name__)

STATUSES = ('complete', 'truncated', 'no_output', 'unreadable', 'halted',
            'failed_integrity', 'quarantined', 'not_reached')

#: Statuses whose point produced no frames at all.
UNOBSERVED = ('no_output', 'not_reached')

#: The guard's registry in a run's `results/` (the builders' `halt_record`).
HALT_RECORD = 'halted.txt'

#: Families whose load grid is modelled as `speed x load x angle`.
GRID_FAMILIES = ('afm', 'sheetonsheet', 'hetero', 'flake')

#: One grid evaluation per PES run, no load at all.
_PES_FAMILIES = ('pes_tip', 'pes_sheet')

#: The `outer_loop` values each family's builder actually splits on
#: (`SPLIT_OUTER_LOOPS` in `src/builders/afm/base.py` and
#: `src/builders/sheetonsheet/base.py`; hetero and flake split nothing). Any
#: other value leaves the single in-script loop, so it must not split chains.
_SPLIT_AXES = {
    'afm': {'force': 'load', 'scan_angle': 'angle'},
    'sheetonsheet': {'pressure': 'load', 'scan_speed': 'speed',
                     'scan_angle': 'angle'},
}

PointKey = Tuple[str, float, float, float]          # (load_kind, load, angle, speed)


def point_key(kind: str, load: float, angle: float, speed: float) -> PointKey:
    """Rounded to 6 decimals so 10, 10.0 and '10' agree."""
    return (kind, round(float(load), 6), round(float(angle), 6),
            round(float(speed), 6))


def load_kind_for_family(family: str) -> str:
    """`force` for afm, `none` for PES, `pressure` for the sheet-like slides."""
    if family in _PES_FAMILIES:
        return 'none'
    return 'force' if family == 'afm' else 'pressure'


def read_run_config(locator) -> Optional[dict]:
    """locator.provenance_dir / 'config.json', or None when missing/unparseable."""
    path = locator.provenance_dir / 'config.json'
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable run config %s: %s", path, exc)
        return None
    return config if isinstance(config, dict) else None


def _as_list(value: Any) -> List[float]:
    """A scalar or list config value as a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def expected_chains(config: Optional[Mapping],
                    family: str) -> Optional[List[List[PointKey]]]:
    """Chains of points in template order (speed > load > angle), split by
    general.outer_loop; None when the grid is unknown (no config, or no
    force/pressure for a slide family) or not modelled (indent).
    PES families: [[('none', 0.0, 0.0, 0.0)]]."""
    if family in _PES_FAMILIES:
        return [[point_key('none', 0.0, 0.0, 0.0)]]
    if family not in GRID_FAMILIES or not isinstance(config, Mapping):
        return None
    general = config.get('general')
    kind = load_kind_for_family(family)
    if not isinstance(general, Mapping) or general.get(kind) is None:
        return None
    try:
        loads = _as_list(general[kind])
        scan_angle = general.get('scan_angle')
        angles = [0.0] if scan_angle is None else _as_list(scan_angle)
        speeds = _as_list(general.get('scan_speed', 2.0))
        gate = general.get('scan_angle_force')
        gated = None if gate is None else {round(v, 6) for v in _as_list(gate)}
    except (TypeError, ValueError):
        return None

    def angles_for(load: float) -> Sequence[float]:
        if gated is None or round(load, 6) in gated:
            return angles
        return angles[:1]

    points = [(speed, load, angle) for speed in speeds for load in loads
              for angle in angles_for(load)]
    axis = _SPLIT_AXES.get(family, {}).get(general.get('outer_loop'))
    index = {'speed': 0, 'load': 1, 'angle': 2}.get(axis)
    chains: Dict[Any, List[PointKey]] = {}
    for speed, load, angle in points:
        split = None if index is None else round((speed, load, angle)[index], 6)
        chains.setdefault(split, []).append(point_key(kind, load, angle, speed))
    return list(chains.values())


def classify(n_rows: Dict[str, int], expected: Dict[str, int]) -> Tuple[str, str]:
    """('complete'|'truncated', detail) for a point that has frames.
    truncated if any present tier is_truncated, or 'window' expected but absent.
    detail e.g. 'window 212/500 rows'."""
    short = [f'{tier} {n_rows.get(tier, 0)}/{rows} rows'
             for tier, rows in sorted(expected.items())
             if (tier in n_rows and is_truncated(n_rows[tier], rows))
             or (tier == 'window' and tier not in n_rows)]
    return ('truncated', '; '.join(short)) if short else ('complete', '')


def read_halt_registry(locator) -> List[Dict[str, str]]:
    """The `# HALT stage=... key=value ...` lines of the run's `halted.txt`.

    One dict per line; `[]` when the guard never fired (no file).
    """
    path = locator.results_dir / HALT_RECORD
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Unreadable halt registry %s: %s", path, exc)
        return []
    return [dict(tok.split('=', 1) for tok in line[len('# HALT'):].split()
                 if '=' in tok)
            for line in text.splitlines() if line.startswith('# HALT ')]


def halt_detail(entries: Sequence[Mapping[str, str]],
                key: PointKey) -> Optional[str]:
    """Why the guard halted the point `key`, or None if it did not.

    A `slide` entry names one point. A `system` entry names the load whose AFM
    ramp halted: every angle and speed of that load is lost (the later loads
    were never written, so they come out `not_reached`). `system_equil` halted
    before any load, which loses the whole run.
    """
    kind, load = key[0], key[1]
    for entry in entries:
        stage = entry.get('stage')
        try:
            if stage == 'system_equil':
                hit = True
            elif stage == 'system':
                hit = round(float(entry['load']), 6) == load
            elif stage == 'slide':
                value = entry['load' if kind == 'force' else 'pressure']
                hit = point_key(kind, value, entry['angle'],
                                entry['speed']) == key
            else:
                hit = False
        except (KeyError, ValueError):
            continue
        if hit:
            return (f"{stage} halted at T={entry.get('T', '?')} "
                    f"step={entry.get('step', '?')}")
    return None


def mark_not_reached(chain: List[PointKey], status: Dict[PointKey, str]) -> None:
    """In chain order: a 'no_output' point after any non-complete point becomes 'not_reached'."""
    failed = False
    for key in chain:
        current = status.get(key)
        if failed and current == 'no_output':
            status[key] = 'not_reached'
        elif current is not None and current != 'complete':
            failed = True
