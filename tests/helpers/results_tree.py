"""Synthetic simulation trees for the postprocessing tests.

Built on `tests.helpers.synthetic_results` -- that module owns the results-file
FORMAT (registry-conformant headers, and the two PES grid headers), this one
owns the DIRECTORY LAYOUT and the artefacts the reader needs beyond a results
file (a real schema-2 `ProvenanceRecord`, PES grids, indent retract curves and
`data/final_*.data`). Two helpers rather than one because the AiiDA store tests
need the format without these layouts, and the layouts must stay the inverse of
`output_dir_for`.
"""
from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from frictionsim2d.data.provenance import (MaterialProvenance, ProvenanceRecord,
                                 write_provenance)
from tests.helpers.synthetic_results import (PES_GRID_CSV, load_kind_for,
                                             record_config_hash, write_results)

#: family -> path template, relative to the simulation root. The AFM-like
#: families carry the sub/tip level and a trailing L{layers}; the sheet-like
#: ones do not. `{pot}` is '' or '/pot_<token>'.
FAMILY_LAYOUT: Dict[str, str] = {
    'afm': 'afm/{mat}/{size}/sub_aSi_tip_Si_r25/K{temp}{pot}/L{layers}',
    'indent': 'indent/{mat}/{size}/sub_aSi_tip_Si_r25/K{temp}{pot}/L{layers}',
    'pes_tip': 'pes_tip/{mat}/{size}/sub_aSi_tip_Si_r25/K{temp}{pot}/L{layers}',
    'sheetonsheet': 'sheetonsheet/{mat}/{size}/K{temp}{pot}',
    'hetero': 'sheetonsheet/{mat}/{size}/K{temp}{pot}',
    'flake': 'sheetonsheet/{mat}/{size}/K{temp}{pot}',
    'pes_sheet': 'pes_sheet/{mat}/{size}/K{temp}{pot}',
}

#: family -> top directory. hetero and flake live under sheetonsheet/, which
#: is exactly why the family cannot be read off the path.
MODEL_FOR_FAMILY: Dict[str, str] = {
    'afm': 'afm', 'indent': 'indent', 'pes_tip': 'pes_tip',
    'sheetonsheet': 'sheetonsheet', 'hetero': 'sheetonsheet',
    'flake': 'sheetonsheet', 'pes_sheet': 'pes_sheet',
}

_AFM_LIKE = ('afm', 'indent', 'pes_tip')

#: Families whose only result is an energy grid, not a friction time series.
_PES_FAMILIES = ('pes_tip', 'pes_sheet')


def _size_token(size: Tuple[float, float]) -> str:
    def fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f'{v:g}'
    return f'{fmt(size[0])}x_{fmt(size[1])}y'


def _material_token(material: str, second: Optional[str],
                    flake: Optional[str]) -> str:
    token = material if second is None else f'{material}+{second}'
    return token if flake is None else f'{token}+flake-{flake}'


def _record(family: str, material: str, second: Optional[str],
            flake: Optional[str], run_uuid: str, config: dict) -> ProvenanceRecord:
    materials = [MaterialProvenance(name=material, role='sheet',
                                    pot_path='/p/MoS.sw', pot_type='sw',
                                    pot_resolution_route='explicit',
                                    pot_file_sha256='ab' * 32)]
    if second:
        materials.append(MaterialProvenance(name=second, role='sheet',
                                            pot_path='/p/WS.sw', pot_type='sw'))
    if flake:
        materials.append(MaterialProvenance(name=flake, role='flake'))
    return ProvenanceRecord(
        schema_version='2.0.0', family=family, model=MODEL_FOR_FAMILY[family],
        run_uuid=run_uuid,
        materials=materials,
        seed_thermal_tip=None, seed_thermal_sub=None, seed_velocity=None,
        substrate_realization_id=None, damping_ratio=None, damp_ev=None,
        git_commit='deadbeef', git_dirty=False, config_hash=record_config_hash(config),
        layer_type_map={'layer_1': [1], 'layer_2': [2]},
        unloaded_reference_z={'layer_1': 0.0, 'layer_2': 6.5},
        contact_cutoff=4.8, cutoff_source='test',
        thermostat_scheme='langevin', run_role='label', load_requested=30.0,
    )


def make_family_run(root: Path, family: str, *, material: str = 'h-MoS2',
                    second_material: Optional[str] = None,
                    flake_material: Optional[str] = None,
                    size: Tuple[float, float] = (100.0, 100.0),
                    temp: int = 300, pot: Optional[str] = None,
                    layers: Optional[int] = None,
                    loads: Sequence[float] = (30.0,),
                    angles: Sequence[float] = (0.0,),
                    speeds: Sequence[float] = (2.0,),
                    tiers: Sequence[str] = ('window',),
                    n: int = 50, final_data: bool = False,
                    config: Optional[dict] = None) -> Path:
    """Write one complete run directory; return the directory.

    `config` is written as `provenance/config.json` (default: the minimal
    `{'general': {'temp': temp}}`, which carries no load grid). Pass
    `slide_config(...)` to give the reader a grid and expected row counts.
    """
    if layers is None and family in _AFM_LIKE:
        layers = 2
    rel = FAMILY_LAYOUT[family].format(
        mat=_material_token(material, second_material, flake_material),
        size=_size_token(size), temp=int(temp),
        pot=f'/pot_{pot}' if pot else '', layers=layers)
    run_dir = Path(root) / rel
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    (run_dir / 'lammps').mkdir(parents=True, exist_ok=True)
    (run_dir / 'lammps' / 'system.in').write_text('# system\n', encoding='utf-8')

    # `provenance/` sits one level up for the AFM-like families, whose
    # provenance.json lives in the L{n} level -- discover_runs.provenance_dir_for
    # resolves exactly this.
    prov_parent = run_dir.parent if family in _AFM_LIKE else run_dir
    (prov_parent / 'provenance').mkdir(parents=True, exist_ok=True)
    config = config if config is not None else {'general': {'temp': temp}}
    (prov_parent / 'provenance' / 'config.json').write_text(json.dumps(config),
                                                            encoding='utf-8')
    # A run_uuid derived from the run's path: distinct per run, yet stable,
    # so the whole-document goldens that carry it stay pinned.
    write_provenance(run_dir, _record(
        family, material, second_material, flake_material,
        str(uuid5(NAMESPACE_URL, f'friction2d-test:{rel}')), config))

    load_kind = load_kind_for(MODEL_FOR_FAMILY[family])
    if family in _PES_FAMILIES:
        # A PES run has no friction files: its whole result is one energy
        # grid, and `rename_map('pes_sheet', 'window')` is empty because the
        # registry declares no pes_* sim_types. Writing registry-derived
        # results here would produce a window frame carrying only `time`, and
        # `summarise_run` -> `grid_descriptors` would KeyError on `dx`.
        write_pes_grid(run_dir, family)
    else:
        write_results(run_dir, family, load_kind=load_kind, loads=loads,
                      angles=angles, speeds=speeds, tiers=tiers,
                      layers=layers if family in _AFM_LIKE else None, n=n)
    if final_data:
        write_final_data(run_dir, family, load_kind=load_kind,
                         load=loads[0], angle=angles[0])
    return run_dir


def slide_config(family: str, *, loads: Sequence[float],
                 angles: Sequence[float] = (0.0,),
                 speeds: Sequence[float] = (2.0,), steps: int = 50_000,
                 freq: int = 1000, outer_loop: Optional[str] = None,
                 scan_angle_force: Optional[Sequence[float]] = None,
                 channels: Optional[Dict[str, bool]] = None) -> dict:
    """A `provenance/config.json` body with a load grid and output settings.

    The shape is the top-level `model_dump` the runner writes: `general` holds
    the sweep (`force` for afm, `pressure` for the sheet-like families),
    `settings` the slide length and the window frequency, so
    `expected_tier_rows` gives `steps // freq` window rows.
    """
    kind = 'force' if family == 'afm' else 'pressure'
    return {
        'general': {
            'temp': 300, kind: list(loads), 'scan_angle': list(angles),
            'scan_speed': list(speeds), 'outer_loop': outer_loop,
            'scan_angle_force': (None if scan_angle_force is None
                                 else list(scan_angle_force)),
        },
        'settings': {
            'simulation': {'slide_run_steps': steps},
            'output': {'results_frequency': freq,
                       'channels': channels or {'window': True, 'raw': False,
                                                'peratom': False}},
        },
    }


def write_pes_grid(run_dir: Path, family: str, *, n: int = 4) -> Path:
    """A `results/pes_scan.csv` grid of `n` x `n` points.

    The header comes from `synthetic_results.PES_GRID_CSV`, which owns the
    format; only the number of rows is this function's business. The two
    families' headers differ, and both must be exercised:

    * `pes_sheet` -> `dx,dy,energy_eV`
      (`frictionsim2d/templates/sheetonsheet/pes_scan.lmp:57`)
    * `pes_tip`   -> `x,y,energy_eV,fx,fy,fz`
      (`frictionsim2d/templates/afm/pes_scan.lmp:82`)

    Getting this wrong hides real coverage rather than failing: `_grid_from_frame`
    accepts `dx|x` and `dy|y` interchangeably, so a tip grid written in the
    sheet's format parses fine -- but `grid_descriptors` computes
    `pes_fmax`/`pes_flat_rms` only when `{fx, fy}` are present and
    `pes_load_mean`/`pes_load_std` only when `fz` is, so all four tip-only
    descriptors would be exercised by no test in either plan.

    This IS a PES run's window tier -- `DataReader` reads it as one, the same
    way the store files `pes_scan` under `'window'`.
    """
    if family not in _PES_FAMILIES:
        raise ValueError(f'write_pes_grid is for {_PES_FAMILIES}, got {family!r}')
    comment, header = PES_GRID_CSV[family].splitlines()[:2]
    tip = family == 'pes_tip'
    rows: List[str] = [header]
    for i in range(n):
        for j in range(n):
            energy = -1.0 + 0.1 * (i + j)
            rows.append(f'{i}.0,{j}.0,{energy},0.1,0.2,-3.0' if tip
                        else f'{i}.0,{j}.0,{energy}')
    path = Path(run_dir) / 'results' / 'pes_scan.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comment + '\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    return path


def write_indent_curve(run_dir: Path, *, load: float = 30.0,
                       layers: int = 1, n: int = 6) -> Path:
    """A `results/indent_f<load>N_l<layer>.csv` retract force-distance curve."""
    rows = ['z_tip,fz_tip']
    for i in range(n):
        rows.append(f'{i * 0.5:g},{5.0 - 2.5 * i if i < 3 else 0.0:g}')
    path = Path(run_dir) / 'results' / f'indent_f{load:g}N_l{layers}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return path


def write_final_data(run_dir: Path, family: str, *, load_kind: str = 'force',
                     load: float = 30.0, angle: float = 0.0) -> Path:
    """A minimal `data/final_*.data` LAMMPS data file.

    Six atoms: types 1 and 2 are the two sheet layers (z 0.0 and 6.5, matching
    `_record`'s `unloaded_reference_z`), type 3 sits above them as the tip.
    """
    del family        # the name is keyed on the load, not the family
    name = (f'final_f{load:g}N_a{angle:g}.data' if load_kind == 'force'
            else f'final_p{load:g}GPa_a{angle:g}.data')
    body = [
        'LAMMPS data file', '', '6 atoms', '3 atom types', '',
        '0.0 100.0 xlo xhi', '0.0 100.0 ylo yhi', '-10.0 40.0 zlo zhi', '',
        'Atoms # atomic', '',
        '1 1 0.0 0.0 0.0', '2 1 3.0 0.0 -0.4',
        '3 2 0.0 0.0 6.5', '4 2 3.0 0.0 6.1',
        '5 3 0.0 0.0 9.6', '6 3 3.0 0.0 10.0', '',
    ]
    path = Path(run_dir) / 'data' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(body), encoding='utf-8')
    return path
