"""Registry-conformant results files for tests, without running LAMMPS."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.observables import rename_map
from src.data.provenance import config_hash

_SETTINGS_STUB = "thermostat:\n  type: langevin\n"

#: Whether the helpers' records carry the real ``config_hash`` of the
#: ``config.json`` they write. Off: the placeholder ``'ff' * 32`` the
#: postprocessing goldens pin. ``tests/aiida/conftest.py`` turns it on, since
#: the importer checks the hash (``calcjob.run_config_file``), as a real
#: build's record would pass.
REAL_CONFIG_HASH = False


def record_config_hash(config) -> str:
    """The ``config_hash`` a helper's record carries for ``config``."""
    return config_hash(config) if REAL_CONFIG_HASH else 'ff' * 32


def seal_config_hash(run_dir: Path, prov_dir: Path) -> None:
    """Set ``run_dir``'s record ``config_hash`` to that of
    ``prov_dir/config.json``: what a rebuild writing that config would record."""
    record_path = Path(run_dir) / 'provenance.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record['config_hash'] = config_hash(
        json.loads((Path(prov_dir) / 'config.json').read_text(encoding='utf-8')))
    record_path.write_text(json.dumps(record), encoding='utf-8')

#: The two PES families write DIFFERENT CSVs and both must be exercised: a
#: fixture that used the sheet shape for both left `pes_fmax`, `pes_flat_rms`,
#: `pes_load_mean` and `pes_load_std` -- every tip-only descriptor -- covered by
#: no test at all. These are the real template headers,
#: `src/templates/afm/pes_scan.lmp:82` and
#: `src/templates/sheetonsheet/pes_scan.lmp:57`.
PES_GRID_CSV = {
    'pes_tip': (
        '# family=pes_tip\n'
        'x,y,energy_eV,fx,fy,fz\n'
        '0,0,-1.0,0.10,0.00,-2.0\n'
        '1,0,-0.5,0.30,0.00,-2.1\n'
        '0,1,-0.6,0.00,0.20,-1.9\n'
        '1,1,-0.2,0.05,0.05,-2.2\n'),
    'pes_sheet': (
        '# family=pes_sheet\n'
        'dx,dy,energy_eV\n'
        '0,0,-1.0\n1,0,-0.5\n0,1,-0.6\n1,1,-0.2\n'),
}

#: The two families that write ONE grid CSV rather than tiered friction files
#: -- see `write_pes_grid`. Neither has any `Observable` registered for it
#: (`rename_map(family, tier)` is `{}` for every tier), so `write_results`
#: refuses them instead of silently emitting a degenerate `friction_*.txt`.
_PES_FAMILIES = ('pes_tip', 'pes_sheet')


def _header(family: str, tier: str) -> tuple[str, list[str], list[str]]:
    """`(title1, native names, channel names)` for one (family, tier)."""
    natives = sorted(rename_map(family, tier))
    channels = [rename_map(family, tier)[n] for n in natives]
    return (f'# schema=2 lf_units=nN family={family}', natives, channels)


def _results_name(family: str, tier: str, load_kind: str, load: float,
                  angle: float, speed: float, layers: int | None) -> str:
    prefix = {'window': 'friction_', 'raw': 'raw_', 'peratom': 'peratom_'}[tier]
    kind = 'f' if load_kind == 'force' else 'p'
    stem = f'{prefix}{kind}{load:g}_a{angle:g}_s{speed:g}'
    if layers is not None:
        stem += f'_layer{layers}'
    return stem + '.txt'


def load_kind_for(model: str) -> str:
    """'force' for the AFM-like models, 'pressure' for the sheet-like ones."""
    return 'force' if model in ('afm', 'indent', 'pes_tip') else 'pressure'


def write_pes_grid(run_dir: Path, family: str) -> Path:
    """Write the one `results/pes_scan.csv` grid a PES-family run produces.

    A real PES scan is a single grid evaluation, not tiered friction files:
    `src/templates/afm/pes_scan.lmp` and `src/templates/sheetonsheet/pes_scan.lmp`
    both `print` the `# family=...` line then the header then one row per grid
    point into exactly this file. `src.postprocessing.pes_scan.discover_grids`
    globs for `results/pes_scan.csv` by name.
    """
    if family not in _PES_FAMILIES:
        raise ValueError(f"write_pes_grid is for {_PES_FAMILIES}, got {family!r}")
    results_dir = run_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / 'pes_scan.csv'
    path.write_text(PES_GRID_CSV[family], encoding='utf-8')
    return path


def write_results(run_dir: Path, family: str, *, load_kind: str = 'force',
                  loads: Sequence[float] = (30.0,),
                  angles: Sequence[float] = (0.0,),
                  speeds: Sequence[float] = (2.0,),
                  tiers: Sequence[str] = ('window',),
                  layers: Optional[int] = None, n: int = 50) -> List[Path]:
    """Write results files into an EXISTING run directory, touching nothing else.

    Separate from `make_run` so a tree built by `generate_package` can be filled
    with synthetic results without its real `provenance.json` being overwritten.

    Refuses a PES family: `rename_map(family, tier)` is `{}` for both of them
    (no `Observable` lists them), so building tiered files here would silently
    write a degenerate `friction_*.txt` (a bare `# TimeStep ` header, no
    columns) instead of failing loudly. Use `write_pes_grid` for those.
    """
    if family in _PES_FAMILIES:
        raise ValueError(
            f"write_results builds tiered friction_*/raw_*/peratom_* files; "
            f"{family!r} runs write a single results/pes_scan.csv grid instead "
            "-- call write_pes_grid (make_run does this for you)")
    results_dir = run_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for tier in tiers:
        title1, natives, _channels = _header(family, tier)
        for load in loads:
            for angle in angles:
                for speed in speeds:
                    name = _results_name(family, tier, load_kind, load, angle,
                                         speed, layers)
                    rows = [f"{(i + 1) * 1000} " +
                            ' '.join(f'{(i + 1) * 0.1:.4f}' for _ in natives)
                            for i in range(n)]
                    path = results_dir / name
                    path.write_text(
                        title1 + '\n# TimeStep ' + ' '.join(natives) + '\n'
                        + '\n'.join(rows) + '\n', encoding='utf-8')
                    written.append(path)
    return written


def make_run(root: Path, rel_path: str, family: str, *,
             loads: Sequence[float] = (30.0,), angles: Sequence[float] = (0.0,),
             speeds: Sequence[float] = (2.0,),
             tiers: Sequence[str] = ('window',), n: int = 50) -> Path:
    """Write one complete run directory under `root`, and return it."""
    run_dir = root / rel_path
    model = rel_path.split('/')[0]
    layers = int(rel_path.rsplit('/L', 1)[1]) if '/L' in rel_path else None

    (run_dir / 'lammps').mkdir(parents=True, exist_ok=True)
    (run_dir / 'lammps' / 'system.in').write_text('# system\n', encoding='utf-8')

    prov_dir = (run_dir.parent if layers is not None else run_dir) / 'provenance'
    prov_dir.mkdir(parents=True, exist_ok=True)
    config = {'general': {'temp': 300}}
    (prov_dir / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    (prov_dir / 'settings.yaml').write_text(_SETTINGS_STUB, encoding='utf-8')
    (run_dir / 'provenance.json').write_text(json.dumps({
        'schema_version': '2.0.0', 'family': family, 'model': model,
        'git_commit': 'deadbeef', 'git_dirty': False,
        'config_hash': record_config_hash(config),
        'layer_type_map': {}, 'contact_cutoff': 4.8, 'cutoff_source': 'test',
        'thermostat_scheme': 'langevin', 'run_role': 'label',
        'seed_thermal_tip': None, 'seed_thermal_sub': None, 'seed_velocity': None,
        'substrate_realization_id': None, 'damping_ratio': None, 'damp_ev': None,
        'load_requested': float(loads[0]),
        'materials': [{'name': 'h-MoS2', 'role': 'sheet', 'pot_type': 'sw',
                       'pot_path': '/p/MoS.sw', 'pot_resolution_route': 'explicit',
                       'pot_file_sha256': 'ab' * 32}],
    }), encoding='utf-8')

    if family in _PES_FAMILIES:
        write_pes_grid(run_dir, family)
    else:
        write_results(run_dir, family, load_kind=load_kind_for(model), loads=loads,
                      angles=angles, speeds=speeds, tiers=tiers, layers=layers, n=n)
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    return run_dir
