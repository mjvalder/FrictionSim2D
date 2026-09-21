"""What a finished run keeps.

One rule, two consumers: `ingest.py` walks a local run directory with
`retained_paths`, and `LammpsFrictionCalcJob.prepare_for_submission` builds its
`retrieve_list` from `retrieve_patterns`. They cannot drift, which is the whole
point -- AiiDA content-addresses whatever lands in the repository, so this list
IS the retention guarantee and nothing re-checksums it.

Trajectories (`visuals/`, `*.lammpstrj`) and build scratch (`build/`) are
deliberately excluded: spec U3 keeps visual files on disk for verification only.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

__all__ = ['RETAINED_GLOBS', 'RETRIEVE_LIST', 'retained_paths',
           'retrieve_patterns']

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
    'log*.lammps',
    '*.out',
)

#: REMOTE retention, evaluated by AiiDA's transport. Three differences from the
#: local list, each forced:
#:   1. `(remote, local, depth)` tuples, not plain strings. A plain string is
#:      flattened to its basename (`execmanager.py:703-712`
#:      `local_names = [os.path.split(rem)[1]]`), so `results/friction_*.txt`
#:      would arrive at the top of the FolderData with no `results/` directory
#:      and the parser would find nothing.
#:   2. No `**`: the transport's glob is fnmatch-style and does not recurse, so
#:      every provenance level is named explicitly.
#:   3. `depth=0` with an explicit local directory, so the retrieved tree is the
#:      run directory's own shape and `ingest_retrieved` can walk it unchanged.
#:
#: `lammps/run_*.log` covers the per-deck HPC job logs (see
#: `src/templates/hpc/_environment.j2` / `pbs_array.j2`, which write
#: `<run>/lammps/run_<deck>.log` and copy them back alongside results/,
#: visuals/ and data/); `log*.lammps` above stays for local (non-HPC) runs.
RETRIEVE_LIST: Tuple[Tuple[str, str, int], ...] = (
    ('provenance.json', '.', 0),
    ('provenance/*', 'provenance', 0),
    ('provenance/cif/*', 'provenance/cif', 0),
    ('provenance/potentials/*', 'provenance/potentials', 0),
    ('lammps/*.in', 'lammps', 0),
    ('lammps/system.in.settings', 'lammps', 0),
    ('lammps/run_*.log', 'lammps', 0),
    ('results/*', 'results', 0),
    ('data/hold_*.txt', 'data', 0),
    ('data/final_*.data', 'data', 0),
    ('data/load_*.data', 'data', 0),
    ('log*.lammps', '.', 0),
    ('*.out', '.', 0),
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


def retrieve_patterns(prefix: str = '') -> List[Tuple[str, str, int]]:
    """`RETRIEVE_LIST`, optionally under a remote working-directory prefix.

    Only the REMOTE half of each tuple is prefixed: the local half is where the
    file lands inside the retrieved `FolderData`, and that layout must be the
    same whether or not the run sat in a subdirectory on the cluster.
    """
    if not prefix:
        return list(RETRIEVE_LIST)
    head = prefix.rstrip('/')
    return [(f'{head}/{remote}', local, depth)
            for remote, local, depth in RETRIEVE_LIST]
