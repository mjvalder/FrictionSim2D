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
from typing import List, Optional, Tuple

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
#: `src/templates/hpc/_environment.j2` / `pbs_array.j2`, which write
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
