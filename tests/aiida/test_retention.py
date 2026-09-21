"""The ONE retention rule: ingest and the CalcJob keep exactly the same files."""
import asyncio

import pytest

from src.aiida.retention import (
    RETAINED_GLOBS, RETRIEVE_LIST, retained_paths, retrieve_patterns)


def _touch(root, *rels):
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x', encoding='utf-8')


def test_keeps_inputs_results_and_state_files(tmp_path):
    _touch(tmp_path,
           'provenance.json',
           'provenance/config.json', 'provenance/potentials/MoS.sw',
           'lammps/system.in', 'lammps/slide.in', 'lammps/system.in.settings',
           'results/friction_f30_a0_s2_layer2.txt',
           'results/raw_f30_a0_s2_layer2.txt',
           'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
           'log_slide.lammps', 'job.out')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {
        'provenance.json', 'provenance/config.json',
        'provenance/potentials/MoS.sw', 'lammps/system.in', 'lammps/slide.in',
        'lammps/system.in.settings',
        'results/friction_f30_a0_s2_layer2.txt',
        'results/raw_f30_a0_s2_layer2.txt',
        'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
        'log_slide.lammps', 'job.out',
    }


def test_keeps_hpc_run_logs(tmp_path):
    """R30: HPC job scripts write per-deck logs to lammps/run_<deck>.log."""
    _touch(tmp_path, 'provenance.json', 'lammps/run_slide.log',
           'lammps/run_hold.log')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {'provenance.json', 'lammps/run_slide.log',
                     'lammps/run_hold.log'}


def test_drops_trajectories_and_build_scratch(tmp_path):
    _touch(tmp_path, 'provenance.json',
           'visuals/slide.lammpstrj', 'build/tip.data', 'data/dump.lammpstrj')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {'provenance.json'}


def test_result_is_sorted_and_absolute(tmp_path):
    _touch(tmp_path, 'results/b.txt', 'results/a.txt', 'provenance.json')

    kept = retained_paths(tmp_path)

    assert all(p.is_absolute() for p in kept)
    assert kept == sorted(kept)


def _category(pattern: str) -> str:
    """The retention category a glob belongs to: its first path segment."""
    head = pattern.split('/')[0]
    return '.' if '*' in head or head.endswith('.json') else head


def test_the_two_dialects_cover_the_same_categories():
    """NOT `retrieve_patterns() == list(RETAINED_GLOBS)`.

    That comparison is a tautology -- it compares a function to the constant it
    returns and cannot fail. The two lists are deliberately DIFFERENT: local
    globs are `Path.glob` (recursive `**`), remote ones are the transport's
    fnmatch (no recursion, and tuple entries to preserve the tree). What must
    not drift is the set of categories they retain.
    """
    assert {_category(g) for g in RETAINED_GLOBS} == \
           {_category(remote) for remote, _local, _depth in RETRIEVE_LIST}


def test_remote_patterns_are_tuples_with_explicit_depth():
    """A plain string is flattened to its basename by execmanager, which puts
    `friction_*.txt` at the top level and leaves no `results/` for the parser.

    This does NOT claim the tuples "preserve the tree" by themselves -- a
    per-category local dir with `depth=0` (the previous shape of this list)
    also looked reasonable and did not: `execmanager` collapses every match of
    one wildcard pattern onto a single local name at depth 0, so two files
    under `results/` collided into one flat file. Real tree preservation is
    verified against the actual `execmanager` function by
    `test_real_retrieval_preserves_the_tree` below; this test only pins the
    shape (tuple arity, and which entries use which depth) that makes that
    verification pass.
    """
    assert all(isinstance(entry, tuple) and len(entry) == 3
               for entry in RETRIEVE_LIST)
    by_remote = {remote: (local, depth) for remote, local, depth in RETRIEVE_LIST}

    # every entry retrieves relative to the run root -- the tree comes from
    # `rel_rem`/`depth`, never from a per-category local directory
    assert all(local == '.' for local, _depth in by_remote.values())

    # the two literal (non-wildcard) entries cannot use depth=None --
    # execmanager's non-wildcard branch does `depth > 0`, which raises
    # TypeError on None -- so they carry their own path-segment count instead,
    # which reproduces the same whole-relative-path effect
    assert by_remote['provenance.json'][1] == 1
    assert by_remote['lammps/system.in.settings'][1] == 2

    # every wildcard entry uses depth=None, so multiple matches never collide
    # on one local name and directories get pre-created
    literal = {'provenance.json', 'lammps/system.in.settings'}
    assert {depth for remote, (_local, depth) in by_remote.items()
            if remote not in literal} == {None}


@pytest.mark.aiida
def test_real_retrieval_preserves_the_tree(tmp_path):
    """Ground truth against the REAL aiida-core execmanager + LocalTransport,
    not reasoning about it -- that is exactly how the first version of
    RETRIEVE_LIST (per-category local dirs, depth=0 everywhere) shipped with a
    bug that only a real transport run exposes: two files matching one
    wildcard collided onto a single flat local name, a nested literal file
    lost its directory prefix, and a `depth=1` literal crashed with TypeError.

    No AiiDA profile needed: `retrieve_files_from_list` only calls
    `calculation.get_remote_workdir()` and reads `.pk` for a debug log, so a
    plain stub is enough and this test never touches the ORM or a broker.
    """
    pytest.importorskip('aiida')
    from aiida.engine.daemon.execmanager import retrieve_files_from_list
    from aiida.transports.plugins.local import LocalTransport

    remote_dir = tmp_path / 'remote'
    retrieved_dir = tmp_path / 'retrieved'
    retrieved_dir.mkdir()
    _touch(remote_dir,
           'provenance.json',
           'provenance/config.json', 'provenance/potentials/MoS.sw',
           'provenance/cif/a.cif',
           'lammps/system.in', 'lammps/slide.in', 'lammps/system.in.settings',
           'lammps/run_slide.log', 'lammps/run_hold.log',
           'results/friction_f30_a0_s2_layer1.txt',
           'results/friction_f30_a0_s2_layer2.txt',
           'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
           'log_slide.lammps', 'job.out')

    class _FakeCalculation:  # only what retrieve_files_from_list touches
        pk = 1

        @staticmethod
        def get_remote_workdir():
            return str(remote_dir)

    transport = LocalTransport()
    transport.open()
    try:
        asyncio.run(retrieve_files_from_list(
            _FakeCalculation(), transport, str(retrieved_dir),
            list(RETRIEVE_LIST)))
    finally:
        transport.close()

    retrieved = {p.relative_to(retrieved_dir).as_posix()
                 for p in retrieved_dir.rglob('*') if p.is_file()}
    expected = {p.relative_to(remote_dir).as_posix()
                for p in retained_paths(remote_dir)}

    assert retrieved == expected
    assert {'results/friction_f30_a0_s2_layer1.txt',
            'results/friction_f30_a0_s2_layer2.txt'} <= retrieved


def test_no_remote_pattern_relies_on_recursive_globbing():
    """The transport's glob is fnmatch-style; `**` matches nothing at depth 1."""
    assert all('**' not in remote for remote, _l, _d in RETRIEVE_LIST)
    remotes = {remote for remote, _l, _d in RETRIEVE_LIST}
    # every provenance level that add_run_files stores is named explicitly
    assert {'provenance/*', 'provenance/cif/*',
            'provenance/potentials/*'} <= remotes


def test_retrieve_patterns_prefixes_only_the_remote_side():
    prefixed = retrieve_patterns('afm/h-MoS2/K300/L2')

    assert prefixed[0] == ('afm/h-MoS2/K300/L2/provenance.json', '.', 1)
    assert ('afm/h-MoS2/K300/L2/results/*', '.', None) in prefixed
    assert retrieve_patterns() == list(RETRIEVE_LIST)


def test_retrieve_list_includes_hpc_run_logs():
    """R30: the CalcJob retrieve_list must pull back lammps/run_*.log too."""
    remotes = {remote for remote, _local, _d in RETRIEVE_LIST}
    assert 'lammps/run_*.log' in remotes
