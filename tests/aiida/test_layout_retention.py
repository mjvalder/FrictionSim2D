"""The CalcJob retrieve list: its shape, and what the real transport retrieves."""
import asyncio

import pytest

from frictionsim2d.aiida.layout import RETRIEVE_LIST


def _touch(root, *rels):
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x', encoding='utf-8')


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
    assert retrieved == {
        'provenance.json', 'provenance/config.json', 'provenance/potentials/MoS.sw',
        'provenance/cif/a.cif', 'lammps/system.in', 'lammps/slide.in',
        'lammps/system.in.settings', 'lammps/run_slide.log', 'lammps/run_hold.log',
        'results/friction_f30_a0_s2_layer1.txt', 'results/friction_f30_a0_s2_layer2.txt',
        'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
        'log_slide.lammps', 'job.out'}
    assert {'results/friction_f30_a0_s2_layer1.txt',
            'results/friction_f30_a0_s2_layer2.txt'} <= retrieved


def test_no_remote_pattern_relies_on_recursive_globbing():
    """The transport's glob is fnmatch-style; `**` matches nothing at depth 1."""
    assert all('**' not in remote for remote, _l, _d in RETRIEVE_LIST)
    remotes = {remote for remote, _l, _d in RETRIEVE_LIST}
    # every provenance level that add_run_files stores is named explicitly
    assert {'provenance/*', 'provenance/cif/*',
            'provenance/potentials/*'} <= remotes


def test_retrieve_list_includes_hpc_run_logs():
    """R30: the CalcJob retrieve_list must pull back lammps/run_*.log too."""
    remotes = {remote for remote, _local, _d in RETRIEVE_LIST}
    assert 'lammps/run_*.log' in remotes


def test_retrieve_list_keeps_the_build_structures():
    """Integrity needs the as-built structure as its bond reference."""
    assert ('build/*.lmp', '.', None) in RETRIEVE_LIST
