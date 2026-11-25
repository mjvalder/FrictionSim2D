"""What the real aiida transport retrieves under the CalcJob retrieve list."""
import asyncio

import pytest

from frictionsim2d.aiida.layout import RETRIEVE_LIST


def _touch(root, *rels):
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x', encoding='utf-8')


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
           'build/a.lmp',            # the as-built structure integrity checks bond against
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
        'build/a.lmp', 'log_slide.lammps', 'job.out'}
    assert {'results/friction_f30_a0_s2_layer1.txt',
            'results/friction_f30_a0_s2_layer2.txt'} <= retrieved
