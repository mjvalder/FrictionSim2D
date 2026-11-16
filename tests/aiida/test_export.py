"""Export, rebuild and archive over the CalcJob graph (plan Task 10).

`export_results` writes a set back into a tree `postprocess read` gives the
same statuses on; `export_archive` writes one campaign as an `.aiida` archive
gated on verdicts, closed over the campaign [I6], anonymised by default
(ruling P2-R11), per-atom data opt-in, with JSON sidecars `compare` reads and a
README [M4]. A second profile is `aiida_profile_clean.reset_storage()`.
"""
import json
import sqlite3
import zipfile

import pytest
import yaml

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = [pytest.mark.aiida, pytest.mark.filterwarnings(
    # aiida-core's own import-group warning (see tests/aiida/test_store.py).
    'ignore:Object of type <DbGroup> not in session:sqlalchemy.exc.SAWarning')]

from aiida import orm  # noqa: E402

import src.aiida.export as export_mod  # noqa: E402
from src.aiida.compare import load_blocks, load_blocks_from_sidecar  # noqa: E402
from src.aiida.export import (  # noqa: E402
    ExportClosureError, UncheckedPointsError, export_archive, export_dataset,
    export_results, import_archive, rebuild_set,
)
from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.query import related  # noqa: E402
from src.aiida.steps import build_dataset, postprocess_group  # noqa: E402
from src.aiida.store import BLOCK_HASH_EXTRA, verify  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.aiida.test_steps_integrity import _pid, _write_ruling  # noqa: E402
from tests.helpers.results_tree import make_family_run, slide_config  # noqa: E402
from tests.postprocessing.test_read_data_integrity import DECK, _afm_snapshots  # noqa: E402

LOG = """LAMMPS (29 Aug 2024 - Update 1)
Loop time of 12.5 on 4 procs for 1000 steps with 600 atoms
Total wall time: 0:01:05
"""
OPTIONS = {'skip_fraction': 0.2}
CALCJOB_TYPE = 'process.calculation.calcjob.CalcJobNode.'


@pytest.fixture
def no_profile_switch(monkeypatch):
    """`aiida_profile_clean` is already loaded; do not load another."""
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _run(root, material='h-MoS2', *, loads=(10.0, 30.0, 50.0), eject=(),
         quarantine=(), tiers=('window',)):
    """An AFM run (system + slide deck) at `loads`: `eject` fail the integrity
    check, `quarantine` lose their end snapshot."""
    run_dir = make_family_run(root, 'afm', material=material, loads=loads, n=50,
                              tiers=tiers, config=slide_config('afm', loads=loads))
    (run_dir / 'lammps' / 'slide.in').write_text(DECK, encoding='utf-8')
    for name in ('system', 'slide'):
        (run_dir / 'lammps' / f'run_{name}.log').write_text(LOG, encoding='utf-8')
    for load in loads:
        _afm_snapshots(run_dir, load, eject=load in eject)
    for load in quarantine:
        for path in (run_dir / 'data').glob(f'final_f{load:g}N*'):
            path.unlink()
    return run_dir


def _statuses(root):
    """Every point the DataReader sees under `root`, with its status."""
    out = {}
    for point in DataReader(str(root)).points:
        rel = point.locator.run_dir.resolve().relative_to(root.resolve()).as_posix()
        out[(rel, point.load_kind, point.load, point.angle, point.speed)] = point.status
    return out


def _calcjobs(label):
    group = orm.Group.collection.get(label=f'friction2d/{label}')
    return [n for n in group.nodes if isinstance(n, orm.CalcJobNode)]


def _archive_db(path, tmp_path):
    db = tmp_path / f'{path.name}.sqlite3'
    with zipfile.ZipFile(path) as archive:
        db.write_bytes(archive.read('db.sqlite3'))
    return sqlite3.connect(db)


def _archive_calcjob_uuids(path, tmp_path):
    con = _archive_db(path, tmp_path)
    try:
        return {u for (u,) in con.execute(
            'SELECT uuid FROM db_dbnode WHERE node_type = ?', (CALCJOB_TYPE,))}
    finally:
        con.close()


def _archive_text(path):
    """Every entry of the archive, decompressed, as one text."""
    with zipfile.ZipFile(path) as archive:
        return '\n'.join(archive.read(name).decode('utf-8', errors='replace')
                         for name in archive.namelist())


# --- export_results: round trip ------------------------------------------------

def test_round_trip_gives_the_source_statuses(aiida_profile_clean, no_profile_switch,
                                              tmp_path):
    root = tmp_path / 'sim'
    run_dir = _run(root, eject=(10.0,), quarantine=(50.0,), loads=(10.0, 30.0, 50.0, 70.0))
    (run_dir / 'results' / 'halted.txt').write_text(
        '# HALT stage=slide load=70 angle=0 speed=2 T=9000 step=3000\n', encoding='utf-8')
    _write_ruling(root, _pid(root, 50.0), 'pass')
    report = import_root(root, label='camp')
    assert report.ok(), report.render()
    assert report.rulings.applied == 1           # import_root applies the rulings
    source = _statuses(root)
    assert set(source.values()) >= {'complete', 'failed_integrity', 'halted'}

    written = export_results('camp', tmp_path / 'out')

    assert written == sorted(written)
    assert _statuses(tmp_path / 'out') == source


def test_export_results_writes_the_run_layout(aiida_profile_clean, no_profile_switch,
                                              tmp_path):
    root = tmp_path / 'sim'
    run_dir = _run(root, loads=(10.0, 30.0))
    rel = run_dir.relative_to(root).as_posix()
    assert import_root(root, label='camp').ok()

    export_results('camp', tmp_path / 'out')

    out = tmp_path / 'out' / rel
    assert (out / 'provenance.json').is_file()
    # AFM: provenance/ is shared by every layer, one level above L2.
    assert (out.parent / 'provenance' / 'config.json').is_file()
    assert {p.name for p in (out / 'lammps').glob('*.in')} == {'system.in', 'slide.in'}
    assert (out / 'lammps' / 'run_slide.log').read_text(encoding='utf-8') == LOG
    assert {p.name for p in (out / 'results').iterdir()} == {
        p.name for p in (run_dir / 'results').iterdir()}
    assert {p.name for p in (out / 'data').iterdir()} == {
        p.name for p in (run_dir / 'data').iterdir()}
    for name in ('load_10N.data', 'final_f30N_a0.data'):
        assert (out / 'data' / name).read_bytes() == (run_dir / 'data' / name).read_bytes()


def test_export_results_of_an_unknown_set_is_a_lookup_error(aiida_profile_clean,
                                                            no_profile_switch, tmp_path):
    with pytest.raises(LookupError, match='nope'):
        export_results('nope', tmp_path / 'out')


def test_rebuild_set_hands_each_config_once_to_the_router(aiida_profile_clean,
                                                          no_profile_switch, tmp_path,
                                                          monkeypatch):
    root = tmp_path / 'sim'
    _run(root, 'h-MoS2', loads=(10.0,))
    _run(root, 'h-WS2', loads=(10.0,))
    assert import_root(root, label='camp').ok()
    calls = []

    class _Result:  # pylint: disable=too-few-public-methods
        created, failures = [], []

    def _fake(config_file, model, **kwargs):
        calls.append((json.loads(config_file.read_text(encoding='utf-8')), model, kwargs))
        return _Result()

    monkeypatch.setattr(export_mod, 'generate_package', _fake)
    created, dest = rebuild_set('camp', tmp_path / 'rebuilt')

    assert created == [] and dest == tmp_path / 'rebuilt'
    assert len(calls) == 2
    assert {model for _c, model, _k in calls} == {'afm'}
    for _config, _model, kwargs in calls:
        assert kwargs['use_aiida'] is False and kwargs['simulation_root_name'] == '.'
        assert kwargs['allow_generate_amorphous'] is False
        assert kwargs['output_root'] == tmp_path / 'rebuilt'


# --- export_archive: gate, contents --------------------------------------------

def test_an_unchecked_point_blocks_the_archive_unless_forced(aiida_profile_clean,
                                                             no_profile_switch, tmp_path):
    root = tmp_path / 'sim'
    _run(root, loads=(10.0, 30.0))
    report = import_root(root, label='camp')
    assert report.ok(), report.render()
    point = next(orm.load_node(u) for u in report.points
                 if orm.load_node(u).base.attributes.get('load') == 30.0)
    from src.aiida.steps import latest_verdict
    latest_verdict(point).base.extras.set('superseded', True)

    with pytest.raises(UncheckedPointsError) as caught:
        export_archive('camp', tmp_path / 'camp.aiida')
    (line,) = caught.value.points
    assert point.base.attributes.get('point_uuid') in line and 'force30_a0_s2' in line
    assert not (tmp_path / 'camp.aiida').exists()

    path = export_archive('camp', tmp_path / 'camp.aiida', force=True)
    assert path.is_file()


def test_archive_carries_verdicts_summaries_and_the_schema(aiida_profile_clean,
                                                          no_profile_switch, tmp_path):
    root = tmp_path / 'sim'
    _run(root, loads=(10.0, 30.0))
    assert import_root(root, label='camp').ok()
    assert postprocess_group('camp', options=OPTIONS) == 2
    path = export_archive('camp', tmp_path / 'camp.aiida')
    readme = tmp_path / 'camp.aiida.README.md'
    assert readme.is_file()
    text = readme.read_text(encoding='utf-8')
    for needle in ('points__', 'snapshots__', 'peratom__', 'blocks__',
                   "parameters['block_names']", 'email', 'RemoteData', 'MB'):
        assert needle in text, needle

    aiida_profile_clean.reset_storage()
    assert import_archive(path) > 0

    def outputs_of(label):
        qb = orm.QueryBuilder().append(orm.CalcFunctionNode, tag='f',
                                       filters={'attributes.process_label': label})
        return qb.append(orm.Dict, with_incoming='f').all(flat=True)

    assert len(outputs_of('check_integrity')) == 2
    assert len(outputs_of('summarise')) == 2
    (calc, *_rest) = _calcjobs('camp')
    assert calc.inputs.schema['fields']['derived.drive_k']['unit'] == 'eV/angstrom^2'
    assert verify().ok, verify().render()


def test_peratom_nodes_are_opt_in(aiida_profile_clean, no_profile_switch, tmp_path):
    root = tmp_path / 'sim'
    _run(root, loads=(10.0, 30.0), tiers=('window', 'peratom'))
    assert import_root(root, label='camp').ok()

    def peratom_count():
        return orm.QueryBuilder().append(orm.CalcJobNode, tag='c').append(
            orm.FolderData, with_incoming='c',
            edge_filters={'label': {'like': 'peratom__%'}}).count()

    assert peratom_count() == 2
    lean = export_archive('camp', tmp_path / 'lean.aiida', force=True)
    full = export_archive('camp', tmp_path / 'full.aiida', force=True, include_peratom=True)
    assert peratom_count() == 2                       # the local store keeps them
    assert lean.stat().st_size < full.stat().st_size

    aiida_profile_clean.reset_storage()
    import_archive(lean)
    assert peratom_count() == 0
    assert all(p.base.attributes.get('has_peratom') for c in _calcjobs('camp')
               if 'points' in c.outputs for p in c.outputs.points.values())
    aiida_profile_clean.reset_storage()
    import_archive(full)
    assert peratom_count() == 2


# --- closure [I6] ---------------------------------------------------------------

def test_a_cross_campaign_dataset_does_not_pull_the_other_campaign_in(
        aiida_profile_clean, no_profile_switch, tmp_path, monkeypatch):
    _run(tmp_path / 'a', 'h-MoS2', loads=(10.0,))
    _run(tmp_path / 'b', 'h-WS2', loads=(10.0,))
    assert import_root(tmp_path / 'a', label='A').ok()
    assert import_root(tmp_path / 'b', label='B').ok()
    postprocess_group('A', options=OPTIONS)
    postprocess_group('B', options=OPTIONS)
    table = build_dataset('both', ['A', 'B'], options=OPTIONS)
    assert {row['set'] for row in table['rows']} == {'A', 'B'}
    b_calcs = {c.uuid for c in _calcjobs('B')}

    path = export_archive('A', tmp_path / 'a.aiida')

    in_archive = _archive_calcjob_uuids(path, tmp_path)
    assert in_archive == {c.uuid for c in _calcjobs('A')}
    assert not in_archive & b_calcs
    con = _archive_db(path, tmp_path)
    groups = {label for (label,) in con.execute('SELECT label FROM db_dbgroup')}
    con.close()
    assert groups == {'friction2d/A'}

    real = export_mod._start_pks
    foreign = _calcjobs('B')[0]
    monkeypatch.setattr(export_mod, '_start_pks',
                        lambda group, **kw: real(group, **kw) | {foreign.pk})
    with pytest.raises(ExportClosureError) as caught:
        export_archive('A', tmp_path / 'leak.aiida')
    assert any(foreign.uuid in line for line in caught.value.calcjobs)
    assert not (tmp_path / 'leak.aiida').exists()


def test_a_dataset_is_its_own_archive(aiida_profile_clean, no_profile_switch, tmp_path):
    _run(tmp_path / 'a', 'h-MoS2', loads=(10.0,))
    _run(tmp_path / 'b', 'h-WS2', loads=(10.0,))
    import_root(tmp_path / 'a', label='A')
    import_root(tmp_path / 'b', label='B')
    postprocess_group('A', options=OPTIONS)
    postprocess_group('B', options=OPTIONS)
    table_uuid = build_dataset('both', ['A', 'B'], options=OPTIONS).uuid
    calcs = {c.uuid for label in ('A', 'B') for c in _calcjobs(label)
             if 'points' in c.outputs}

    path = export_dataset('both', tmp_path / 'both.aiida')

    assert calcs <= _archive_calcjob_uuids(path, tmp_path)
    aiida_profile_clean.reset_storage()
    import_archive(path)
    group = orm.Group.collection.get(label='friction2d/datasets/both')
    assert table_uuid in {n.uuid for n in group.nodes}
    with pytest.raises(LookupError, match='none'):
        export_dataset('none', tmp_path / 'none.aiida')


# --- Review Focus 3 [C2]: two profiles share hashed nodes -----------------------

def test_a_campaign_imported_into_another_profile_joins_its_shared_blocks(
        aiida_profile_clean, no_profile_switch, tmp_path):
    from src.aiida.store import machine_computer
    from tests.aiida.test_query import A_UUID, B_UUID, _campaign

    a = _campaign(tmp_path / 'A', A_UUID, machine_computer('localhost', {}),
                  loads=(10.0,), timestep=0.001)
    (a_calc,) = [c for c in _calcjobs('A') if 'points' in c.outputs]
    a_hashes = {name: node.base.extras.get(BLOCK_HASH_EXTRA)
                for name, node in a_calc.inputs.blocks.items()}
    a_point = a[10.0].uuid
    path = export_archive('A', tmp_path / 'a.aiida', force=True)

    aiida_profile_clean.reset_storage()                          # profile B
    b = _campaign(tmp_path / 'B', B_UUID, machine_computer('localhost', {}),
                  loads=(10.0,), timestep=0.0005)
    (b_calc,) = [c for c in _calcjobs('B') if 'points' in c.outputs]
    b_hashes = {name: node.base.extras.get(BLOCK_HASH_EXTRA)
                for name, node in b_calc.inputs.blocks.items()}
    shared = {name for name in a_hashes if a_hashes[name] == b_hashes.get(name)}
    assert {'thermostat', 'potential_sheet_0'} <= shared
    assert a_hashes['integration'] != b_hashes['integration']

    import_archive(path)

    for name in a_hashes:
        same = orm.QueryBuilder().append(
            orm.Dict, filters={f'extras.{BLOCK_HASH_EXTRA}': a_hashes[name]}).count()
        assert same == 1, name
    assert [p.uuid for p in related(b[10.0], 'integration')] == [a_point]


# --- sidecars --------------------------------------------------------------------

def test_sidecars_round_trip_through_compare(aiida_profile_clean, no_profile_switch,
                                             tmp_path):
    root = tmp_path / 'sim'
    _run(root, 'h-MoS2', loads=(10.0,))
    _run(root, 'h-WS2', loads=(10.0,))
    assert import_root(root, label='camp').ok()
    path = export_archive('camp', tmp_path / 'camp.aiida')
    params = tmp_path / 'camp.aiida.params'

    for calc in _calcjobs('camp'):
        p = calc.inputs.parameters
        sidecar = params / p['rel_path'] / f"{p['deck_name']}.json"
        assert sidecar.is_file(), sidecar
        one = load_blocks_from_sidecar(sidecar)
        stored = load_blocks(str(calc.pk))
        assert one.blocks == stored.blocks
        assert one.derived == stored.derived
        assert one.units == stored.units
    assert list(params.glob('schema-*.json'))
    whole = load_blocks_from_sidecar(params)
    assert whole.blocks == load_blocks('camp').blocks
    assert path.is_file()


# --- anonymisation (ruling P2-R11) ------------------------------------------------

SECRETS = {
    'hpc_home': '/home/secretuser/hpc',
    'scratch_dir': '/scratch/secretuser',
    'log_dir': '/logs/secretuser',
    'conda_env': '/envs/secretuser/lammps',
    'hpc_host': 'login.secret-cluster.example',
}
JOB_EXECUTABLE = '/apps/secret/bin/lmp_mpi'
JOB_CD = '/rds/secretuser/runs'


def _hpc_tree(root):
    run_dir = _run(root, loads=(10.0,))
    prov = run_dir.parent / 'provenance'
    hpc = {'scheduler_type': 'pbs', **SECRETS}
    (prov / 'settings.yaml').write_text(yaml.safe_dump({'hpc': hpc}), encoding='utf-8')
    config = json.loads((prov / 'config.json').read_text(encoding='utf-8'))
    config['settings']['hpc'] = dict(hpc)
    (prov / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    (root / 'hpc').mkdir()
    (root / 'hpc' / 'job.pbs').write_text(
        f'#!/bin/bash\ncd {JOB_CD}\nmpirun -np 4 {JOB_EXECUTABLE} -in lammps/slide.in\n',
        encoding='utf-8')
    return run_dir


def test_archive_anonymises_hpc_details_by_default(aiida_profile_clean, no_profile_switch,
                                                   tmp_path):
    root = tmp_path / 'sim'
    _hpc_tree(root)
    assert import_root(root, label='camp', machine='mycluster').ok()
    calc = next(c for c in _calcjobs('camp') if 'code' in c.inputs)
    assert calc.inputs.code.filepath_executable.as_posix() == JOB_EXECUTABLE
    assert calc.inputs.parameters['hpc_run_path'].startswith(JOB_CD)
    assert calc.computer.hostname == SECRETS['hpc_host']

    path = export_archive('camp', tmp_path / 'camp.aiida')

    text = _archive_text(path)
    for secret in (*SECRETS.values(), JOB_EXECUTABLE, JOB_CD):
        assert secret not in text, secret
    # The local store keeps everything.
    assert calc.inputs.code.filepath_executable.as_posix() == JOB_EXECUTABLE
    assert calc.computer.hostname == SECRETS['hpc_host']
    # A re-import into the source profile deduplicates onto the full nodes.
    assert import_archive(path) == 0

    kept = export_archive('camp', tmp_path / 'kept.aiida', keep_hpc_paths=True)
    text = _archive_text(kept)
    for secret in (*SECRETS.values(), JOB_EXECUTABLE, JOB_CD):
        assert secret in text, secret

    aiida_profile_clean.reset_storage()
    import_archive(path)
    calc = next(c for c in _calcjobs('camp') if 'code' in c.inputs)
    assert calc.inputs.code.filepath_executable.as_posix() == 'lmp_mpi'
    assert 'hpc_run_path' not in calc.inputs.parameters.get_dict()
    assert calc.computer.hostname == 'mycluster'
    settings = yaml.safe_load(calc.inputs.provenance.base.repository.get_object_content(
        'provenance/settings.yaml', mode='r'))
    assert settings['hpc'] == {'scheduler_type': 'pbs', 'hpc_host': 'mycluster'}
    assert calc.inputs.provenance.base.extras.get('friction2d_anonymised') is True
    assert verify().ok, verify().render()
