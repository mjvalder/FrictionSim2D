"""The import-only CalcJob: one per deck of a finished run (spec §5.1).

Launched with `engine.run` and a `remote_folder`, so aiida-core skips upload
and submit and only retrieves (real `LocalTransport`) and parses. Replaces
`test_calcjob_parser.py`, `test_calcjob_afm_staging.py` and
`test_aiida_calcjob.py`, whose subject -- staging and running LAMMPS through
AiiDA -- is gone.
"""
import json

import pytest
import yaml

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.calcjob import (  # noqa: E402
    LammpsFrictionCalcJob, LammpsFrictionParser, MissingRunUUID, deck_inputs,
    import_deck, import_run, point_label,
)
from src.aiida.decks import discover_decks  # noqa: E402
from src.aiida.retention import retrieve_patterns  # noqa: E402
from src.aiida.store import machine_computer  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.results_tree import (  # noqa: E402
    make_family_run, slide_config, write_final_data,
)

VERSION = '29 Aug 2024 - Update 1'

SYSTEM_LOG = f"""LAMMPS ({VERSION})
Loop time of 12.5 on 4 procs for 1000 steps with 600 atoms
Total wall time: 0:01:05
"""

SLIDE_LOG = f"""LAMMPS ({VERSION})
Loop time of 100.0 on 8 procs for 50000 steps with 600 atoms
Total wall time: 1:00:10
"""

SECRET = 'hunter2-central-db-password'


def _afm_run(root, *, loads=(10.0, 30.0), written=(10.0, 30.0), logs=True,
             conflict=False, snapshots=None, tiers=('window',)):
    """An AFM run with `system.in` + `slide.in`, `loads` configured, results
    for `written` only, `load_*.data` from the system deck and a shared build."""
    run_dir = make_family_run(root, 'afm', loads=written, final_data=False,
                              tiers=tiers,
                              config=slide_config('afm', loads=loads, steps=50_000,
                                                  freq=1000))
    lammps = run_dir / 'lammps'
    rel = run_dir.relative_to(root).as_posix()
    include = f'include         {rel}/lammps/system.in.settings\n'
    (lammps / 'system.in').write_text('# system\n' + include, encoding='utf-8')
    (lammps / 'slide.in').write_text('# slide\n' + include, encoding='utf-8')
    (lammps / 'system.in.settings').write_text('mass 1 95.95\n', encoding='utf-8')
    if logs:
        (lammps / 'run_system.log').write_text(SYSTEM_LOG, encoding='utf-8')
        (lammps / 'run_slide.log').write_text(SLIDE_LOG, encoding='utf-8')
    data = run_dir / 'data'
    data.mkdir(exist_ok=True)
    for load in loads:
        (data / f'load_{load:g}N.data').write_text(f'LAMMPS data {load}\n',
                                                   encoding='utf-8')
        (data / f'hold_{load:g}N.txt').write_text('# hold\n', encoding='utf-8')
    for load in (written if snapshots is None else snapshots):
        write_final_data(run_dir, 'afm', load=load)
    build = run_dir.parent / 'build'
    build.mkdir(exist_ok=True)
    (build / 'sheet.lmp').write_text('LAMMPS sheet\n', encoding='utf-8')
    if conflict:
        results = run_dir / 'results'
        original = sorted(results.glob('friction_f10_*'))[0]
        twin = results / original.name.replace(
            '.txt', '.sync-conflict-20261014-173911-U5EC3BU.txt')
        twin.write_text('# corrupted twin\n', encoding='utf-8')
    (loc,) = [r for r in discover_runs(root) if r.run_dir == run_dir]
    return loc


@pytest.fixture
def computer(aiida_profile_clean):
    return machine_computer('localhost')


def _points(node):
    return dict(node.outputs.points.items())


# --- the declared surface -------------------------------------------------------

def test_spec_is_import_only(aiida_profile_clean):
    spec = LammpsFrictionCalcJob.spec()
    for port in ('remote_folder', 'deck', 'provenance', 'blocks', 'potentials',
                 'structures', 'parameters'):
        assert port in spec.inputs, port
    for port in ('points', 'structures', 'run_info', 'log'):
        assert port in spec.outputs, port
    assert {c.status for c in spec.exit_codes.values()} >= {320}
    assert not {300, 310} & {c.status for c in spec.exit_codes.values()}
    assert 'simulation_dir' not in spec.inputs
    assert not spec.inputs['code'].required


def test_parser_is_the_registered_entry_point(aiida_profile_clean):
    from aiida.plugins import CalculationFactory, ParserFactory
    assert ParserFactory('friction2d.lammps') is LammpsFrictionParser
    assert CalculationFactory('friction2d.lammps') is LammpsFrictionCalcJob


# --- AFM: system + slide, structure handoff -------------------------------------

def test_afm_system_and_slide_hand_off_structures(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    nodes = import_run(loc, computer=computer)

    assert [n.inputs.parameters['deck_name'] for n in nodes] == ['system', 'slide']
    system, slide = nodes
    assert system.is_finished_ok and slide.is_finished_ok
    assert system.is_imported and slide.is_imported

    produced = {n.uuid for n in system.outputs.structures.values()}
    assert len(produced) == 2                      # load_10N, load_30N
    consumed = {n.uuid for n in slide.inputs.structures.values()}
    assert produced <= consumed                    # the SAME nodes, by uuid
    # the system deck reads the shared build; the slide deck does not
    assert any(n.filename == 'sheet.lmp' for n in system.inputs.structures.values())
    assert not any(n.filename == 'sheet.lmp' for n in slide.inputs.structures.values())

    assert 'points' not in system.outputs
    points = _points(slide)
    assert set(points) == {'force10_a0_s2', 'force30_a0_s2'}
    p10 = points['force10_a0_s2']
    attrs = p10.base.attributes.all
    for key in ('family', 'load_kind', 'load', 'angle', 'speed', 'layers',
                'status', 'content_sha256', 'rel_path', 'deck_name'):
        assert key in attrs, key
    assert (attrs['family'], attrs['load_kind'], attrs['load'], attrs['layers']) == \
        ('afm', 'force', 10.0, 2)
    assert attrs['status'] == 'complete'
    assert attrs['rel_path'] == loc.tokens.rel_path
    assert attrs['deck_name'] == 'slide'
    assert p10.base.repository.list_object_names() == ['results']
    names = set(p10.base.repository.list_object_names('results'))
    assert names == {'friction_f10_a0_s2_layer2.txt'}
    assert attrs['has_snapshot'] is True and attrs['has_peratom'] is False
    snapshot = slide.outputs.snapshots['force10_a0_s2']
    assert snapshot.filename == 'final_f10N_a0.data'
    assert snapshot.get_content() == (loc.data_dir / 'final_f10N_a0.data').read_text()
    assert 'peratom' not in slide.outputs

    # every deck that includes system.in.settings carries it, as ONE node
    settings_nodes = {n.inputs.includes['system_in_settings'].uuid for n in nodes}
    assert len(settings_nodes) == 1
    assert nodes[0].inputs.includes['system_in_settings'].get_content() == 'mass 1 95.95\n'


def test_retrieve_lists_come_from_the_retention_rule(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    system, slide = import_run(loc, computer=computer)
    patterns = set(retrieve_patterns())
    for node in (system, slide):
        temporary = {tuple(e) for e in node.base.attributes.get('retrieve_temporary_list')}
        assert temporary and temporary <= patterns
    slide_tmp = {e[0] for e in slide.base.attributes.get('retrieve_temporary_list')}
    assert 'results/*' in slide_tmp and 'data/load_*.data' not in slide_tmp
    # aiida-core appends its two scheduler-output names as plain strings
    assert [e[0] for e in slide.base.attributes.get('retrieve_list')
            if isinstance(e, (list, tuple))] == ['lammps/run_slide.log']


def test_array_mode_decks_split_points_by_load(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    (loc.lammps_dir / 'slide.in').unlink()
    (loc.lammps_dir / 'slide_f10N.in').write_text('# f10\n', encoding='utf-8')
    (loc.lammps_dir / 'slide_f30N.in').write_text('# f30\n', encoding='utf-8')
    nodes = import_run(loc, computer=computer)
    by_deck = {n.inputs.parameters['deck_name']: n for n in nodes}
    assert set(_points(by_deck['slide_f10N'])) == {'force10_a0_s2'}
    produced = {n.filename: n.uuid for n in by_deck['system'].outputs.structures.values()}
    for deck, name in (('slide_f10N', 'load_10N.data'), ('slide_f30N', 'load_30N.data')):
        given = {n.filename: n.uuid for n in by_deck[deck].inputs.structures.values()}
        assert given == {name: produced[name]}, deck
    assert set(_points(by_deck['slide_f30N'])) == {'force30_a0_s2'}


def test_import_logs_no_scheduler_warnings(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    for node in import_run(loc, computer=computer):
        messages = [log.message for log in orm.Log.collection.get_logs_for(node)]
        assert not [m for m in messages if 'scheduler' in m], messages


def test_sheet_run_without_system_deck_reads_build_in_its_slide(computer, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'sheetonsheet', loads=(0.5,),
                              config=slide_config('sheetonsheet', loads=(0.5,)))
    (run_dir / 'lammps' / 'system.in').unlink()
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    (run_dir / 'build').mkdir()
    (run_dir / 'build' / 'bilayer.lmp').write_text('LAMMPS\n', encoding='utf-8')
    (loc,) = discover_runs(root)
    (slide,) = import_run(loc, computer=computer)
    assert slide.is_finished_ok
    assert [n.filename for n in slide.inputs.structures.values()] == ['bilayer.lmp']
    assert set(_points(slide)) == {'pressure0p5_a0_s2'}


def test_pes_refine_deck_holds_the_refine_grid(computer, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'pes_sheet')
    (run_dir / 'lammps' / 'system.in').unlink()
    for stem in ('slide', 'slide_refine'):
        (run_dir / 'lammps' / f'{stem}.in').write_text('# pes\n', encoding='utf-8')
    refine = run_dir / 'results' / 'pes_scan_refine.csv'
    refine.write_text((run_dir / 'results' / 'pes_scan.csv').read_text(), encoding='utf-8')
    (loc,) = discover_runs(root)
    base, fine = import_run(loc, computer=computer)
    assert base.outputs.points['none0_a0_s0'].base.repository.list_object_names(
        'results') == ['pes_scan.csv']
    assert fine.outputs.points['none0_a0_s0'].base.repository.list_object_names(
        'results') == ['pes_scan_refine.csv']


def test_peratom_tier_is_its_own_node(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', tiers=('window', 'peratom'))
    _system, slide = import_run(loc, computer=computer)
    p10 = _points(slide)['force10_a0_s2']
    assert p10.base.repository.list_object_names('results') == \
        ['friction_f10_a0_s2_layer2.txt']
    assert p10.base.attributes.get('has_peratom') is True
    peratom = slide.outputs.peratom['force10_a0_s2']
    assert peratom.base.repository.list_object_names('results') == \
        ['peratom_f10_a0_s2_layer2.txt']


def test_a_point_with_results_but_no_snapshot(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', snapshots=(30.0,))
    _system, slide = import_run(loc, computer=computer)
    p10 = _points(slide)['force10_a0_s2']
    assert p10.base.attributes.get('status') == 'complete'
    assert p10.base.attributes.get('has_snapshot') is False
    assert set(slide.outputs.snapshots) == {'force30_a0_s2'}


def test_configured_indent_load_without_results_gets_a_node(computer, tmp_path):
    root = tmp_path / 'sim'
    config = {'general': {'temp': 300, 'force': [10.0, 30.0], 'scan_angle': 0.0,
                          'scan_speed': [2.0]}}
    run_dir = make_family_run(root, 'indent', loads=(10.0,), config=config)
    (run_dir / 'results' / 'indent_f10N_l2.csv').write_text('z_tip,fz_tip\n0,1\n',
                                                            encoding='utf-8')
    (run_dir / 'lammps' / 'indent.in').write_text('# indent\n', encoding='utf-8')
    (loc,) = discover_runs(root)
    deck = [d for d in discover_decks(loc) if d.name == 'indent'][0]
    _result, node = import_deck(loc, deck, computer=computer)
    points = _points(node)
    assert set(points) == {'force10_a0_s2', 'force30_a0_s2'}
    assert points['force30_a0_s2'].base.attributes.get('status') == 'no_output'
    assert points['force10_a0_s2'].base.repository.list_object_names('results') == \
        ['friction_f10_a0_s2_layer2.txt', 'indent_f10N_l2.csv']


def test_every_output_file_goes_through_redaction(computer, tmp_path, monkeypatch):
    import src.aiida.store as store_mod
    staged = []
    real = store_mod.stage_file

    def spy(repository, source, name):
        staged.append(name)
        return real(repository, source, name)

    monkeypatch.setattr(store_mod, 'stage_file', spy)
    loc = _afm_run(tmp_path / 'sim')
    system, slide = import_run(loc, computer=computer)
    outputs = [n for calc in (system, slide)
               for n in calc.base.links.get_outgoing().all_nodes()
               if isinstance(n, (orm.SinglefileData, orm.FolderData))
               and n.uuid != calc.outputs.retrieved.uuid]
    n_files = sum(len(list(n.base.repository.walk())[0][2]) if isinstance(n, orm.SinglefileData)
                  else sum(len(f) for _r, _d, f in n.base.repository.walk())
                  for n in outputs)
    assert n_files and len([x for x in staged
                            if x.startswith(('results/', 'data/', 'lammps/run_'))]) >= n_files


def test_log_without_version_keeps_the_recorded_version(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    (loc.lammps_dir / 'run_slide.log').write_bytes(
        b'Loop time of 1.0 on 2 procs for 10 steps\n\xff\xfe bad bytes\n')
    record = json.loads(loc.provenance_file.read_text(encoding='utf-8'))
    record['lammps_version'] = '2 Aug 2023'
    loc.provenance_file.write_text(json.dumps(record), encoding='utf-8')
    _system, slide = import_run(loc, computer=computer)
    info = slide.outputs.run_info.get_dict()
    assert info['lammps_version'] == '2 Aug 2023'
    assert info['log_has_version'] is False and info['log_missing'] is False
    assert info['n_procs'] == 2


# --- configured but absent ------------------------------------------------------

def test_configured_load_without_results_is_an_empty_no_output_point(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', loads=(10.0, 30.0), written=(10.0,))
    _system, slide = import_run(loc, computer=computer)
    points = _points(slide)
    missing = points['force30_a0_s2']
    assert missing.base.attributes.get('status') == 'no_output'
    assert missing.base.attributes.get('load') == 30.0
    assert missing.base.repository.list_object_names() == []
    assert missing.base.attributes.get('has_snapshot') is False
    assert set(slide.outputs.snapshots) == {'force10_a0_s2'}
    assert points['force10_a0_s2'].base.attributes.get('status') == 'complete'


def test_truncated_point_is_classified(computer, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'afm', loads=(10.0,), n=10,
                              config=slide_config('afm', loads=(10.0,), steps=50_000,
                                                  freq=1000))
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    (loc,) = discover_runs(root)
    deck = [d for d in discover_decks(loc) if d.name == 'slide'][0]
    _result, node = import_deck(loc, deck, computer=computer)
    point = node.outputs.points['force10_a0_s2']
    assert point.base.attributes.get('status') == 'truncated'


# --- run_info / log ------------------------------------------------------------

def test_run_info_reads_the_log(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    system, slide = import_run(loc, computer=computer)
    info = slide.outputs.run_info.get_dict()
    assert info['lammps_version'] == VERSION
    assert info['wall_time_s'] == 3610.0
    assert info['n_procs'] == 8
    assert info['log_missing'] is False and info['log_has_version'] is True
    assert system.outputs.run_info['n_procs'] == 4
    assert slide.outputs.log.get_content() == SLIDE_LOG


def test_no_log_sets_log_missing(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', logs=False)
    system, slide = import_run(loc, computer=computer)
    for node in (system, slide):
        assert node.is_finished_ok
        assert node.outputs.run_info['log_missing'] is True
        assert node.outputs.run_info['lammps_version'] is None
        assert 'log' not in node.outputs


# --- sync-conflict copies -------------------------------------------------------

def test_sync_conflict_copy_never_replaces_the_original(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', conflict=True)
    _system, slide = import_run(loc, computer=computer)
    p10 = _points(slide)['force10_a0_s2']
    names = p10.base.repository.list_object_names('results')
    assert names == ['friction_f10_a0_s2_layer2.txt']
    original = (loc.results_dir / names[0]).read_text(encoding='utf-8')
    assert p10.base.repository.get_object_content(f'results/{names[0]}') == original


# --- exit code 320 --------------------------------------------------------------

def test_a_deck_with_neither_points_nor_structures_is_320(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    for path in loc.data_dir.glob('load_*.data'):
        path.unlink()
    system = [d for d in discover_decks(loc) if d.name == 'system'][0]
    _result, node = import_deck(loc, system, computer=computer)
    assert node.exit_status == LammpsFrictionCalcJob.exit_codes.ERROR_NO_RESULTS.status


# --- inputs --------------------------------------------------------------------

def test_inputs_are_content_addressed(computer, tmp_path):
    """Two layer counts of one AFM build share the provenance/ dir, build and
    blocks; the provenance FolderData differs only by provenance.json."""
    root = tmp_path / 'sim'
    loc2 = _afm_run(root)
    run1 = make_family_run(root, 'afm', layers=1, loads=(10.0, 30.0),
                           config=slide_config('afm', loads=(10.0, 30.0),
                                               steps=50_000, freq=1000))
    record = json.loads((run1 / 'provenance.json').read_text(encoding='utf-8'))
    record['layer_type_map'] = {'layer_1': [1]}
    (run1 / 'provenance.json').write_text(json.dumps(record), encoding='utf-8')
    (loc1,) = [r for r in discover_runs(root) if r.run_dir == run1]
    assert loc1.provenance_dir == loc2.provenance_dir

    def system_inputs(loc):
        return deck_inputs(loc, discover_decks(loc)[0], computer=computer)

    one, two = system_inputs(loc1), system_inputs(loc2)
    assert one['structures']['build_sheet_lmp'].uuid == two['structures']['build_sheet_lmp'].uuid
    shared = set(one['blocks']) & set(two['blocks'])
    assert shared and all(one['blocks'][k].uuid == two['blocks'][k].uuid
                          for k in shared if not k.startswith('geometry'))
    assert one['provenance'].uuid != two['provenance'].uuid
    p1, p2 = one['provenance'].base.repository, two['provenance'].base.repository
    assert p1.list_object_names('provenance') == p2.list_object_names('provenance')
    for name in p1.list_object_names('provenance'):
        assert p1.get_object_content(f'provenance/{name}') == \
            p2.get_object_content(f'provenance/{name}')
    assert p1.get_object_content('provenance.json') != p2.get_object_content('provenance.json')
    # importing the same run twice reuses every input node
    again = system_inputs(loc2)
    assert again['provenance'].uuid == two['provenance'].uuid
    assert again['deck'].uuid == two['deck'].uuid
    assert all('__' not in key for key in two['blocks'])
    params = two['parameters'].get_dict()
    assert params['deck_name'] == 'system'
    assert params['rel_path'] == loc2.tokens.rel_path
    assert params['family'] == 'afm' and params['n_layers'] == 2
    assert params['point_filter'] == {}
    assert params['run_uuid'] == json.loads(
        loc2.provenance_file.read_text(encoding='utf-8'))['run_uuid']


def test_potentials_sharing_a_stem_both_survive(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    pots = loc.provenance_dir / 'potentials'
    pots.mkdir(exist_ok=True)
    (pots / 'MoS.sw').write_text('sw\n', encoding='utf-8')
    (pots / 'MoS.ILP').write_text('ilp\n', encoding='utf-8')
    inputs = deck_inputs(loc, discover_decks(loc)[0], computer=computer)
    assert {k: v.filename for k, v in inputs['potentials'].items()} == \
        {'MoS_sw': 'MoS.sw', 'MoS_ILP': 'MoS.ILP'}


def test_a_label_collision_raises():
    from src.aiida.calcjob import _unique
    with pytest.raises(ValueError, match='a_b'):
        _unique([('a_b', 1), ('a_b', 2)], 'potentials')


def test_point_label_is_a_valid_link_label():
    from types import SimpleNamespace
    point = SimpleNamespace(load_kind='pressure', load=0.5, angle=-45.0, speed=2.0)
    assert point_label(point) == 'pressure0p5_am45_s2'


# --- credentials (ported from test_secrets_redaction.py) --------------------------

def _all_repository_text(node):
    out = []
    for root, _dirs, files in node.base.repository.walk():
        for name in files:
            out.append(node.base.repository.get_object_content(
                (root / name).as_posix(), mode='rb').decode('utf-8', 'ignore'))
    return '\n'.join(out)


def test_a_password_in_provenance_settings_never_reaches_the_store(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    (loc.provenance_dir / 'settings.yaml').write_text(
        yaml.safe_dump({'thermostat': {'type': 'langevin'},
                        'database': {'host': 'db.example.org', 'password': SECRET}}),
        encoding='utf-8')
    config = json.loads((loc.provenance_dir / 'config.json').read_text(encoding='utf-8'))
    config['settings']['database'] = {'password': SECRET}
    (loc.provenance_dir / 'config.json').write_text(json.dumps(config), encoding='utf-8')

    nodes = import_run(loc, computer=computer)
    assert all(n.is_finished_ok for n in nodes)

    stored = orm.QueryBuilder().append(orm.Node).all(flat=True)
    for node in stored:
        assert SECRET not in json.dumps(node.base.attributes.all, default=str), node
        assert SECRET not in _all_repository_text(node), node
    prov = nodes[0].inputs.provenance.base.repository
    settings = yaml.safe_load(prov.get_object_content('provenance/settings.yaml'))
    assert settings['database']['host'] == 'db.example.org'


# --- run and point identity (Task 5b) --------------------------------------------

RUN_UUID = '3f0c2a7e-1b7d-4c4e-9a51-6d2f0e8b9c11'


def _set_record(loc, **fields):
    record = json.loads(loc.provenance_file.read_text(encoding='utf-8'))
    record.update(fields)
    loc.provenance_file.write_text(json.dumps(record), encoding='utf-8')


def test_point_uuid_derives_from_the_run_uuid(computer, tmp_path):
    from uuid import UUID, uuid5
    loc = _afm_run(tmp_path / 'sim')
    _set_record(loc, run_uuid=RUN_UUID)
    system, slide = import_run(loc, computer=computer)

    p10 = _points(slide)['force10_a0_s2'].base.attributes
    assert p10.get('point_uuid') == str(uuid5(UUID(RUN_UUID), 'force10_a0_s2'))
    assert slide.inputs.parameters['run_uuid'] == RUN_UUID
    assert system.base.extras.get('friction2d_run_uuid') == RUN_UUID
    assert slide.base.extras.get('friction2d_run_uuid') == RUN_UUID


def test_the_same_point_from_a_moved_copy_keeps_its_uuid(computer, tmp_path):
    import shutil
    loc = _afm_run(tmp_path / 'sim')
    _set_record(loc, run_uuid=RUN_UUID)
    shutil.copytree(tmp_path / 'sim', tmp_path / 'elsewhere' / 'sim2')
    (moved,) = discover_runs(tmp_path / 'elsewhere' / 'sim2')
    assert moved.run_dir != loc.run_dir

    first = _points(import_run(loc, computer=computer)[1])['force10_a0_s2']
    second = _points(import_run(moved, computer=computer)[1])['force10_a0_s2']

    assert first.uuid != second.uuid
    assert first.base.attributes.get('point_uuid') == \
        second.base.attributes.get('point_uuid')


def test_point_units_per_family(computer, tmp_path):
    afm = _points(import_run(_afm_run(tmp_path / 'afm'), computer=computer)[1])
    assert afm['force10_a0_s2'].base.attributes.get('units') == \
        {'load': 'nN', 'angle': 'deg', 'speed': 'm/s'}

    root = tmp_path / 'sheet'
    run_dir = make_family_run(root, 'sheetonsheet', loads=(0.5,),
                              config=slide_config('sheetonsheet', loads=(0.5,)))
    (run_dir / 'lammps' / 'system.in').unlink()
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    (sheet,) = discover_runs(root)
    (slide,) = import_run(sheet, computer=computer)
    points = _points(slide)
    assert points['pressure0p5_a0_s2'].base.attributes.get('units')['load'] == 'GPa'


def test_point_carries_the_achieved_load_and_ramp(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    _set_record(loc, load_achieved_by_load={'10': 9.7},
                ramp_by_load={'10': {'converged': False, 'r': 0.3, 'iterations': 40}})
    points = _points(import_run(loc, computer=computer)[1])

    p10, p30 = (points[k].base.attributes for k in ('force10_a0_s2', 'force30_a0_s2'))
    assert p10.get('load_achieved') == 9.7 and p10.get('ramp_converged') is False
    assert p30.get('load_achieved') is None and p30.get('ramp_converged') is None


def test_a_run_without_run_uuid_raises(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    record = json.loads(loc.provenance_file.read_text(encoding='utf-8'))
    del record['run_uuid']
    loc.provenance_file.write_text(json.dumps(record), encoding='utf-8')

    with pytest.raises(MissingRunUUID, match='stamp-uuid') as info:
        deck_inputs(loc, discover_decks(loc)[0], computer=computer)
    assert str(loc.provenance_file) in str(info.value)
    assert isinstance(info.value, ValueError)


def test_a_non_uuid_run_uuid_raises_clearly(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim')
    _set_record(loc, run_uuid='run-42')
    with pytest.raises(MissingRunUUID, match='is not a UUID') as info:
        deck_inputs(loc, discover_decks(loc)[0], computer=computer)
    assert str(loc.provenance_file) in str(info.value)


# --- the store rework (Task 5c) --------------------------------------------------

def _three_deck_afm_run(root):
    """An AFM run with a system deck and one slide deck per load."""
    loc = _afm_run(root)
    lammps = loc.lammps_dir
    slide = (lammps / 'slide.in').read_text(encoding='utf-8')
    (lammps / 'slide.in').unlink()
    for stem in ('slide_f10N', 'slide_f30N'):
        (lammps / f'{stem}.in').write_text(slide, encoding='utf-8')
    (loc,) = [r for r in discover_runs(root) if r.run_dir == loc.run_dir]
    return loc


def test_a_run_is_split_once_and_indexed_once(computer, tmp_path, monkeypatch):
    from src.aiida import blocks, calcjob
    loc = _three_deck_afm_run(tmp_path / 'sim')
    assert len(discover_decks(loc)) == 3

    splits, loads = [], []
    real_split, real_load = blocks.split_blocks, calcjob.StoreIndex.load

    def counting_split(*args, **kwargs):
        splits.append(1)
        return real_split(*args, **kwargs)

    def counting_load():
        loads.append(1)
        return real_load()

    monkeypatch.setattr(blocks, 'split_blocks', counting_split)
    monkeypatch.setattr(calcjob.StoreIndex, 'load', staticmethod(counting_load))
    nodes = import_run(loc, computer=computer)
    assert [n.inputs.parameters['deck_name'] for n in nodes] == \
        ['system', 'slide_f10N', 'slide_f30N']
    assert all(n.is_finished_ok for n in nodes)
    assert splits == [1] and loads == [1]
    # the run's shared inputs are one node each
    for port in ('derived', 'schema', 'provenance'):
        assert len({n.inputs[port].uuid for n in nodes}) == 1, port


def test_every_calcjob_carries_the_schema_and_derived(computer, tmp_path):
    from src.aiida.store import node_uuid, schema_hash
    nodes = import_run(_afm_run(tmp_path / 'sim'), computer=computer)
    for node in nodes:
        schema = node.inputs.schema
        assert schema.get_dict()['fields']['derived.drive_k']['unit'] == 'eV/angstrom^2'
        assert schema.uuid == node_uuid('schema', schema_hash(schema.get_dict()))
        assert 'drive_k' in node.inputs.derived.get_dict()['fields']


def test_block_names_record_the_unambiguous_name(computer, tmp_path):
    from src.aiida.blocks import block_hash
    from src.aiida.store import node_uuid
    (system, _slide) = import_run(_afm_run(tmp_path / 'sim'), computer=computer)
    names = system.inputs.parameters['block_names']
    assert set(names) == set(system.inputs.blocks)
    assert names['material_sheet_0'] == 'material__sheet_0'
    for label, block in system.inputs.blocks.items():
        assert block.get_dict()['kind'] == names[label].split('__')[0]
        assert block.uuid == node_uuid('block', block_hash(block.get_dict()))


def _legacy_spring(loc, *, schema_version):
    from src.aiida.block_schema import LEGACY_AFM_SPRING_FACTOR
    config = json.loads((loc.provenance_dir / 'config.json').read_text(encoding='utf-8'))
    spring = (config.get('general') or {}).get('driving_spring') or 50.0
    _set_record(loc, schema_version=schema_version,
                drive_k=spring / LEGACY_AFM_SPRING_FACTOR)


def test_a_legacy_afm_run_imports(computer, tmp_path):
    """A pre-2.2.0 AFM run's drive_k used / 16.02: check_units accepts it from the record."""
    loc = _afm_run(tmp_path / 'sim')
    _legacy_spring(loc, schema_version='2.0.0')
    nodes = import_run(loc, computer=computer)
    assert all(n.is_finished_ok for n in nodes)


def test_a_current_run_with_the_legacy_factor_is_refused(computer, tmp_path):
    from src.aiida.block_schema import UnitMismatch
    loc = _afm_run(tmp_path / 'sim')
    _legacy_spring(loc, schema_version='2.2.0')
    with pytest.raises(UnitMismatch, match='drive_k'):
        import_run(loc, computer=computer)


def test_reindex_restores_the_run_extra(computer, tmp_path):
    from src.aiida.store import reindex, verify
    nodes = import_run(_afm_run(tmp_path / 'sim'), computer=computer)
    run_uuid = nodes[0].inputs.parameters['run_uuid']
    for node in nodes:
        node.base.extras.clear()
    nodes[0].inputs.deck.base.extras.clear()
    assert reindex() == len(nodes) + 1
    assert all(n.base.extras.get('friction2d_run_uuid') == run_uuid for n in nodes)
    assert nodes[0].inputs.deck.base.extras.get('friction2d_sha256')
    assert verify().ok, verify().render()
