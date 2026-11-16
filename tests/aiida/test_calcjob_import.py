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
    LammpsFrictionCalcJob, LammpsFrictionParser, deck_inputs, import_deck,
    import_run, point_label,
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
             conflict=False):
    """An AFM run with `system.in` + `slide.in`, `loads` configured, results
    for `written` only, `load_*.data` from the system deck and a shared build."""
    run_dir = make_family_run(root, 'afm', loads=written, final_data=False,
                              config=slide_config('afm', loads=loads, steps=50_000,
                                                  freq=1000))
    lammps = run_dir / 'lammps'
    (lammps / 'slide.in').write_text('# slide\n', encoding='utf-8')
    if logs:
        (lammps / 'run_system.log').write_text(SYSTEM_LOG, encoding='utf-8')
        (lammps / 'run_slide.log').write_text(SLIDE_LOG, encoding='utf-8')
    data = run_dir / 'data'
    data.mkdir(exist_ok=True)
    for load in loads:
        (data / f'load_{load:g}N.data').write_text(f'LAMMPS data {load}\n',
                                                   encoding='utf-8')
        (data / f'hold_{load:g}N.txt').write_text('# hold\n', encoding='utf-8')
    for load in written:
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
    names = set(p10.base.repository.list_object_names('results'))
    assert names == {'friction_f10_a0_s2_layer2.txt'}
    assert p10.base.repository.list_object_names('data') == ['final_f10N_a0.data']


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


# --- configured but absent ------------------------------------------------------

def test_configured_load_without_results_is_an_empty_no_output_point(computer, tmp_path):
    loc = _afm_run(tmp_path / 'sim', loads=(10.0, 30.0), written=(10.0,))
    _system, slide = import_run(loc, computer=computer)
    points = _points(slide)
    missing = points['force30_a0_s2']
    assert missing.base.attributes.get('status') == 'no_output'
    assert missing.base.attributes.get('load') == 30.0
    assert missing.base.repository.list_object_names() == []
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
    assert info['log_missing'] is False
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
    loc = _afm_run(root)
    deck = discover_decks(loc)[0]
    first = deck_inputs(loc, deck, computer=computer)
    second = deck_inputs(loc, deck, computer=computer)
    assert first['provenance'].uuid == second['provenance'].uuid
    assert first['deck'].uuid == second['deck'].uuid
    assert {k: v.uuid for k, v in first['blocks'].items()} == \
        {k: v.uuid for k, v in second['blocks'].items()}
    assert all('__' not in key for key in first['blocks'])
    prov = first['provenance'].base.repository
    assert 'provenance.json' in prov.list_object_names()
    assert 'config.json' in prov.list_object_names('provenance')
    params = first['parameters'].get_dict()
    assert params['deck_name'] == 'system'
    assert params['rel_path'] == loc.tokens.rel_path
    assert params['family'] == 'afm' and params['n_layers'] == 2
    assert params['point_filter'] == {}


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
