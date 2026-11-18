"""Integrity verdicts at import and the idempotent rulings pass (plan Task 6a).

Every imported point gets one verdict from `steps.check_integrity`, the
ported checker (`integrity.check_files`) run as a calcfunction; the user's
`integrity_rulings.yaml` is applied by `steps.apply_rulings`, one tracked
`apply_ruling` per (verdict, ruling) pair.
"""
import pytest
import yaml

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402
from aiida.common.links import LinkType  # noqa: E402

from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.integration import clear_all_nodes, delete_simulation_set  # noqa: E402
from src.aiida.steps import apply_rulings, recheck_set  # noqa: E402
from src.postprocessing.integrity_rulings import RULINGS_FILE, point_id  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.helpers.results_tree import make_family_run, slide_config  # noqa: E402
from tests.helpers.synthetic_results import write_results  # noqa: E402
from tests.postprocessing.test_integrity import _stack, _write_data  # noqa: E402
from tests.postprocessing.test_read_data_integrity import (  # noqa: E402
    DECK, _afm_snapshots, _afm_tree, _point)

LOG = """LAMMPS (29 Aug 2024 - Update 1)
Loop time of 12.5 on 4 procs for 1000 steps with 600 atoms
Total wall time: 0:01:05
"""


@pytest.fixture
def no_profile_switch(monkeypatch):
    """`aiida_profile_clean` is already loaded; do not load another."""
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _tree(root, *, eject=(), quarantine=(), no_output=()):
    """An AFM run at 10/30/50 N: `eject` fail, `quarantine` lose their end
    snapshot, `no_output` lose their results."""
    run_dir = _afm_tree(root, loads=(10.0, 30.0, 50.0), eject=eject)
    for name in ('system', 'slide'):
        (run_dir / 'lammps' / f'run_{name}.log').write_text(LOG, encoding='utf-8')
    for load in quarantine:
        for path in (run_dir / 'data').glob(f'final_f{load:g}N*'):
            path.unlink()
    for load in no_output:
        for path in (run_dir / 'results').glob(f'friction_f{load:g}_*'):
            path.unlink()
    return run_dir


def _verdicts_of(point):
    """The verdict Dicts `check_integrity` made from `point`."""
    out = []
    for link in point.base.links.get_outgoing(node_class=orm.CalcFunctionNode,
                                              link_type=LinkType.INPUT_CALC).all():
        func = link.node
        if func.process_label == 'check_integrity':
            out += [n.node for n in func.base.links.get_outgoing(
                link_type=LinkType.CREATE).all()]
    return out


def _nodes(label):
    return orm.QueryBuilder().append(
        orm.CalcFunctionNode, filters={'attributes.process_label': label}).all(flat=True)


def _ruled_outputs():
    return [n.outputs.result for n in _nodes('apply_ruling')]


def _write_ruling(root, pid, ruling, note='looked at it', override=False):
    text = (f'- point: "{pid}"\n  ruling: {ruling}\n  note: {note}\n'
            f'  date: 2026-11-16\n')
    if override:
        text += '  override: true\n'
    (root / RULINGS_FILE).write_text(text, encoding='utf-8')


def _pid(root, load):
    return point_id(_point(DataReader(str(root)), load, points=True), root)


# --- verdicts at import -------------------------------------------------------

def test_every_point_gets_exactly_one_verdict(aiida_profile_clean, no_profile_switch,
                                              tmp_path):
    _tree(tmp_path, quarantine=(30.0,), no_output=(50.0,))
    report = import_root(tmp_path, label='camp')
    assert report.ok(), report.render()

    points = [orm.load_node(u) for u in report.points]
    assert len(points) == 3
    statuses = {}
    for point in points:
        (verdict,) = _verdicts_of(point)
        assert isinstance(verdict, orm.Dict)
        assert verdict['point_uuid'] == point.base.attributes.get('point_uuid')
        assert verdict['thresholds_version'] == '1.0.0'
        assert verdict['code_version']
        statuses[point.base.attributes.get('load')] = verdict['status']
    assert statuses == {10.0: 'complete', 30.0: 'quarantined', 50.0: 'no_output'}
    assert report.verdict_counts == {'complete': 1, 'quarantined': 1, 'no_output': 1}
    assert 'verdict quarantined: 1' in report.render()

    no_output = next(p for p in points if p.base.attributes.get('load') == 50.0)
    (func,) = [link.node for link in no_output.base.links.get_outgoing(
        node_class=orm.CalcFunctionNode).all()]
    # No snapshot, but the bond reference the DataReader uses too (n_heat).
    assert set(func.base.links.get_incoming().all_link_labels()) == {
        'point', 'reference', 'context', 'settings'}
    assert func.inputs.context['status'] == 'no_output'


def _live_verdict(point):
    (verdict,) = [v for v in _verdicts_of(point)
                  if not v.base.extras.get('superseded', False)]
    return verdict


def _assert_matches_reader(report, root, uuids=None):
    """Every point's live store verdict is the one `postprocess read` gives."""
    assert report.ok(), report.render()
    reader = DataReader(str(root))
    statuses = {}
    for uuid in (report.points if uuids is None else uuids):
        point = orm.load_node(uuid)
        verdict = _live_verdict(point)
        attrs = point.base.attributes.all
        (match,) = [p for p in reader.points if p.load == attrs['load']
                    and p.tokens.rel_path == attrs['rel_path']]
        expected = match.integrity.as_dict()
        got = verdict.get_dict()
        for key in ('status', 'failed', 'observed', 'reasons', 'quarantine',
                    'not_applicable', 'thresholds_version'):
            assert got[key] == expected[key], key
        assert got['measures'].keys() == expected['measures'].keys()
        for key, value in expected['measures'].items():
            if value is None:
                assert got['measures'][key] is None, key
            else:
                assert got['measures'][key] == pytest.approx(value), key
        statuses[attrs['load']] = got['status']
    return statuses


def test_verdicts_match_the_datareader(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,), quarantine=(30.0,), no_output=(50.0,))
    report = import_root(tmp_path, label='camp')
    statuses = _assert_matches_reader(report, tmp_path)
    assert statuses == {10.0: 'failed_integrity', 30.0: 'quarantined', 50.0: 'no_output'}


def test_chain_after_a_truncated_load_is_not_reached(aiida_profile_clean,
                                                    no_profile_switch, tmp_path):
    """Stage-2 parity: the template ramps each load from the last, so a
    no-output load after a truncated one was never reached."""
    run_dir = _tree(tmp_path, no_output=(50.0,))
    write_results(run_dir, 'afm', load_kind='force', loads=(30.0,), layers=2, n=10)
    report = import_root(tmp_path, label='camp')
    statuses = _assert_matches_reader(report, tmp_path)
    assert statuses == {10.0: 'complete', 30.0: 'truncated', 50.0: 'not_reached'}


def test_legacy_short_window_is_truncated(aiida_profile_clean, no_profile_switch,
                                          tmp_path):
    """A config without output settings has no expected rows: the old rule
    judges the window against the longest of its family and size."""
    config = {'general': {'temp': 300, 'force': [10.0, 30.0], 'scan_angle': [0.0],
                          'scan_speed': [2.0]}}
    run_dir = make_family_run(tmp_path, 'afm', loads=(10.0, 30.0), n=50, config=config)
    (run_dir / 'lammps' / 'slide.in').write_text(DECK, encoding='utf-8')
    for load in (10.0, 30.0):
        _afm_snapshots(run_dir, load)
    write_results(run_dir, 'afm', load_kind='force', loads=(30.0,), layers=2, n=40)
    report = import_root(tmp_path, label='camp')
    statuses = _assert_matches_reader(report, tmp_path)
    assert statuses == {10.0: 'complete', 30.0: 'truncated'}


@pytest.mark.parametrize('level', ['run', 'parent'])
def test_sheet_bond_reference_is_looked_up_where_the_reader_looks(
        aiida_profile_clean, no_profile_switch, tmp_path, level):
    """Sheet families take the reference from the run's own build/ only: a
    build one level up is no reference (the point is quarantined)."""
    run_dir = make_family_run(tmp_path, 'sheetonsheet', loads=(1.0,), n=50,
                              config=slide_config('sheetonsheet', loads=(1.0,)))
    (run_dir / 'lammps' / 'slide.in').write_text(
        'write_data ./data/final_p$(v_p)GPa_a$(v_a).data\n', encoding='utf-8')
    stack, box = _stack(2)
    build = (run_dir if level == 'run' else run_dir.parent) / 'build'
    _write_data(build / 'sheet.lmp', stack, box)
    _write_data(run_dir / 'data' / 'final_p1GPa_a0.data', stack, box)
    report = import_root(tmp_path, label='camp')
    (status,) = _assert_matches_reader(report, tmp_path).values()
    assert (status == 'quarantined') == (level == 'parent')


def test_reimport_runs_no_new_check(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path)
    import_root(tmp_path, label='camp')
    before = orm.QueryBuilder().append(orm.Node).count()

    again = import_root(tmp_path, label='camp')

    assert again.verdict_counts == {}
    assert orm.QueryBuilder().append(orm.Node).count() == before


def test_dry_run_checks_nothing(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path)
    import_root(tmp_path, label='camp', dry_run=True)
    assert _nodes('check_integrity') == []


# --- rulings (I5) ---------------------------------------------------------------

def test_rulings_pass_is_idempotent(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    first = import_root(tmp_path, label='camp')
    assert first.verdict_counts['quarantined'] == 1
    assert _nodes('apply_ruling') == []

    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    again = import_root(tmp_path, label='camp')

    assert again.skipped_decks == first.created_decks
    (ruled,) = _ruled_outputs()
    assert ruled['status'] == 'complete'
    assert ruled['ruled_by_user'] and not ruled['overridden']
    assert ruled['original_status'] == 'quarantined'
    assert (ruled['note'], ruled['date']) == ('looked at it', '2026-11-16')
    (func,) = _nodes('apply_ruling')
    assert func.inputs.ruling.base.extras.get('friction2d_author') == 'user'
    assert again.rulings.applied == 1

    report = apply_rulings('camp')
    assert (report.applied, report.skipped) == (0, 1)
    assert len(_nodes('apply_ruling')) == 1

    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass', note='second look')
    report = apply_rulings('camp')
    assert (report.applied, report.skipped) == (1, 0)
    live = [n for n in _ruled_outputs() if not n.base.extras.get('superseded', False)]
    assert [n['note'] for n in live] == ['second look']


def test_ruling_for_a_failed_point_is_not_quarantined(aiida_profile_clean,
                                                      no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'pass')

    report = apply_rulings('camp')

    assert report.not_quarantined == [pid]
    assert report.applied == 0 and _nodes('apply_ruling') == []


def test_expert_override_applies_and_is_listed(aiida_profile_clean, no_profile_switch,
                                               tmp_path):
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'pass', note='viewer artefact', override=True)

    report = import_root(tmp_path, label='camp')

    assert report.overrides == [(pid, 'failed_integrity', 'complete', 'viewer artefact')]
    assert 'override' in report.render()
    (ruled,) = _ruled_outputs()
    assert ruled['overridden'] and ruled['original_status'] == 'failed_integrity'
    assert ruled['status'] == 'complete' and not ruled['failed']


def _ruled_live():
    return [n for n in _ruled_outputs() if not n.base.extras.get('superseded', False)]


@pytest.mark.parametrize('edit', ['delete', 'no_override'])
def test_withdrawn_override_returns_the_checker_verdict(
        aiida_profile_clean, no_profile_switch, tmp_path, edit):
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'pass', note='viewer artefact', override=True)
    assert apply_rulings('camp').overrides == [
        (pid, 'failed_integrity', 'complete', 'viewer artefact')]

    if edit == 'delete':
        (tmp_path / RULINGS_FILE).unlink()
    else:
        _write_ruling(tmp_path, pid, 'pass', note='viewer artefact')
    report = import_root(tmp_path, label='camp')

    assert report.rulings.withdrawn == [pid]
    assert report.overrides == [] and report.rulings.overrides == []
    assert _ruled_live() == []
    (back,) = _nodes('withdraw_ruling')
    out = back.outputs.result
    assert out['status'] == 'failed_integrity' and not out.get('overridden')
    assert out['withdrawn_ruling'] == back.inputs.ruled.uuid
    assert apply_rulings('camp').withdrawn == []          # idempotent


def test_partial_explicit_file_withdraws_nothing(aiida_profile_clean, no_profile_switch,
                                                 tmp_path):
    _tree(tmp_path, eject=(30.0,), quarantine=(10.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass', note='ok', override=True)
    assert apply_rulings('camp').applied == 1
    partial = tmp_path / 'partial.yaml'
    partial.write_text(f'- point: "{_pid(tmp_path, 10.0)}"\n  ruling: pass\n',
                       encoding='utf-8')

    report = apply_rulings('camp', partial)

    assert report.applied == 1 and report.withdrawn == []
    assert len(_ruled_live()) == 2 and len(report.overrides) == 1
    # --withdraw on another file withdraws only that file's rulings.
    assert apply_rulings('camp', partial, withdraw=True).withdrawn == []
    partial.write_text('[]\n', encoding='utf-8')
    assert apply_rulings('camp', partial, withdraw=True).withdrawn == [_pid(tmp_path, 10.0)]
    assert len(_ruled_live()) == 1


def test_a_ruling_from_another_file_survives_the_own_file_pass(
        aiida_profile_clean, no_profile_switch, tmp_path):
    """I1: the own-file pass at import withdraws only the own file's rulings."""
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    review = tmp_path / 'reviews' / 'x.yaml'
    review.parent.mkdir()
    review.write_text(f'- point: "{pid}"\n  ruling: pass\n  note: viewer artefact\n'
                      f'  override: true\n', encoding='utf-8')
    assert apply_rulings('camp', review.parent / '.' / 'x.yaml').applied == 1
    (func,) = _nodes('apply_ruling')
    assert func.inputs.ruling['source'] == str(review.resolve())
    assert func.inputs.ruling['own_file'] is False

    (tmp_path / RULINGS_FILE).write_text('[]\n', encoding='utf-8')   # own file empty
    again = import_root(tmp_path, label='camp')

    assert again.rulings.withdrawn == []
    assert again.overrides == [(pid, 'failed_integrity', 'complete', 'viewer artefact')]
    assert len(_ruled_live()) == 1
    assert apply_rulings('camp').withdrawn == []                      # `aiida rule`
    assert apply_rulings('camp', review, withdraw=True).withdrawn == []
    review.write_text('[]\n', encoding='utf-8')
    assert apply_rulings('camp', review, withdraw=True).withdrawn == [pid]
    assert _ruled_live() == []


def test_an_own_file_ruling_records_its_source(aiida_profile_clean, no_profile_switch,
                                               tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    assert apply_rulings('camp', tmp_path).applied == 1
    (func,) = _nodes('apply_ruling')
    assert func.inputs.ruling['own_file'] is True
    assert func.inputs.ruling['source'] == str((tmp_path / RULINGS_FILE).resolve())
    assert apply_rulings('camp').skipped == 1          # one key, whichever form named it


def test_a_ruling_without_a_source_counts_as_the_own_files(
        aiida_profile_clean, no_profile_switch, tmp_path, monkeypatch):
    """Ruling Dicts made before `source`/`own_file` existed are withdrawn by
    the own-file pass, as they always were."""
    from src.aiida import steps
    real = steps._step_input

    def legacy(content, kind):
        if kind == 'ruling':
            content = {k: v for k, v in content.items() if k not in ('source', 'own_file')}
        return real(content, kind)
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'pass', note='viewer artefact', override=True)
    monkeypatch.setattr(steps, '_step_input', legacy)
    assert apply_rulings('camp').applied == 1
    monkeypatch.undo()
    (func,) = _nodes('apply_ruling')
    assert 'own_file' not in func.inputs.ruling.get_dict()

    assert apply_rulings('camp').skipped == 1          # its key still matches
    (tmp_path / RULINGS_FILE).unlink()
    assert import_root(tmp_path, label='camp').rulings.withdrawn == [pid]


def test_unmatched_ruling_and_explicit_path(aiida_profile_clean, no_profile_switch,
                                            tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    elsewhere = tmp_path / 'other.yaml'
    pid = _pid(tmp_path, 30.0).replace('force=30', 'force=99')
    elsewhere.write_text(f'- point: "{pid}"\n  ruling: pass\n', encoding='utf-8')

    report = apply_rulings('camp', elsewhere)

    assert report.unmatched == [pid] and report.applied == 0


# --- clear / delete collect the step inputs ------------------------------------

def test_clear_leaves_no_data(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    assert apply_rulings('camp').applied == 1

    clear_all_nodes()

    assert orm.QueryBuilder().append(orm.Data).count() == 0
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0


def test_delete_leaves_no_step_inputs(aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    assert apply_rulings('camp').applied == 1

    delete_simulation_set('camp')

    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0
    assert orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': 'friction2d_step_input'}}).count() == 0


def test_unreadable_settings_warn(aiida_profile_clean, no_profile_switch, tmp_path,
                                  monkeypatch):
    """`load_settings` skips a broken file silently; the import must not."""
    xdg = tmp_path / 'xdg'
    (xdg / 'FrictionSim2D').mkdir(parents=True)
    (xdg / 'FrictionSim2D' / 'settings.yaml').write_text(
        'integrity: [unclosed\n  hole_bonds: : 2\n', encoding='utf-8')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(xdg))
    root = tmp_path / 'sim'
    _tree(root)
    report = import_root(root, label='camp')
    assert any('default thresholds' in w and 'settings.yaml' in w
               for w in report.warnings), report.warnings


# --- re-checks when the set's context changes -----------------------------------

_LEGACY = {'general': {'temp': 300, 'force': [10.0], 'scan_angle': [0.0],
                       'scan_speed': [2.0]}}


def _legacy_run(root, material, n):
    """A one-load AFM run whose config gives no expected rows."""
    run_dir = make_family_run(root, 'afm', material=material, loads=(10.0,), n=n,
                              config=_LEGACY)
    (run_dir / 'lammps' / 'slide.in').write_text(DECK, encoding='utf-8')
    _afm_snapshots(run_dir, 10.0)
    return run_dir


def test_longer_window_rechecks_an_earlier_legacy_point(aiida_profile_clean,
                                                        no_profile_switch, tmp_path):
    first_dir = _legacy_run(tmp_path, 'h-MoS2', n=40)
    first = import_root(tmp_path, label='camp')
    assert first.verdict_counts == {'complete': 1}
    (old_uuid,) = first.points

    _legacy_run(tmp_path, 'h-WS2', n=50)
    again = import_root(tmp_path, label='camp')

    rel = first_dir.relative_to(tmp_path).as_posix()
    assert again.rechecked == [(f'{rel}|force=10|angle=0|speed=2', 'complete', 'truncated')]
    assert 're-checked' in again.render()
    assert _assert_matches_reader(again, tmp_path, uuids=[old_uuid]) == {10.0: 'truncated'}
    assert _assert_matches_reader(again, tmp_path) == {10.0: 'complete'}
    assert len(_verdicts_of(orm.load_node(old_uuid))) == 2     # old one superseded

    # A same-window run joins the group: everything is re-examined
    # (`_unchanged`), nothing differs, only the new point is checked.
    _legacy_run(tmp_path, 'h-NbSe2', n=50)
    third = import_root(tmp_path, label='camp')
    assert third.rechecked == [] and third.verdict_counts == {'complete': 1}
    assert len(_verdicts_of(orm.load_node(old_uuid))) == 2


def test_recheck_drops_the_rulings_of_the_dead_verdict(aiida_profile_clean,
                                                       no_profile_switch, tmp_path):
    _legacy_run(tmp_path, 'h-MoS2', n=40)
    first = import_root(tmp_path, label='camp')
    (point_uuid,) = first.points
    dead = _live_verdict(orm.load_node(point_uuid))
    pid = _pid(tmp_path, 10.0)
    _write_ruling(tmp_path, pid, 'fail', note='looks wrong', override=True)
    assert apply_rulings('camp').applied == 1

    _legacy_run(tmp_path, 'h-WS2', n=50)
    again = import_root(tmp_path, label='camp')

    assert [r[0] for r in again.rechecked] == [pid]
    on_dead = [n for n in orm.QueryBuilder().append(
        orm.Dict, filters={'extras.friction2d_verdict_uuid': dead.uuid}).all(flat=True)
        if not n.base.extras.get('superseded', False)]
    assert on_dead == []
    live = _ruled_live()
    assert [n.base.extras.get('friction2d_verdict_uuid') for n in live] == [
        _live_verdict(orm.load_node(point_uuid)).uuid]          # re-applied
    assert live[0]['original_status'] == 'truncated'


def test_stage2_failure_keeps_verdicts_and_checks_new_points_provisionally(
        aiida_profile_clean, no_profile_switch, tmp_path, monkeypatch):
    """A run whose window cannot be read must not lower its group's longest
    window: MoS2 keeps `truncated`; the new NbSe2 point is provisional until a
    healthy import re-checks it."""
    import src.aiida.steps as steps

    _legacy_run(tmp_path, 'h-MoS2', n=40)
    _legacy_run(tmp_path, 'h-WS2', n=50)
    import_root(tmp_path, label='camp')
    by_material = lambda m: next(orm.load_node(u) for u in _uuids()  # noqa: E731
                                 if m in orm.load_node(u).base.attributes.get('rel_path'))
    mos2 = by_material('h-MoS2')
    assert _live_verdict(mos2)['status'] == 'truncated'

    real = steps._window_rows

    def flaky(point, directory):
        if 'h-WS2' in point.base.attributes.get('rel_path'):
            raise OSError('disk gone')
        return real(point, directory)
    monkeypatch.setattr(steps, '_window_rows', flaky)
    _legacy_run(tmp_path, 'h-NbSe2', n=40)
    broken = import_root(tmp_path, label='camp')

    assert broken.rechecked == []
    assert _live_verdict(mos2)['status'] == 'truncated'
    nbse2 = by_material('h-NbSe2')
    provisional = _live_verdict(nbse2)
    assert provisional.base.extras.get('friction2d_provisional') is True
    assert provisional['status'] == 'complete'             # parser status
    assert any('provisionally' in w for w in broken.warnings), broken.warnings

    monkeypatch.setattr(steps, '_window_rows', real)
    healthy = import_root(tmp_path, label='camp')
    rel = nbse2.base.attributes.get('rel_path')
    assert healthy.rechecked == [(f'{rel}|force=10|angle=0|speed=2', 'complete', 'truncated')]
    assert _assert_matches_reader(healthy, tmp_path,
                                  uuids=[nbse2.uuid, mos2.uuid]) == {10.0: 'truncated'}
    assert not _live_verdict(nbse2).base.extras.get('friction2d_provisional')
    assert import_root(tmp_path, label='camp').rechecked == []


def _uuids():
    return [p.uuid for p in orm.QueryBuilder().append(
        orm.FolderData, filters={'attributes': {'has_key': 'point_uuid'}}).all(flat=True)
        if not p.base.extras.get('superseded', False)]


def test_explicit_own_rulings_path_counts_a_missing_file_as_empty(
        aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass', note='ok', override=True)
    assert apply_rulings('camp').applied == 1
    (tmp_path / RULINGS_FILE).unlink()

    report = apply_rulings('camp', tmp_path / RULINGS_FILE)

    assert report.withdrawn == [_pid(tmp_path, 30.0)] and report.overrides == []


def test_stage2_failure_falls_back_to_parser_status(aiida_profile_clean,
                                                   no_profile_switch, tmp_path,
                                                   monkeypatch):
    import src.aiida.steps as steps

    def broken(*_args):
        raise OSError('disk gone')
    monkeypatch.setattr(steps, '_window_rows', broken)
    _legacy_run(tmp_path, 'h-MoS2', n=40)
    _legacy_run(tmp_path, 'h-WS2', n=50)
    report = import_root(tmp_path, label='camp')
    assert report.ok(), report.render()
    assert report.verdict_counts == {'complete': 2}
    assert any('stage-2 status failed' in w for w in report.warnings)


@pytest.mark.parametrize('form', ['default', 'file', 'directory'])
def test_unreachable_root_withdraws_nothing(aiida_profile_clean, no_profile_switch,
                                            tmp_path, form):
    _tree(tmp_path, eject=(30.0,))
    import_root(tmp_path, label='camp')
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass', note='ok', override=True)
    assert apply_rulings('camp').applied == 1
    group = orm.Group.collection.get(label='friction2d/camp')
    unmounted = tmp_path / 'unmounted'
    group.base.extras.set('source_root', str(unmounted))

    path = {'default': None, 'file': unmounted / RULINGS_FILE,
            'directory': unmounted}[form]
    report = apply_rulings('camp', path)

    assert report.withdrawn == [] and len(report.overrides) == 1
    assert any('not reachable' in w for w in report.warnings)


def test_provisional_recheck_to_the_same_status_is_not_reported(
        aiida_profile_clean, no_profile_switch, tmp_path, monkeypatch):
    import src.aiida.steps as steps

    _legacy_run(tmp_path, 'h-MoS2', n=50)
    _legacy_run(tmp_path, 'h-WS2', n=40)
    import_root(tmp_path, label='camp')
    real = steps._window_rows

    def flaky(point, directory):
        if 'h-WS2' in point.base.attributes.get('rel_path'):
            raise OSError('disk gone')
        return real(point, directory)
    monkeypatch.setattr(steps, '_window_rows', flaky)
    _legacy_run(tmp_path, 'h-NbSe2', n=50)
    import_root(tmp_path, label='camp')
    nbse2 = next(orm.load_node(u) for u in _uuids()
                 if 'h-NbSe2' in orm.load_node(u).base.attributes.get('rel_path'))
    assert _live_verdict(nbse2).base.extras.get('friction2d_provisional') is True

    monkeypatch.setattr(steps, '_window_rows', real)
    healthy = import_root(tmp_path, label='camp')

    assert healthy.rechecked == []                      # complete -> complete
    assert healthy.verdict_counts == {'complete': 1}    # but it was re-checked
    assert not _live_verdict(nbse2).base.extras.get('friction2d_provisional')
    assert _live_verdict(nbse2)['status'] == 'complete'


def test_ruling_that_stops_applying_after_a_recheck_says_why(
        aiida_profile_clean, no_profile_switch, tmp_path):
    """A quarantined point's plain ruling stops applying once a re-check
    moves the point out of quarantine; the warning must say so."""
    run_dir = _legacy_run(tmp_path, 'h-MoS2', n=40)
    for path in (run_dir / 'data').glob('final_f10N*'):
        path.unlink()
    first = import_root(tmp_path, label='camp')
    assert first.verdict_counts == {'quarantined': 1}
    pid = _pid(tmp_path, 10.0)
    _write_ruling(tmp_path, pid, 'pass')
    assert apply_rulings('camp').applied == 1

    _legacy_run(tmp_path, 'h-WS2', n=50)
    again = import_root(tmp_path, label='camp')

    assert again.rechecked == [(pid, 'quarantined', 'truncated')]
    assert _ruled_live() == []
    msgs = [w for w in again.warnings + again.rulings.warnings if pid in w]
    assert any('no longer quarantined' in w and 'override: true' in w
               and 'quarantined to truncated' in w for w in msgs), msgs


# --- aiida recheck: new integrity settings (I2) ------------------------------------

def _settings_file(path, version=None, **integrity):
    if version is not None:
        integrity['version'] = version
    path.write_text(yaml.safe_dump({'integrity': integrity}), encoding='utf-8')
    return path


@pytest.fixture
def no_user_settings(monkeypatch, tmp_path):
    """No per-user settings file: the import checks under the defaults."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'no-xdg'))


def test_recheck_applies_new_thresholds_and_reapplies_rulings(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root = tmp_path / 'sim'
    _tree(root, eject=(30.0,))
    first = import_root(root, label='camp')
    assert first.verdict_counts == {'complete': 2, 'failed_integrity': 1}
    pid10, pid30 = _pid(root, 10.0), _pid(root, 30.0)
    _write_ruling(root, pid10, 'fail', note='tip artefact', override=True)
    assert apply_rulings('camp').applied == 1
    loose = _settings_file(tmp_path / 'settings.yaml', version='1.1.0',
                           ejected_atoms=3, isolated_atoms=3)

    report = recheck_set('camp', settings_file=loose)

    assert report.version == '1.1.0'
    assert sum(report.verdict_counts.values()) == 3
    assert report.rechecked == [(pid30, 'failed_integrity', 'complete')]
    assert f're-checked {pid30}: failed_integrity -> complete' in report.render()
    statuses = {}
    for point in (orm.load_node(u) for u in first.points):
        verdict = _live_verdict(point)
        assert verdict['thresholds_version'] == '1.1.0'
        statuses[point.base.attributes.get('load')] = verdict['status']
    assert statuses[30.0] == 'complete'
    # The ruling re-applies to the new verdict; the old ruled one is superseded.
    (ruled,) = _ruled_live()
    assert ruled.creator.inputs.verdict['thresholds_version'] == '1.1.0'
    assert ruled['status'] == 'failed_integrity' and ruled['overridden']
    assert report.rulings.applied == 1 and report.rulings.withdrawn == []

    checks = len(_nodes('check_integrity'))
    again = recheck_set('camp', settings_file=loose)
    assert again.verdict_counts == {} and again.rechecked == []
    assert (again.rulings.applied, again.rulings.skipped) == (0, 1)
    assert len(_nodes('check_integrity')) == checks


def test_recheck_follows_a_settings_edit_without_a_version_bump(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root = tmp_path / 'sim'
    _tree(root, eject=(30.0,))
    import_root(root, label='camp')
    assert recheck_set('camp').verdict_counts == {}        # same settings: nothing

    edited = _settings_file(tmp_path / 'settings.yaml', ejected_atoms=3, isolated_atoms=3)
    report = recheck_set('camp', settings_file=edited)

    assert report.version == '1.0.0' and sum(report.verdict_counts.values()) == 3
    assert report.rechecked == [(_pid(root, 30.0), 'failed_integrity', 'complete')]


def _foreign_setup(tmp_path):
    """An ejected 30 N point with a pass override from another file, x.yaml."""
    root = tmp_path / 'sim'
    _tree(root, eject=(30.0,))
    import_root(root, label='camp')
    pid = _pid(root, 30.0)
    review = tmp_path / 'x.yaml'
    review.write_text(f'- point: "{pid}"\n  ruling: pass\n'
                      f'  note: fine\n  override: true\n', encoding='utf-8')
    assert apply_rulings('camp', review).applied == 1
    return root, pid, review


def _bump(tmp_path):
    return _settings_file(tmp_path / 'settings.yaml', version='1.1.0')


def _live_foreign():
    return [(n['status'], n.creator.inputs.ruling['source'],
             n.creator.inputs.verdict['thresholds_version']) for n in _ruled_live()]


def test_recheck_reapplies_a_foreign_ruling_from_its_source(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root, pid, review = _foreign_setup(tmp_path)

    report = recheck_set('camp', settings_file=_bump(tmp_path))

    assert report.rechecked == [] and sum(report.verdict_counts.values()) == 3
    assert _live_foreign() == [('complete', str(review.resolve()), '1.1.0')]
    assert report.rulings.pending == []
    assert not any('--rulings' in w for w in report.warnings)
    assert recheck_set('camp', settings_file=_bump(tmp_path)).rulings.applied == 0


def test_a_foreign_ruling_whose_file_is_missing_is_pending(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root, pid, review = _foreign_setup(tmp_path)
    moved = review.with_suffix('.bak')
    review.rename(moved)

    report = recheck_set('camp', settings_file=_bump(tmp_path))

    assert _ruled_live() == []
    src = str(review.resolve())
    assert report.rulings.pending == [(pid, src, 'missing')]
    assert f'ruling pending: {src} missing' in report.render()
    assert f'--rulings {src}' in report.render()
    again = import_root(root, label='camp', settings_file=_bump(tmp_path))
    assert again.rulings.pending == [(pid, src, 'missing')]
    assert f'ruling pending: {src} missing' in again.render()
    assert apply_rulings('camp').pending == [(pid, src, 'missing')]   # `aiida rule`

    moved.rename(review)                        # the file is back: re-applied
    back = import_root(root, label='camp', settings_file=_bump(tmp_path))
    assert back.rulings.pending == []
    assert _live_foreign() == [('complete', src, '1.1.0')]


def test_a_pending_ruling_ends_when_its_file_withdraws_it(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root, pid, review = _foreign_setup(tmp_path)
    review.write_text('[]\n', encoding='utf-8')

    report = recheck_set('camp', settings_file=_bump(tmp_path))

    src = str(review.resolve())
    assert report.rulings.pending == [(pid, src, 'no longer rules the point')]
    assert apply_rulings('camp', review).pending == [
        (pid, src, 'no longer rules the point')]     # not withdrawn without --withdraw
    withdrawn = apply_rulings('camp', review, withdraw=True)
    assert withdrawn.pending == [] and withdrawn.pending_withdrawn == [(pid, src)]
    assert apply_rulings('camp').pending == []


def test_a_pending_ruling_of_a_lost_file_is_withdrawn_explicitly(
        aiida_profile_clean, no_profile_switch, no_user_settings, tmp_path):
    root, pid, review = _foreign_setup(tmp_path)
    review.unlink()
    recheck_set('camp', settings_file=_bump(tmp_path))

    report = apply_rulings('camp', withdraw_pending=review)

    assert report.pending == [] and report.pending_withdrawn == [(pid, str(review.resolve()))]
    assert import_root(root, label='camp', settings_file=_bump(tmp_path)).rulings.pending == []


# --- own file wins over any other rulings file -------------------------------

def _review(root, pid, ruling, note='expert pass'):
    review = root / 'reviews' / 'x.yaml'
    review.parent.mkdir(exist_ok=True)
    review.write_text(f'- point: "{pid}"\n  ruling: {ruling}\n  note: {note}\n',
                      encoding='utf-8')
    return review


def _live_rulings():
    return [(n['status'], n.creator.inputs.ruling['own_file']) for n in _ruled_live()]


def test_another_file_cannot_overrule_the_own_file(aiida_profile_clean, no_profile_switch,
                                                   tmp_path):
    """The own file rules first; another file's ruling on the same point is not
    applied, and the conflict is listed by `aiida rule` and every import."""
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'fail', note='own says fail')
    assert apply_rulings('camp').applied == 1
    review = _review(tmp_path, pid, 'pass')

    other = apply_rulings('camp', review)

    assert other.applied == 0
    assert _live_rulings() == [('failed_integrity', True)]
    conflict = (pid, 'fail', str(review.resolve()), 'pass')
    assert other.conflicts == [conflict]
    assert f'conflict {pid}' in other.render()
    for _ in range(2):                                   # every import, not once
        again = import_root(tmp_path, label='camp')
        assert again.rulings.applied == 0
        assert again.rulings.conflicts == [conflict]
        assert f'ruling conflict {pid}' in again.render()
        assert _live_rulings() == [('failed_integrity', True)]
    assert apply_rulings('camp').conflicts == [conflict]    # `aiida rule`


def test_the_own_file_takes_a_point_another_file_ruled_first(
        aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    review = _review(tmp_path, pid, 'pass')
    assert apply_rulings('camp', review).applied == 1    # the own file rules nothing yet
    assert _live_rulings() == [('complete', False)]

    _write_ruling(tmp_path, pid, 'fail', note='own says fail')
    again = import_root(tmp_path, label='camp')

    assert _live_rulings() == [('failed_integrity', True)]
    conflict = (pid, 'fail', str(review.resolve()), 'pass')
    assert again.rulings.conflicts == [conflict]
    assert import_root(tmp_path, label='camp').rulings.conflicts == [conflict]
    assert apply_rulings('camp', review).conflicts == [conflict]
    assert _live_rulings() == [('failed_integrity', True)]


def test_the_same_decision_from_both_files_is_no_conflict(aiida_profile_clean,
                                                          no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'fail', note='own says fail')
    assert apply_rulings('camp').applied == 1
    review = _review(tmp_path, pid, 'fail', note='I agree')

    other = apply_rulings('camp', review)

    assert (other.applied, other.conflicts) == (0, [])
    assert _live_rulings() == [('failed_integrity', True)]
    assert import_root(tmp_path, label='camp').rulings.conflicts == []


def test_the_own_file_agreeing_with_an_earlier_ruling_is_no_conflict(
        aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    assert apply_rulings('camp', _review(tmp_path, pid, 'pass')).applied == 1
    _write_ruling(tmp_path, pid, 'pass', note='own agrees')

    again = import_root(tmp_path, label='camp')

    assert _live_rulings() == [('complete', True)]
    assert again.rulings.conflicts == []
    assert import_root(tmp_path, label='camp').rulings.conflicts == []

def test_a_conflict_ends_with_the_other_files_ruling(aiida_profile_clean,
                                                     no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'fail', note='own says fail')
    assert apply_rulings('camp').applied == 1
    review = _review(tmp_path, pid, 'pass')
    assert apply_rulings('camp', review).conflicts
    review.write_text('[]\n', encoding='utf-8')

    assert apply_rulings('camp', review).conflicts == []
    assert import_root(tmp_path, label='camp').rulings.conflicts == []


def test_a_conflict_whose_own_ruling_is_gone_warns(aiida_profile_clean, no_profile_switch,
                                                   tmp_path):
    """The other file's ruling is not applied behind the user's back."""
    _tree(tmp_path, quarantine=(30.0,))
    import_root(tmp_path, label='camp')
    pid = _pid(tmp_path, 30.0)
    _write_ruling(tmp_path, pid, 'fail', note='own says fail')
    assert apply_rulings('camp').applied == 1
    review = _review(tmp_path, pid, 'pass')
    apply_rulings('camp', review)
    (tmp_path / RULINGS_FILE).write_text('[]\n', encoding='utf-8')

    again = import_root(tmp_path, label='camp')

    assert again.rulings.withdrawn == [pid]
    assert again.rulings.conflicts == []
    assert _ruled_live() == []
    assert any(str(review.resolve()) in w and '--rulings' in w for w in again.warnings)
    assert apply_rulings('camp', review).applied == 1
    assert _live_rulings() == [('complete', False)]
    assert not any('--rulings' in w for w in import_root(tmp_path, label='camp').warnings)
