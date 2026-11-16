"""Integrity verdicts at import and the idempotent rulings pass (plan Task 6a).

Every imported point gets one verdict from `steps.check_integrity`, the
ported checker (`integrity.check_files`) run as a calcfunction; the user's
`integrity_rulings.yaml` is applied by `steps.apply_rulings`, one tracked
`apply_ruling` per (verdict, ruling) pair.
"""
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402
from aiida.common.links import LinkType  # noqa: E402

from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.integration import clear_all_nodes, delete_simulation_set  # noqa: E402
from src.aiida.steps import apply_rulings  # noqa: E402
from src.postprocessing.integrity_rulings import RULINGS_FILE, point_id  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.postprocessing.test_read_data_integrity import _afm_tree, _point  # noqa: E402

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
    assert set(func.base.links.get_incoming().all_link_labels()) == {
        'point', 'context', 'settings'}


def test_verdicts_match_the_datareader(aiida_profile_clean, no_profile_switch, tmp_path):
    """The store's verdict is the one `postprocess read` gives the same tree."""
    _tree(tmp_path, eject=(10.0,), quarantine=(30.0,), no_output=(50.0,))
    report = import_root(tmp_path, label='camp')
    assert report.ok(), report.render()

    reader = DataReader(str(tmp_path))
    for uuid in report.points:
        point = orm.load_node(uuid)
        (verdict,) = _verdicts_of(point)
        expected = _point(reader, point.base.attributes.get('load'),
                          points=True).integrity.as_dict()
        got = verdict.get_dict()
        for key in ('status', 'failed', 'observed', 'reasons', 'quarantine',
                    'not_applicable', 'thresholds_version'):
            assert got[key] == expected[key], key
        if got['status'] == 'no_output':
            # The plan gives a no_output point no reference, so no n_heat;
            # the DataReader counts it from the load structure anyway.
            assert got['measures'] == {'n_heat': None}
            continue
        assert got['measures'].keys() == expected['measures'].keys()
        for key, value in expected['measures'].items():
            assert got['measures'][key] == pytest.approx(value, nan_ok=True), key


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
