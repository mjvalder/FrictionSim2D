"""Tracked summaries, the latest verdict, and datasets (plan Task 7).

`steps.summarise` is a calcfunction over one point; `postprocess_group` runs
it on every live point whose selected verdict is `complete`; `build_dataset`
gathers the selected summaries into `friction2d/datasets/<name>` with one
Dict naming the policy. Selection is `policy.select`: by declared version,
never by ctime (critique I4).
"""
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

import src.aiida.steps as steps  # noqa: E402
from src.aiida.block_schema import SUMMARY_UNITS  # noqa: E402
from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.integration import clear_all_nodes, delete_simulation_set  # noqa: E402
from src.aiida.policy import Policy  # noqa: E402
from src.aiida.steps import (DATASET_PREFIX, apply_rulings, build_dataset,  # noqa: E402
                             is_provisional, latest_verdict, merge_units,
                             postprocess_group)
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.aiida.test_steps_integrity import (  # noqa: E402
    _pid, _tree, _verdicts_of, _write_ruling)


@pytest.fixture
def no_profile_switch(monkeypatch):
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _points(report):
    return {orm.load_node(u).base.attributes.get('load'): orm.load_node(u)
            for u in report.points}


def _summaries(point=None):
    qb = orm.QueryBuilder()
    if point is not None:
        qb.append(orm.FolderData, filters={'id': point.pk}, tag='p')
        qb.append(orm.CalcFunctionNode, with_incoming='p', tag='f',
                  filters={'attributes.process_label': 'summarise'})
    else:
        qb.append(orm.CalcFunctionNode, tag='f',
                  filters={'attributes.process_label': 'summarise'})
    qb.append(orm.Dict, with_incoming='f')
    return qb.all(flat=True)


def _checker(point):
    (verdict,) = [v for v in _verdicts_of(point)
                  if not v.base.extras.get('superseded', False)]
    return verdict


# --- latest_verdict -----------------------------------------------------------

def test_latest_is_the_checker_verdict_without_rulings(aiida_profile_clean,
                                                       no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    points = _points(import_root(tmp_path, label='camp'))
    for point in points.values():
        assert latest_verdict(point).uuid == _checker(point).uuid
        assert not is_provisional(latest_verdict(point))


def test_ruled_beats_checker_and_withdrawn_beats_checker(aiida_profile_clean,
                                                         no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    point = _points(import_root(tmp_path, label='camp'))[30.0]
    checker = _checker(point)
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    assert apply_rulings('camp').applied == 1

    ruled = latest_verdict(point)
    assert ruled.base.extras.get('friction2d_kind') == 'ruled_verdict'
    assert ruled['status'] == 'complete' and checker['status'] == 'quarantined'

    (tmp_path / 'integrity_rulings.yaml').unlink()
    assert apply_rulings('camp').withdrawn
    back = latest_verdict(point)
    assert back.base.extras.get('friction2d_kind') == 'withdrawn_verdict'
    assert back['status'] == 'quarantined'

    # All three live at once (not what the rulings pass leaves): ruled wins.
    ruled.base.extras.delete('superseded')
    assert latest_verdict(point).uuid == ruled.uuid


def test_latest_is_by_thresholds_version_not_creation_time(aiida_profile_clean,
                                                           no_profile_switch, tmp_path):
    _tree(tmp_path)
    point = _points(import_root(tmp_path, label='camp'))[30.0]
    first = _checker(point)
    func = first.creator
    old_settings = orm.Dict({**func.inputs.settings.get_dict(), 'version': '0.9.0'})
    inputs = {k: func.inputs[k] for k in ('point', 'context')}
    for key in ('snapshot', 'reference'):
        if key in func.inputs:
            inputs[key] = func.inputs[key]
    older_version = steps.check_integrity(**inputs, settings=old_settings)
    assert older_version.ctime > first.ctime
    assert older_version['thresholds_version'] == '0.9.0'

    assert latest_verdict(point).uuid == first.uuid
    assert latest_verdict(point, Policy(verdict='0.9.0')).uuid == older_version.uuid
    assert latest_verdict(point, Policy(verdict='9.9')) is None


def test_provisional_is_reported_through_a_ruling(aiida_profile_clean,
                                                  no_profile_switch, tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    point = _points(import_root(tmp_path, label='camp'))[30.0]
    _checker(point).base.extras.set('friction2d_provisional', True)
    assert is_provisional(latest_verdict(point))
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    assert apply_rulings('camp').applied == 1
    ruled = latest_verdict(point)
    assert ruled.base.extras.get('friction2d_kind') == 'ruled_verdict'
    assert is_provisional(ruled)


# --- summarise / postprocess_group ------------------------------------------

def test_postprocess_summarises_complete_points_only(aiida_profile_clean,
                                                     no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,))
    points = _points(import_root(tmp_path, label='camp'))
    assert latest_verdict(points[10.0])['status'] == 'failed_integrity'

    assert postprocess_group('camp') == 2
    assert _summaries(points[10.0]) == []
    for load in (30.0, 50.0):
        (summary,) = _summaries(points[load])
        content = summary.get_dict()
        assert content['point_uuid'] == points[load].base.attributes.get('point_uuid')
        assert content['code_version'] == steps.CODE_VERSION
        assert content['units']['mean_cof'] == '1'
        assert content['units']['mean_nf'] == 'nN'
        assert set(content['units']) <= set(SUMMARY_UNITS['afm'])
        assert content['mean_cof'] is not None
        assert summary.base.extras.get('friction2d_kind') == 'summary'

    assert postprocess_group('camp') == 0                 # idempotent


def test_summaries_match_the_datareader(aiida_profile_clean, no_profile_switch, tmp_path):
    """The one COF: the store's summary is `postprocess read`'s summary row."""
    _tree(tmp_path, eject=(10.0,))
    points = _points(import_root(tmp_path, label='camp'))
    postprocess_group('camp')
    rows = {r['load']: r for r in DataReader(str(tmp_path)).summary_rows()}
    for load in (30.0, 50.0):
        (summary,) = _summaries(points[load])
        for key in ('mean_cof', 'std_cof', 'mean_nf', 'mean_lfx', 'std_lfy'):
            assert summary[key] == pytest.approx(rows[load][key]), key
        assert summary['ntimesteps'] == rows[load]['ntimesteps']
        assert summary['time_series_hash'] == rows[load]['time_series_hash']
        # The post-stream channel, from the end snapshot and provenance.json.
        assert summary['deflect_max'] == pytest.approx(rows[load]['deflect_max'])
        assert summary['units']['deflect_max'] == 'angstrom'
        assert 'gap_min' not in summary.get_dict()       # nan in the row: no tip atom


def test_a_code_version_bump_adds_summaries_beside_the_old(aiida_profile_clean,
                                                           no_profile_switch, tmp_path,
                                                           monkeypatch):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    assert postprocess_group('camp') == 2
    old = {s.uuid for s in _summaries()}

    monkeypatch.setattr(steps, 'CODE_VERSION', '99.0.0')
    assert postprocess_group('camp') == 2
    assert postprocess_group('camp') == 0
    every = _summaries()
    assert len(every) == 4
    assert {s.uuid for s in every if s['code_version'] == '99.0.0'}.isdisjoint(old)
    assert not any(s.base.extras.get('superseded', False) for s in every)


def test_a_ruled_complete_point_is_summarised(aiida_profile_clean, no_profile_switch,
                                              tmp_path):
    _tree(tmp_path, quarantine=(30.0,))
    points = _points(import_root(tmp_path, label='camp'))
    assert postprocess_group('camp') == 2                 # 10 and 50 complete
    assert _summaries(points[30.0]) == []
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    apply_rulings('camp')
    assert postprocess_group('camp') == 1
    assert len(_summaries(points[30.0])) == 1


def test_stale_thresholds_and_provisional_verdicts_are_reported(
        aiida_profile_clean, no_profile_switch, tmp_path, monkeypatch):
    _tree(tmp_path)
    points = _points(import_root(tmp_path, label='camp'))
    _checker(points[30.0]).base.extras.set('friction2d_provisional', True)
    current = steps._settings_dump()
    monkeypatch.setattr(steps, '_settings_dump',
                        lambda warnings=None: {**current, 'version': '1.1.0'})

    warnings = []
    assert postprocess_group('camp', warnings=warnings) == 3
    stale = [w for w in warnings if 'thresholds v1.0.0' in w]
    assert len(stale) == 1 and '3 point' in stale[0] and 'v1.1.0' in stale[0]
    provisional = [w for w in warnings if 'provisional' in w]
    assert len(provisional) == 1 and _pid(tmp_path, 30.0) in provisional[0]
    (summary,) = _summaries(points[30.0])
    assert summary.base.extras.get('friction2d_provisional') is True
    (other,) = _summaries(points[10.0])
    assert other.base.extras.get('friction2d_provisional') is False


def test_a_failing_summary_is_skipped_or_raised(aiida_profile_clean, no_profile_switch,
                                                tmp_path, monkeypatch):
    _tree(tmp_path)
    import_root(tmp_path, label='camp')

    def boom(*args, **kwargs):
        raise RuntimeError('broken frame')
    monkeypatch.setattr(steps, 'summarise', boom)
    warnings = []
    assert postprocess_group('camp', warnings=warnings) == 0
    assert sum('broken frame' in w for w in warnings) == 3
    with pytest.raises(RuntimeError, match='broken frame'):
        postprocess_group('camp', skip_failed=False)


# --- datasets -----------------------------------------------------------------

def test_build_dataset_groups_summaries_and_one_dict(aiida_profile_clean,
                                                     no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,))
    points = _points(import_root(tmp_path, label='camp'))
    postprocess_group('camp')

    table = build_dataset('d', ['camp'])

    group = orm.Group.collection.get(label=f'{DATASET_PREFIX}/d')
    nodes = list(group.nodes)
    summaries = [n for n in nodes if n.base.extras.get('friction2d_kind') == 'summary']
    dicts = [n for n in nodes if n.base.extras.get('friction2d_kind') == 'dataset']
    assert len(summaries) == 2 and len(nodes) == 3
    assert dicts == [table] or [d.uuid for d in dicts] == [table.uuid]
    assert table['policy'] == {'verdict': 'newest', 'summary': 'newest'}
    assert set(table['units']) == {'afm'}
    stored = {}
    for summary in summaries:
        stored.update(summary['units'])
    assert table['units']['afm'] == stored
    assert stored.items() <= SUMMARY_UNITS['afm'].items() and stored['mean_cof'] == '1'
    assert set(table['columns']) >= {'point_uuid', 'calc_uuid', 'summary_uuid',
                                     'block_hashes'}
    rows = {r['point_uuid']: r for r in table['rows']}
    assert set(rows) == {points[x].base.attributes.get('point_uuid') for x in (30.0, 50.0)}
    for load in (30.0, 50.0):
        row = rows[points[load].base.attributes.get('point_uuid')]
        assert row['calc_uuid'] == points[load].creator.uuid
        assert row['summary_uuid'] == _summaries(points[load])[0].uuid
        blocks = points[load].creator.inputs.blocks
        assert row['block_hashes'] == {
            name: node.base.extras.get('friction2d_block_hash')
            for name, node in blocks.items()}
        assert row['block_hashes']
        assert row['verdict_status'] == 'complete' and row['provisional'] is False
    with pytest.raises(ValueError, match='exists'):
        build_dataset('d', ['camp'])


def test_dataset_policy_selects_the_summary_generation(aiida_profile_clean,
                                                       no_profile_switch, tmp_path,
                                                       monkeypatch):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')
    old = steps.CODE_VERSION
    monkeypatch.setattr(steps, 'CODE_VERSION', '99.0.0')
    postprocess_group('camp')

    newest = build_dataset('new', ['camp'])
    pinned = build_dataset('old', ['camp'], policy=Policy(summary=old))
    by_uuid = {s.uuid: s['code_version'] for s in _summaries()}
    assert {by_uuid[r['summary_uuid']] for r in newest['rows']} == {'99.0.0'}
    assert {by_uuid[r['summary_uuid']] for r in pinned['rows']} == {old}
    assert pinned['policy'] == {'verdict': 'newest', 'summary': old}
    assert build_dataset('none', ['camp'], policy=Policy(summary='0.0.1'))['rows'] == []


def test_dataset_leaves_out_points_no_longer_complete(aiida_profile_clean,
                                                      no_profile_switch, tmp_path):
    """A summary outlives a ruling withdrawn later: the dataset re-selects the
    verdict, so the point drops out."""
    _tree(tmp_path, quarantine=(30.0,))
    points = _points(import_root(tmp_path, label='camp'))
    _write_ruling(tmp_path, _pid(tmp_path, 30.0), 'pass')
    apply_rulings('camp')
    postprocess_group('camp')
    (tmp_path / 'integrity_rulings.yaml').unlink()
    apply_rulings('camp')

    table = build_dataset('d', ['camp'])
    uuid = points[30.0].base.attributes.get('point_uuid')
    assert uuid not in {r['point_uuid'] for r in table['rows']}
    assert len(table['rows']) == 2


def test_merge_units_refuses_disagreeing_units():
    units = {}
    merge_units(units, 'afm', {'mean_nf': 'nN', 'mean_cof': '1'})
    merge_units(units, 'afm', {'mean_nf': 'nN'})
    merge_units(units, 'sheetonsheet', {'mean_nf': 'eV/angstrom'})
    assert units == {'afm': {'mean_nf': 'nN', 'mean_cof': '1'},
                     'sheetonsheet': {'mean_nf': 'eV/angstrom'}}
    with pytest.raises(ValueError, match="mean_nf"):
        merge_units(units, 'afm', {'mean_nf': 'eV/angstrom'})


# --- delete / clear respect datasets -------------------------------------------

def test_delete_refuses_a_set_a_dataset_draws_on(aiida_profile_clean, no_profile_switch,
                                                 tmp_path):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')
    build_dataset('paper', ['camp'])
    before = orm.QueryBuilder().append(orm.Node).count()

    with pytest.raises(ValueError, match="dataset.*'paper'"):
        delete_simulation_set('camp')

    assert orm.QueryBuilder().append(orm.Node).count() == before
    assert orm.Group.collection.get(label='friction2d/camp')


def test_delete_without_a_dataset_takes_the_summaries(aiida_profile_clean,
                                                      no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')

    delete_simulation_set('camp')

    assert _summaries() == []
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0
    assert orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': 'friction2d_step_input'}}).count() == 0


def test_clear_takes_datasets_and_summaries(aiida_profile_clean, no_profile_switch,
                                            tmp_path):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')
    build_dataset('paper', ['camp'])

    clear_all_nodes()

    assert orm.QueryBuilder().append(orm.Data).count() == 0
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0
    assert not orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': 'friction2d/%'}}).count()
