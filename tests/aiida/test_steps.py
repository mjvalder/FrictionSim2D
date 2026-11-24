"""Tracked summaries, the latest verdict, and datasets (plan Task 7).

`steps.summarise` is a calcfunction over one point; `postprocess_group` runs
it on every live point whose selected verdict is `complete`; `build_dataset`
gathers the selected summaries into `friction2d/datasets/<name>` with one
Dict naming the policy. Selection is `policy.select`: by declared version,
never by ctime (critique I4).
"""
import logging
import numbers
import re

import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402
from aiida.common import exceptions  # noqa: E402

import frictionsim2d.aiida.steps as steps  # noqa: E402
from frictionsim2d.aiida.block_schema import SUMMARY_UNITS  # noqa: E402
from frictionsim2d.aiida.ingest import import_root  # noqa: E402
from frictionsim2d.aiida.layout import DATASET_PREFIX  # noqa: E402
from frictionsim2d.aiida.rulings import Policy  # noqa: E402
from frictionsim2d.aiida.steps import (apply_rulings, build_dataset,  # noqa: E402
                             clear_all_nodes, delete_dataset, delete_simulation_set,
                             is_provisional, latest_verdict,
                             merge_units, normalise_options, options_hash,
                             postprocess_group, summary_options)
from frictionsim2d.postprocessing.read_data import DataReader  # noqa: E402
from tests.aiida.test_steps_integrity import (  # noqa: E402
    _pid, _tree, _verdicts_of, _write_ruling)
from tests.helpers.results_tree import (  # noqa: E402
    make_family_run, slide_config, write_indent_curve)
from tests.postprocessing.test_integrity import _stack, _write_data  # noqa: E402
from tests.postprocessing.test_read_data_integrity import (  # noqa: E402
    DECK, _afm_snapshots)

DEFAULT_OPTIONS = {'skip_fraction': 0.2}


@pytest.fixture
def no_profile_switch(monkeypatch):
    import frictionsim2d.aiida as aiida_pkg
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
        assert content['options'] == DEFAULT_OPTIONS
        assert content['options_hash'] == options_hash(DEFAULT_OPTIONS)
        assert summary.base.extras.get('friction2d_kind') == 'summary'
        assert summary.base.extras.get('friction2d_options_hash') == \
            options_hash(DEFAULT_OPTIONS)

    assert postprocess_group('camp') == 0                 # idempotent


#: Numeric columns of a `summary_rows` row that are the point's identity or
#: tokens, or derived by the row itself (`_LF_PAIRS`), not by `summarise_run`.
_ROW_ONLY = {'load', 'angle', 'speed', 'size_x', 'size_y', 'temperature', 'layers',
             'tip_radius', 'mean_lf_sx_sy', 'mean_lf_fx_fy'}
#: What the store summarises and the DataReader does not: it never loads the
#: retract tier (`summary_rows` row `tiers` is `window` for an indent point),
#: while ingest pairs `indent_f*N_l*.csv` with its point as `retract`.
_RETRACT = {'n_points', 'z_span', 'f_adh_min', 'pulloff_force', 'snapoff_z',
            'work_of_adhesion_nnA', 'work_of_adhesion', 'contact_stiffness', 'detached'}


def _numeric(value):
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def _assert_parity(points, root, family):
    """Every numeric key of every `summary_rows` row equals the store's
    summary (nan in the row = absent or None in the store), and the store has
    no number the row lacks (beyond the retract features)."""
    postprocess_group('camp')
    rows = {r['load']: r for r in DataReader(str(root)).summary_rows()}
    assert rows and set(rows) == set(points)
    for load, point in points.items():
        row = rows[load]
        assert row['status'] == 'complete' and row['family'] == family
        (summary,) = _summaries(point)
        content = summary.get_dict()
        compared = 0
        for key, value in row.items():
            if not _numeric(value) or key in _ROW_ONLY:
                continue
            if value != value:                                  # nan
                assert content.get(key) is None, key
            else:
                assert content[key] == pytest.approx(value), key
                compared += 1
        assert compared >= 15
        extra = {k for k, v in content.items() if _numeric(v)} - set(row)
        assert extra == (_RETRACT if family == 'indent' else set())
        assert summary['time_series_hash'] == row['time_series_hash']
        assert set(content['units']) == {k for k, v in content.items()
                                         if v is None or _numeric(v)}


def test_summaries_match_the_datareader(aiida_profile_clean, no_profile_switch, tmp_path):
    """The one COF: the store's summary is `postprocess read`'s summary row."""
    _tree(tmp_path)
    points = _points(import_root(tmp_path, label='camp'))
    _assert_parity(points, tmp_path, 'afm')
    (summary,) = _summaries(points[30.0])
    assert summary['units']['deflect_max'] == 'angstrom'       # the post channel


def test_indent_summary_matches_the_datareader(aiida_profile_clean, no_profile_switch,
                                               tmp_path):
    config = {'general': {'temp': 300, 'force': [30.0], 'scan_angle': 0.0,
                          'scan_speed': [2.0]}}
    run_dir = make_family_run(tmp_path, 'indent', loads=(30.0,), n=50, config=config)
    (run_dir / 'lammps' / 'indent.in').write_text(DECK, encoding='utf-8')
    _afm_snapshots(run_dir, 30.0)
    curve = write_indent_curve(run_dir, load=30.0, layers=2)
    points = _points(import_root(tmp_path, label='camp'))
    _assert_parity(points, tmp_path, 'indent')
    (summary,) = _summaries(points[30.0])
    import pandas as pd

    from frictionsim2d.postprocessing.indent import retract_features
    expected = retract_features(pd.read_csv(curve))
    for key, value in expected.items():
        assert summary[key] == (None if value != value else pytest.approx(value)), key
    assert summary['units']['pulloff_force'] == 'nN'


def test_sheet_summary_matches_the_datareader(aiida_profile_clean, no_profile_switch,
                                              tmp_path):
    run_dir = make_family_run(tmp_path, 'sheetonsheet', loads=(1.0,), n=50,
                              tiers=('window', 'raw'),
                              config=slide_config('sheetonsheet', loads=(1.0,)))
    (run_dir / 'lammps' / 'slide.in').write_text(
        'write_data ./data/final_p$(v_p)GPa_a$(v_a).data\n', encoding='utf-8')
    stack, box = _stack(2)
    _write_data(run_dir / 'build' / 'sheet.lmp', stack, box)
    _write_data(run_dir / 'data' / 'final_p1GPa_a0.data', stack, box)
    points = _points(import_root(tmp_path, label='camp'))
    _assert_parity(points, tmp_path, 'sheetonsheet')
    (summary,) = _summaries(points[1.0])
    assert summary['units']['slip_amplitude_mean'] == 'nN'


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
                        lambda warnings=None, settings_file=None: {**current,
                                                                   'version': '1.1.0'})

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

    again = []
    assert postprocess_group('camp', warnings=again) == 0
    assert not [w for w in again if 'provisional' in w]       # nothing summarised


def test_a_new_skip_fraction_makes_new_summaries_and_datasets_pick_theirs(
        aiida_profile_clean, no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    other = {'skip_fraction': 0.5}
    assert postprocess_group('camp') == 2
    assert postprocess_group('camp', options=other) == 2
    assert postprocess_group('camp', options=other) == 0
    assert postprocess_group('camp') == 0
    every = _summaries()
    assert sorted(s['skip_fraction'] for s in every) == [0.2, 0.2, 0.5, 0.5]
    assert not any(s.base.extras.get('superseded', False) for s in every)

    default = build_dataset('d02', ['camp'])
    half = build_dataset('d05', ['camp'], options=other)
    by_uuid = {s.uuid: s for s in every}
    assert {by_uuid[r['summary_uuid']]['skip_fraction'] for r in default['rows']} == {0.2}
    assert {by_uuid[r['summary_uuid']]['skip_fraction'] for r in half['rows']} == {0.5}
    assert {r['skip_fraction'] for r in half['rows']} == {0.5}
    assert {r['options_hash'] for r in half['rows']} == {options_hash(other)}
    assert half['policy'] == {'verdict': 'newest', 'summary': 'newest',
                              'summary_options': other,
                              'options_hash': options_hash(other)}
    assert build_dataset('none', ['camp'], options={'skip_fraction': 0.3})['rows'] == []


def test_settings_skip_fraction_is_read_strictly(aiida_profile_clean, no_profile_switch,
                                                 tmp_path, monkeypatch, caplog):
    config = tmp_path / 'xdg'
    (config / 'FrictionSim2D').mkdir(parents=True)
    settings = config / 'FrictionSim2D' / 'settings.yaml'
    monkeypatch.setenv('XDG_CONFIG_HOME', str(config))

    settings.write_text('aiida:\n  skip_fraction: 0.4\n', encoding='utf-8')
    warnings = []
    assert summary_options(warnings) == {'skip_fraction': 0.4} and warnings == []

    settings.write_text('aiida: [skip_fraction: 0.4\n', encoding='utf-8')   # broken YAML
    assert summary_options(warnings) == {'skip_fraction': 0.2}
    (line,) = warnings
    assert str(settings) in line and 'default skip_fraction 0.2' in line

    _tree(tmp_path / 'sim')
    import_root(tmp_path / 'sim', label='camp')
    warnings = []
    assert postprocess_group('camp', warnings=warnings) == 3
    assert sum('default skip_fraction' in w for w in warnings) == 1
    assert sum('default thresholds' in w for w in warnings) == 1
    assert {s['skip_fraction'] for s in _summaries()} == {0.2}

    # build_dataset says so too: into its list, or as a logged WARNING.
    warnings = []
    table = build_dataset('d', ['camp'], warnings=warnings)
    assert sum('default skip_fraction' in w for w in warnings) == 1
    assert table['policy']['summary_options'] == {'skip_fraction': 0.2}
    with caplog.at_level(logging.WARNING, logger='frictionsim2d.aiida.steps'):
        build_dataset('d2', ['camp'])
    assert any('default skip_fraction' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


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


def test_options_are_normalised_before_hashing():
    assert options_hash({'skip_fraction': 1}) == options_hash({'skip_fraction': 1.0})
    assert normalise_options({'skip_fraction': 1}) == {'skip_fraction': 1.0}
    assert isinstance(normalise_options({'skip_fraction': 1})['skip_fraction'], float)
    assert options_hash({'skip_fraction': 0.2}) != options_hash({'skip_fraction': 0.3})
    with pytest.raises(ValueError, match='unknown.*bogus'):
        normalise_options({'skip_fraction': 0.2, 'bogus': 1})
    with pytest.raises(ValueError, match='lack'):
        normalise_options({})


def test_int_and_float_options_are_one_summary(aiida_profile_clean, no_profile_switch,
                                               tmp_path):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    assert postprocess_group('camp', options={'skip_fraction': 0.5}) == 2
    assert postprocess_group('camp', options={'skip_fraction': 0.5}) == 0
    assert postprocess_group('camp', options={'skip_fraction': 0}) == 2
    assert postprocess_group('camp', options={'skip_fraction': 0.0}) == 0
    assert len(build_dataset('one', ['camp'], options={'skip_fraction': 0.0})['rows']) == 2
    with pytest.raises(ValueError, match='bogus'):
        postprocess_group('camp', options={'skip_fraction': 0.5, 'bogus': True})
    with pytest.raises(ValueError, match='bogus'):
        build_dataset('bad', ['camp'], options={'skip_fraction': 0.5, 'bogus': True})


# --- datasets -----------------------------------------------------------------

def test_build_dataset_groups_summaries_and_one_dict(aiida_profile_clean,
                                                     no_profile_switch, tmp_path):
    _tree(tmp_path, eject=(10.0,))
    points = _points(import_root(tmp_path, label='camp'))
    postprocess_group('camp')

    table = build_dataset('d', ['camp'])

    group = orm.Group.collection.get(label=f'{DATASET_PREFIX}d')
    nodes = list(group.nodes)
    summaries = [n for n in nodes if n.base.extras.get('friction2d_kind') == 'summary']
    dicts = [n for n in nodes if n.base.extras.get('friction2d_kind') == 'dataset']
    assert len(summaries) == 2 and len(nodes) == 3
    assert dicts == [table] or [d.uuid for d in dicts] == [table.uuid]
    assert table['policy'] == {'verdict': 'newest', 'summary': 'newest',
                               'summary_options': DEFAULT_OPTIONS,
                               'options_hash': options_hash(DEFAULT_OPTIONS)}
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
        assert row['skip_fraction'] == 0.2
        assert row['options_hash'] == options_hash(DEFAULT_OPTIONS)
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
    assert {k: pinned['policy'][k] for k in ('verdict', 'summary')} == {
        'verdict': 'newest', 'summary': old}
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

    with pytest.raises(ValueError, match=re.escape("delete_dataset('paper')")):
        delete_simulation_set('camp')

    assert orm.QueryBuilder().append(orm.Node).count() == before
    assert orm.Group.collection.get(label='friction2d/camp')

    with pytest.raises(ValueError, match="no dataset 'nope'"):
        delete_dataset('nope')
    assert delete_dataset('paper') == (1, 0)
    assert orm.QueryBuilder().append(orm.Node).count() == before - 1
    assert len(_summaries()) == 2                           # the set still owns them
    assert delete_simulation_set('camp') > 0
    assert _summaries() == []


def test_delete_dataset_counts_and_warns_of_the_analyses_it_takes(
        aiida_profile_clean, no_profile_switch, tmp_path, caplog):
    import pandas as pd
    pytest.importorskip('scipy')
    from frictionsim2d.analysis.base import get_analysis
    from frictionsim2d.analysis.store import ANALYSIS_GROUP_PREFIX, store_result
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')
    build_dataset('paper', ['camp'])
    table = pd.DataFrame([{'material': 'h_MoS2', 'layer': 2, 'period_x_top': 3.16}])
    uuids = [store_result(get_analysis('sheet_geometry'), 'paper', {}, table, dropped=[])
             for _ in range(2)]

    with caplog.at_level('WARNING'):
        deleted = delete_dataset('paper')

    assert deleted.tables == 1 and deleted.analyses == 2
    assert "2 analysis results" in caplog.text and 'paper' in caplog.text
    for uuid in uuids:
        with pytest.raises(exceptions.NotExistent):
            orm.load_node(uuid)
    assert not orm.QueryBuilder().append(
        orm.Group, filters={'label': f'{ANALYSIS_GROUP_PREFIX}paper'}).count()


def test_a_failed_dataset_store_leaves_nothing(aiida_profile_clean, no_profile_switch,
                                               tmp_path, monkeypatch):
    _tree(tmp_path, eject=(10.0,))
    import_root(tmp_path, label='camp')
    postprocess_group('camp')
    before = orm.QueryBuilder().append(orm.Node).count()

    def boom(self, *args, **kwargs):
        raise RuntimeError('store failed')
    monkeypatch.setattr(orm.Group, 'add_nodes', boom)
    with pytest.raises(RuntimeError, match='store failed'):
        build_dataset('paper', ['camp'])

    assert orm.QueryBuilder().append(orm.Node).count() == before
    assert not orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': f'{DATASET_PREFIX}%'}}).count()


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


def test_an_explicit_settings_file_beats_the_xdg_one(aiida_profile_clean, no_profile_switch,
                                                     tmp_path, monkeypatch):
    """`settings_file=` is read instead of the per-user file, for the summary
    options, the integrity thresholds of the import and the dataset."""
    config = tmp_path / 'xdg'
    (config / 'FrictionSim2D').mkdir(parents=True)
    (config / 'FrictionSim2D' / 'settings.yaml').write_text(
        'aiida:\n  skip_fraction: 0.4\n', encoding='utf-8')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(config))
    explicit = tmp_path / 'run-settings.yaml'
    explicit.write_text('aiida:\n  skip_fraction: 0.3\n', encoding='utf-8')

    assert summary_options() == {'skip_fraction': 0.4}
    assert summary_options(settings_file=explicit) == {'skip_fraction': 0.3}

    _tree(tmp_path / 'sim')
    assert import_root(tmp_path / 'sim', label='camp', settings_file=explicit).ok()
    assert postprocess_group('camp', settings_file=explicit) == 3
    assert {s['skip_fraction'] for s in _summaries()} == {0.3}
    table = build_dataset('d', ['camp'], settings_file=explicit)
    assert table['policy']['summary_options'] == {'skip_fraction': 0.3}
    assert len(table['rows']) == 3
