"""The query layer over the CalcJob graph: `query`, `point_table`, `related`,
`latest` (plan Task 8; critique C1a, C1b, I8).

`related(point, vary)` is point-level and field-level: two runs relate when
their points agree on every requested field except those under `vary`,
whatever deck shape (array or combined mode) carried them.
"""
import json

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida import query as query_mod  # noqa: E402
from src.aiida.calcjob import import_run  # noqa: E402
from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.query import (Friction2DDB, QueryResult, latest,  # noqa: E402
                             point_table, related)
from src.aiida.steps import latest_verdict, postprocess_group  # noqa: E402
from src.aiida.store import TOKENS_EXTRA, machine_computer, reindex  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import seal_config_hash  # noqa: E402
from tests.aiida.test_calcjob_import import _afm_run, _set_record  # noqa: E402

A_UUID = '0b7c2f5e-1d3a-4c8e-9f60-2a4b6c8d0e1f'
B_UUID = '7a1d9c3e-2f44-4b58-8d3b-0c9e5f1a6b22'
C_UUID = '3c9e1a7b-5d2f-4e60-8a1b-9c0d2e4f6a8b'
TIMESTEP = 'integration.settings.simulation.timestep'
OPTIONS = {'skip_fraction': 0.2}


@pytest.fixture
def computer(aiida_profile_clean):
    return machine_computer('localhost', {})


@pytest.fixture
def no_profile_switch(monkeypatch):
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _campaign(root, run_uuid, computer, *, loads, timestep=None, array=False,
              seeds=None, label=None):
    """One AFM run imported as a set: `loads` swept, in array mode (one
    `slide_f<L>N` deck per load) or combined mode (one `slide` deck).
    Returns load -> point node."""
    loc = _afm_run(root, loads=loads, written=loads)
    _set_record(loc, run_uuid=run_uuid, **(seeds or {}))
    if timestep is not None:
        config = loc.run_dir.parent / 'provenance' / 'config.json'
        data = json.loads(config.read_text(encoding='utf-8'))
        data.setdefault('settings', {}).setdefault('simulation', {})['timestep'] = timestep
        config.write_text(json.dumps(data), encoding='utf-8')
        seal_config_hash(loc.run_dir, config.parent)
    if array:
        (loc.lammps_dir / 'slide.in').unlink()
        for load in loads:
            (loc.lammps_dir / f'slide_f{load:g}N.in').write_text('# f\n', encoding='utf-8')
    (loc,) = [r for r in discover_runs(root) if r.run_dir == loc.run_dir]
    nodes = import_run(loc, computer=computer)
    group = orm.Group(label=f'friction2d/{label or root.name}').store()
    group.add_nodes(nodes)
    return {p.base.attributes.get('load'): p
            for n in nodes if 'points' in n.outputs for p in n.outputs.points.values()}


@pytest.fixture
def campaigns(computer, tmp_path):
    """Review Focus 2: A in array mode at 10/20/30 N, timestep 0.001 (the
    default, so elided); B in combined mode at 10/20/30/40 N, timestep 0.0005."""
    a = _campaign(tmp_path / 'A', A_UUID, computer, loads=(10.0, 20.0, 30.0),
                  timestep=0.001, array=True)
    b = _campaign(tmp_path / 'B', B_UUID, computer, loads=(10.0, 20.0, 30.0, 40.0),
                  timestep=0.0005)
    return a, b


def _uuids(nodes):
    return [n.uuid for n in nodes]


# --- related: Review Focus 2 ----------------------------------------------------

def test_deck_shapes_are_what_the_campaigns_say(campaigns):
    a, b = campaigns
    assert {p.base.attributes.get('deck_name') for p in a.values()} == {
        'slide_f10N', 'slide_f20N', 'slide_f30N'}
    assert {p.base.attributes.get('deck_name') for p in b.values()} == {'slide'}


def test_related_along_one_field_crosses_deck_shapes(campaigns):
    a, b = campaigns
    assert _uuids(related(a[10.0], 'settings.simulation.timestep')) == [b[10.0].uuid]
    assert _uuids(related(a[10.0], TIMESTEP)) == [b[10.0].uuid]


def test_related_along_a_block_kind(campaigns):
    a, b = campaigns
    assert _uuids(related(a[10.0], 'integration')) == [b[10.0].uuid]
    assert related(a[10.0], 'thermostat') == []


def test_related_along_a_swept_axis(campaigns):
    a, _b = campaigns
    assert _uuids(related(a[10.0], 'load')) == [a[20.0].uuid, a[30.0].uuid]


def test_related_along_two_axes_includes_the_extra_load(campaigns):
    a, b = campaigns
    found = set(_uuids(related(a[10.0], ('load', 'integration'))))
    assert b[40.0].uuid in found
    assert found == {a[20.0].uuid, a[30.0].uuid, *(p.uuid for p in b.values())}


def test_related_takes_a_point_uuid_and_symmetric(campaigns):
    a, b = campaigns
    puuid = b[30.0].base.attributes.get('point_uuid')
    assert _uuids(related(puuid, 'integration')) == [a[30.0].uuid]


def test_related_excludes_superseded_points(campaigns):
    a, b = campaigns
    b[10.0].base.extras.set('superseded', True)
    assert related(a[10.0], 'integration') == []
    assert b[10.0].uuid in set(point_table(include_superseded=True)['node_uuid'])
    assert b[10.0].uuid not in set(point_table()['node_uuid'])


def test_related_rejects_an_unknown_vary(campaigns):
    a, _b = campaigns
    with pytest.raises(ValueError, match='no_such'):
        related(a[10.0], 'settings.simulation.no_such')


# --- related: realization -------------------------------------------------------

def test_replicates_relate_only_when_realization_is_named(computer, tmp_path):
    p = _campaign(tmp_path / 'r1', A_UUID, computer, loads=(10.0,),
                  seeds={'seed_thermal_tip': 111, 'seed_velocity': 5})[10.0]
    q = _campaign(tmp_path / 'r2', C_UUID, computer, loads=(10.0,),
                  seeds={'seed_thermal_tip': 222, 'seed_velocity': 6})[10.0]
    table = point_table()
    assert {c for c in table.columns if c.startswith('realization.')} >= {
        'realization.seed_thermal_tip', 'realization.seed_velocity'}

    assert related(p, 'thermostat', table=table) == []
    assert _uuids(related(p, ('thermostat', 'realization'), table=table)) == [q.uuid]
    assert _uuids(related(p, 'realization', table=table)) == [q.uuid]


def test_seeds_do_not_block_a_sibling_along_another_field(computer, tmp_path):
    """Seeds differ between every pair of AFM runs: comparing them would relate
    nothing, so a timestep sibling with other seeds still relates."""
    p = _campaign(tmp_path / 's1', A_UUID, computer, loads=(10.0,),
                  seeds={'seed_thermal_tip': 111})[10.0]
    q = _campaign(tmp_path / 's2', C_UUID, computer, loads=(10.0,), timestep=0.0005,
                  seeds={'seed_thermal_tip': 222})[10.0]
    assert _uuids(related(p, 'integration')) == [q.uuid]


# --- point_table ----------------------------------------------------------------

def test_point_table_columns_are_flat_field_paths(campaigns):
    a, b = campaigns
    table = point_table().set_index('node_uuid')
    assert {'load', 'angle', 'speed', 'point_uuid', 'calc_uuid'} <= set(table.columns)
    # Elided in A (the default), written in B; refilled from the schema.
    assert table.loc[a[10.0].uuid, TIMESTEP] == 0.001
    assert table.loc[b[10.0].uuid, TIMESTEP] == 0.0005
    assert table.loc[a[10.0].uuid, 'thermostat.general.temp'] == 300.0
    assert 'material__sheet_0.name' in table.columns
    assert len(table) == 7


def test_point_table_is_bulk(campaigns, monkeypatch):
    calls = []
    real_all = orm.QueryBuilder.all
    real_init = orm.QueryBuilder.__init__

    def counting_all(self, *args, **kwargs):
        calls.append('all')
        return real_all(self, *args, **kwargs)

    built = []

    def counting_init(self, *args, **kwargs):
        built.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(orm.QueryBuilder, 'all', counting_all)
    monkeypatch.setattr(orm.QueryBuilder, '__init__', counting_init)
    table = point_table()
    assert len(table) == 7
    assert len(calls) <= 3, calls
    assert len(built) <= 3, built


# --- query / to_dataframe ---------------------------------------------------------

def _checked_set(tmp_path):
    from tests.aiida.test_steps_integrity import _tree
    _tree(tmp_path, quarantine=(30.0,))
    report = import_root(tmp_path, label='camp')
    assert report.ok(), report.render()
    postprocess_group('camp', options=OPTIONS)
    return {orm.load_node(u).base.attributes.get('load'): orm.load_node(u)
            for u in report.points}


def test_dataframe_has_status_summary_and_units(aiida_profile_clean, no_profile_switch,
                                                tmp_path):
    points = _checked_set(tmp_path)
    result = Friction2DDB().query(set_label='camp', summary_options=OPTIONS)
    assert isinstance(result, QueryResult)
    df = result.to_dataframe()
    for column in ('status', 'reasons', 'mean_cof', 'load_unit', 'angle_unit',
                   'speed_unit', 'point_uuid', 'rel_path', 'model', 'set_label'):
        assert column in df.columns, column
    assert len(df) == result.total_count == 3
    rows = df.set_index('load')
    assert rows.loc[30.0, 'status'] == latest_verdict(points[30.0])['status'] == 'quarantined'
    assert rows.loc[10.0, 'status'] == 'complete'
    assert rows.loc[10.0, 'mean_cof'] == pytest.approx(
        latest(points[10.0], 'summary', options=OPTIONS)['mean_cof'])
    assert set(df['load_unit']) == {'nN'}


def test_query_excludes_superseded_points(aiida_profile_clean, no_profile_switch,
                                          tmp_path):
    points = _checked_set(tmp_path)
    (material,) = Friction2DDB().get_available_materials()
    assert Friction2DDB().query(materials=[material]).total_count == 3
    points[10.0].base.extras.set('superseded', True)
    result = Friction2DDB().query(materials=[material])
    assert result.total_count == 2
    assert points[10.0].uuid not in set(result.to_dataframe()['node_uuid'])


def test_query_filters(campaigns):
    a, b = campaigns
    db = Friction2DDB()
    assert db.query().total_count == 7
    assert db.query(set_label='A').total_count == 3
    assert db.query(force_range=(15.0, 35.0)).total_count == 4
    assert db.query(force_range=(35.0, None)).total_count == 1
    assert db.query(pressure_range=(0.0, None)).total_count == 0
    assert db.query(model='afm', family='afm', layers=2, tip_radius=25.0,
                    substrate_amorphous=True).total_count == 7
    assert db.query(materials=['h-MoS2', 'no-such']).total_count == 0
    assert db.query(temperature_range=(299.0, 301.0)).total_count == 7
    assert db.query(size_y=999.0).total_count == 0
    with db.snapshot():
        assert db.query(set_label='B').total_count == 4
        assert db.get_available_materials() == ['h-MoS2']
    limited = db.query(limit=2, order_by='-load')
    assert limited.total_count == 7
    assert list(limited.to_dataframe()['load']) == [40.0, 30.0]
    assert b[40.0].uuid in set(limited.to_dataframe()['node_uuid'])
    assert a  # both campaigns present


def test_tokens_extra_is_set_and_reindexed(campaigns):
    a, _b = campaigns
    calc = a[10.0].creator
    tokens = calc.base.extras.get(TOKENS_EXTRA)
    assert tokens['model'] == 'afm' and tokens['tip_radius'] == 25.0
    calc.base.extras.delete(TOKENS_EXTRA)
    assert reindex() >= 1
    assert calc.base.extras.get(TOKENS_EXTRA) == tokens


def test_available_conditions_materials_sets_and_statistics(campaigns):
    db = Friction2DDB()
    conditions = db.get_available_conditions()
    assert conditions['forces'] == [10.0, 20.0, 30.0, 40.0]
    assert conditions['pressures'] == []
    assert conditions['set_labels'] == ['A', 'B']
    assert conditions['families'] == ['afm'] and conditions['models'] == ['afm']
    assert db.get_available_materials() == ['h-MoS2']
    sets = {s['label']: s for s in db.list_sets()}
    assert sets['A']['n_runs'] == 1 and sets['A']['n_points'] == 3
    assert sets['B']['n_decks'] == 2
    stats = db.get_statistics()
    assert (stats['n_groups'], stats['n_decks'], stats['n_points']) == (2, 6, 7)
    assert stats['by_verdict'] == {'unchecked': 7}


def test_latest_reads_policy_selection(aiida_profile_clean, no_profile_switch, tmp_path):
    points = _checked_set(tmp_path)
    point = points[10.0]
    assert latest(point, 'verdict').uuid == latest_verdict(point).uuid
    puuid = point.base.attributes.get('point_uuid')
    assert latest(puuid, 'verdict').uuid == latest_verdict(point).uuid
    assert latest(point, 'summary', options=OPTIONS)['options_hash']
    assert latest(point, 'summary', options={'skip_fraction': 0.5}) is None
    with pytest.raises(ValueError):
        latest(point, 'nonsense')


def test_query_module_has_no_per_point_loads(campaigns, monkeypatch):
    """`query` reads every point through bulk projections, never `load_node`."""
    def refuse(*_args, **_kwargs):
        raise AssertionError('load_node called')
    monkeypatch.setattr(query_mod.orm, 'load_node', refuse)
    assert Friction2DDB().query().total_count == 7
    a, _b = campaigns
    assert len(related(a[10.0], 'load')) == 2


def test_aiida_query_cli_prints_point_rows(campaigns, no_profile_switch):
    from click.testing import CliRunner

    from src.cli import cli
    result = CliRunner().invoke(cli, ['aiida', 'query', '--set-label', 'A',
                                      '--format', 'csv'])
    assert result.exit_code == 0, result.output
    header = next(line for line in result.output.splitlines() if 'point_uuid' in line)
    assert 'load_unit' in header and 'status' in header
    assert 'Found 3 results' in result.output
    assert 'Showing' not in result.output
    limited = CliRunner().invoke(cli, ['aiida', 'query', '--limit', '2'])
    assert limited.exit_code == 0, limited.output
    assert 'Showing 2 of 7' in limited.output


# --- related over a hand-made table: tolerance, units, vary names ---------------

def _row(uuid, **fields):
    row = {'point_uuid': uuid, 'node_uuid': f'n-{uuid}', 'family': 'afm',
           'load_kind': 'force', 'load': 10.0, 'angle': 0.0, 'speed': 2.0,
           'load_unit': 'nN', 'angle_unit': 'deg', 'speed_unit': 'm/s',
           'schema_hash': 'S1', TIMESTEP: 0.001, 'thermostat.general.temp': 300.0}
    row.update(fields)
    return row


def _hand_table(rows, units=None):
    import pandas as pd
    table = pd.DataFrame(rows)
    table.attrs['units'] = units or {'S1': {}, 'S2': {}}
    return table


@pytest.fixture
def uuids_back(monkeypatch):
    """`related` returns the node uuids it would load."""
    monkeypatch.setattr(query_mod, '_nodes_by_uuid', lambda uuids: list(uuids))


def test_reals_equal_to_compare_tolerance(uuids_back):
    table = _hand_table([_row('p', **{TIMESTEP: 0.3}),
                         _row('q', **{TIMESTEP: 0.1 + 0.2}),
                         _row('r', **{TIMESTEP: 0.3, 'thermostat.general.temp': 300.3})])
    assert related('p', 'thermostat', table=table) == ['n-r']
    table = _hand_table([_row('p', **{TIMESTEP: 1.0}), _row('q', **{TIMESTEP: 1.001})])
    assert related('p', 'thermostat', table=table) == []
    assert related('p', 'integration', table=table) == ['n-q']


def test_unit_clash_on_a_compared_field_excludes(uuids_back):
    units = {'S1': {'general.temp': 'K'}, 'S2': {'general.temp': 'degC'}}
    table = _hand_table([_row('p'), _row('q', schema_hash='S2', **{TIMESTEP: 0.002})],
                        units)
    assert related('p', 'integration', table=table) == []
    # The clashing field varies: not compared, so no clash.
    assert related('p', ('integration', 'thermostat'), table=table) == ['n-q']


def test_unit_clash_on_a_field_both_leave_absent_does_not_exclude(uuids_back):
    units = {'S1': {'tip.r': 'angstrom'}, 'S2': {'tip.r': 'nm'}}
    table = _hand_table([_row('p', **{'geometry.tip.r': None}),
                         _row('q', schema_hash='S2',
                              **{'geometry.tip.r': None, TIMESTEP: 0.002})], units)
    assert related('p', 'integration', table=table) == ['n-q']


def test_axis_units_are_compared(uuids_back):
    table = _hand_table([_row('p'), _row('q', load=20.0, load_unit='GPa')])
    assert related('p', 'load', table=table) == []
    table = _hand_table([_row('p'), _row('q', load=20.0)])
    assert related('p', 'load', table=table) == ['n-q']


@pytest.mark.parametrize('name', ['family', 'rel_path', 'deck_name', 'run_uuid',
                                  'point_uuid', 'load_unit'])
def test_vary_rejects_identity_columns(uuids_back, name):
    with pytest.raises(ValueError, match='identity'):
        related('p', name, table=_hand_table([_row('p'), _row('q')]))


def test_points_without_schema_warn_once_per_run(campaigns, monkeypatch, caplog):
    real = query_mod._contents
    monkeypatch.setattr(query_mod, '_contents',
                        lambda blocks, schemas: (real(blocks, schemas)[0], {}))
    with caplog.at_level('WARNING', logger='src.aiida.query'):
        assert point_table().empty
    lines = sorted(r.getMessage() for r in caplog.records if 'left out' in r.getMessage())
    assert len(lines) == 2                       # runs A and B, not 7 points
    for line, root in zip(lines, ('A', 'B')):
        run_uuid = {'A': A_UUID, 'B': B_UUID}[root]
        assert line.startswith(f'run {run_uuid} (afm/'), line
        assert line.endswith('): points left out of the point table: no schema to '
                             'refill defaults from; run `FrictionSim2D aiida reindex`'), line


def test_snapshot_shares_one_points_projection(campaigns, monkeypatch):
    calls = []
    real = query_mod._points

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(query_mod, '_points', counting)
    db = Friction2DDB()
    with db.snapshot():
        db.query()
        db.get_available_materials()
        db.get_available_conditions()
    assert len(calls) == 1


def test_large_integers_relate_exactly(uuids_back):
    steps = 'integration.settings.simulation.slide_run_steps'
    table = _hand_table([_row('p', **{steps: 12345678}), _row('q', **{steps: 12345679}),
                         _row('r', **{steps: 12345678, 'thermostat.general.temp': 400.0})])
    assert related('p', 'thermostat', table=table) == ['n-r']
    assert related('p', 'integration', table=table) == ['n-q']


def test_point_table_keeps_integers_beside_absent_cells(campaigns):
    table = point_table()
    column = next(c for c in table.columns if c.endswith('.slide_run_steps'))
    assert {type(v) for v in table[column]} == {int}
