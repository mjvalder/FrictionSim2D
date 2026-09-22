"""Ingest is idempotent and set-scoped (spec §4.2)."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

from src.aiida.ingest import (  # noqa: E402
    LoadPoint, find_or_create_run, find_or_create_set, group_label,
    load_points, results_exist,
)
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import PES_GRID_CSV, make_run  # noqa: E402

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def _root(tmp_path, name='simulation_a'):
    root = tmp_path / name
    make_run(root, REL, 'afm')
    return root


@dataclass
class _Point:
    """A minimal duck-typed stand-in for `LoadPoint` (Task 15, now landed).

    `find_results`/`results_exist` only read `load_kind`, `load`, `angle`,
    `speed`, `layers` and `content_sha256` off their `point` argument, so a
    plain dataclass with those fields exercises them without needing the
    real node type. `test_results_exist_matches_conditions_and_bytes` (R10)
    was migrated to the real `LoadPoint`; this stand-in stays for
    `test_find_results_matches_layers_none_without_a_query_filter` (R4),
    which only needs the same five fields.
    """
    load_kind: str
    load: float
    angle: float
    speed: float
    layers: object
    content_sha256: str


def test_set_is_created_once_per_label(aiida_profile_clean, tmp_path):
    root = _root(tmp_path)

    first = find_or_create_set(root, '260910-afm')
    second = find_or_create_set(root, '260910-afm')

    assert first.uuid == second.uuid
    assert first.is_stored
    assert first.label == '260910-afm'


def test_set_members_land_in_the_group(aiida_profile_clean, tmp_path):
    from aiida.orm import Group, QueryBuilder

    root = _root(tmp_path)
    set_node = find_or_create_set(root, '260910-afm')
    loc, = discover_runs(root)
    run_node, created = find_or_create_run(set_node, loc)

    group = Group.collection.get(label=group_label('260910-afm'))

    assert created is True
    assert {n.uuid for n in group.nodes} == {set_node.uuid, run_node.uuid}
    assert QueryBuilder().append(Group, filters={'label': 'friction2d/260910-afm'}).count() == 1


def test_run_is_keyed_on_set_uuid_and_rel_path(aiida_profile_clean, tmp_path):
    root = _root(tmp_path)
    set_node = find_or_create_set(root, '260910-afm')
    loc, = discover_runs(root)

    first, created_first = find_or_create_run(set_node, loc)
    second, created_second = find_or_create_run(set_node, loc)

    assert created_first is True
    assert created_second is False
    assert first.uuid == second.uuid
    assert second.rel_path == REL
    assert second.set_uuid == str(set_node.uuid)


def test_the_same_rel_path_in_another_set_is_a_different_run(aiida_profile_clean, tmp_path):
    root_a = _root(tmp_path, 'simulation_a')
    root_b = _root(tmp_path, 'simulation_b')
    loc_a, = discover_runs(root_a)
    loc_b, = discover_runs(root_b)

    run_a, _ = find_or_create_run(find_or_create_set(root_a, 'a'), loc_a)
    run_b, _ = find_or_create_run(find_or_create_set(root_b, 'b'), loc_b)

    assert run_a.uuid != run_b.uuid
    assert run_a.rel_path == run_b.rel_path


def test_results_exist_matches_conditions_and_bytes(aiida_profile_clean, tmp_path):
    from src.aiida.data import FrictionResultsData
    from src.aiida.ingest import find_results, supersede

    root = _root(tmp_path)
    loc, = discover_runs(root)
    run_node, _ = find_or_create_run(find_or_create_set(root, 'a'), loc)
    point = LoadPoint(family='afm', load_kind='force', load=30.0, angle=0.0,
                      speed=2.0, layers=2, tiers={}, aux=(),
                      content_sha256='deadbeef')

    assert find_results(run_node, point) is None
    assert results_exist(run_node, point) is False

    node = FrictionResultsData()
    node.family = 'afm'
    node.load_kind = 'force'
    node.load = 30.0
    node.angle = 0.0
    node.speed = 2.0
    node.layers = 2
    node.run_uuid = str(run_node.uuid)
    node.content_sha256 = 'deadbeef'
    node.store()

    assert results_exist(run_node, point) is True

    # same conditions, different bytes: a re-run, not a duplicate
    rerun = replace(point, content_sha256='cafe')
    assert find_results(run_node, rerun) is not None
    assert results_exist(run_node, rerun) is False

    supersede(node, run_node)
    assert node.base.extras.get('superseded') is True
    assert node.base.extras.get('superseded_at').startswith('20')

    # R9: a superseded node must never be returned again -- otherwise a
    # re-ingest would compare against, or supersede, a stale node again.
    assert find_results(run_node, point) is None
    assert results_exist(run_node, point) is False


def test_find_results_matches_layers_none_without_a_query_filter(aiida_profile_clean, tmp_path):
    """R4: `{'attributes.layers': None}` raises TypeError on SQLite, so a
    sheet-family point (`layers is None`) must be matched in Python instead
    of via the QueryBuilder filter."""
    from src.aiida.data import FrictionResultsData
    from src.aiida.ingest import find_results

    root = _root(tmp_path)
    loc, = discover_runs(root)
    run_node, _ = find_or_create_run(find_or_create_set(root, 'a'), loc)
    point = _Point(load_kind='pressure', load=1.0, angle=0.0, speed=1.0,
                   layers=None, content_sha256='beef')

    assert find_results(run_node, point) is None
    assert results_exist(run_node, point) is False

    node = FrictionResultsData()
    node.family = 'sheetonsheet'
    node.load_kind = 'pressure'
    node.load = 1.0
    node.angle = 0.0
    node.speed = 1.0
    node.run_uuid = str(run_node.uuid)
    node.content_sha256 = 'beef'
    node.store()
    assert node.layers is None

    found = find_results(run_node, point)
    assert found is not None and found.uuid == node.uuid
    assert results_exist(run_node, point) is True


def test_a_second_tree_under_one_label_is_refused(aiida_profile_clean, tmp_path):
    from src.aiida.ingest import SetRootMismatch

    root_a = _root(tmp_path, 'simulation_a')
    root_b = _root(tmp_path, 'simulation_b')
    find_or_create_set(root_a, 'shared')

    with pytest.raises(SetRootMismatch, match='different tree'):
        find_or_create_set(root_b, 'shared')

    # re-ingesting the SAME tree is the idempotent path and stays allowed
    assert find_or_create_set(root_a, 'shared').source_root == str(root_a.resolve())


def test_slide_points_group_the_three_tiers(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0),
             speeds=(2.0,), tiers=('window', 'raw'))
    loc, = discover_runs(root)

    points = load_points(loc)

    assert len(points) == 4
    assert all(isinstance(p, LoadPoint) for p in points)
    assert sorted((p.load, p.angle) for p in points) == [
        (10.0, 0.0), (10.0, 90.0), (30.0, 0.0), (30.0, 90.0)]
    one = points[0]
    assert one.family == 'afm'
    assert one.load_kind == 'force'
    assert one.speed == 2.0
    assert one.layers == 2
    assert set(one.tiers) == {'window', 'raw'}
    assert len(one.content_sha256) == 64


def test_sheet_points_are_pressures(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    make_run(root, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet',
             loads=(1.0,), angles=(0.0,), speeds=(2.0,))
    loc, = discover_runs(root)

    point, = load_points(loc)

    assert point.load_kind == 'pressure'
    assert point.load == 1.0
    assert point.layers is None


def test_pes_grid_is_one_point(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_run(root, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet',
                       tiers=())
    (run_dir / 'results' / 'pes_scan.csv').write_text(
        PES_GRID_CSV['pes_sheet'], encoding='utf-8')
    loc, = discover_runs(root)

    point, = load_points(loc)

    assert point.family == 'pes_sheet'
    assert point.load_kind == 'none'
    assert point.load == 0.0
    assert point.tiers['window'].name == 'pes_scan.csv'


def test_indent_pairs_its_retract_curve_with_the_hold(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_run(root, 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1',
                       'indent', loads=(30.0,))
    (run_dir / 'results' / 'indent_f30N_l1.csv').write_text(
        'step,z_tip,fz_tip,fx_tip,fy_tip,pe\n0,0,5.0,0,0,-1\n1,1,-2.0,0,0,-1\n',
        encoding='utf-8')
    loc, = discover_runs(root)

    point, = load_points(loc)

    assert point.family == 'indent'
    assert point.load == 30.0
    assert point.load_kind == 'force'
    assert point.tiers['window'].name.startswith('friction_f30')
    assert point.tiers['retract'].name == 'indent_f30N_l1.csv'


def test_aux_files_travel_with_the_point(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'data' / 'final_f30.data').write_text('x', encoding='utf-8')
    (run_dir / 'log_slide.lammps').write_text('LAMMPS (1 Jan 2026)\n', encoding='utf-8')
    loc, = discover_runs(root)

    point, = load_points(loc)

    names = {Path(p).name for p in point.aux}
    assert names == {'final_f30.data', 'log_slide.lammps'}


def test_a_run_with_no_results_yields_no_points(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', tiers=())
    loc, = discover_runs(root)

    assert load_points(loc) == []


#: (family, rel_path, expected load_kind). Every entry of FAMILIES is covered --
#: U1 requires all eight to round-trip, and load_points is the only place the
#: families differ, so this is where the coverage has to be.
FAMILY_CASES = [
    ('afm', 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2', 'force'),
    ('indent', 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'force'),
    ('pes_tip', 'pes_tip/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'none'),
    ('sheetonsheet', 'sheetonsheet/h-MoS2/100x_100y/K300', 'pressure'),
    ('hetero', 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300', 'pressure'),
    ('flake', 'sheetonsheet/h-MoS2+flake-h-WS2/100x_100y/K300', 'pressure'),
    ('pes_sheet', 'pes_sheet/h-MoS2/100x_100y/K300', 'none'),
]


@pytest.mark.parametrize('family, rel, load_kind', FAMILY_CASES)
def test_every_family_yields_a_load_point(aiida_profile_clean, tmp_path,
                                          family, rel, load_kind):
    from src.core.observables import FAMILIES

    assert family in FAMILIES
    root = tmp_path / 'sim'
    run_dir = make_run(root, rel, family,
                       tiers=() if family in PES_GRID_CSV else ('window',))
    if family in PES_GRID_CSV:
        (run_dir / 'results' / 'pes_scan.csv').write_text(
            PES_GRID_CSV[family], encoding='utf-8')
    loc, = discover_runs(root)

    point, = load_points(loc)

    assert point.family == family
    assert point.load_kind == load_kind
    assert 'window' in point.tiers


def test_tip_only_pes_descriptors_are_produced(aiida_profile_clean, tmp_path):
    """pes_fmax / pes_flat_rms / pes_load_mean / pes_load_std exist only for the
    tip scan, and only if the fixture uses the tip template's real columns."""
    import pandas as pd

    from src.data.models import summarise_run

    root = tmp_path / 'sim'
    rel = 'pes_tip/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1'
    run_dir = make_run(root, rel, 'pes_tip', tiers=())
    (run_dir / 'results' / 'pes_scan.csv').write_text(
        PES_GRID_CSV['pes_tip'], encoding='utf-8')
    loc, = discover_runs(root)
    point, = load_points(loc)

    grid = pd.read_csv(point.tiers['window'], comment='#')
    out = summarise_run({'window': grid}, 'pes_tip')

    assert list(grid.columns) == ['x', 'y', 'energy_eV', 'fx', 'fy', 'fz']
    for key in ('pes_fmax', 'pes_flat_rms', 'pes_load_mean', 'pes_load_std'):
        assert key in out.extras, key
    assert out.extras['pes_load_mean'] == pytest.approx(-2.05)


def test_families_tuple_is_fully_covered():
    from src.core.observables import FAMILIES

    assert {f for f, _rel, _kind in FAMILY_CASES} == set(FAMILIES)
