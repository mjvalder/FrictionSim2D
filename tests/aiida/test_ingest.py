"""Ingest is idempotent and set-scoped (spec §4.2)."""
import subprocess
import sys

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

from src.aiida.ingest import (  # noqa: E402
    IngestReport, LoadPoint, create_results, find_or_create_run,
    find_or_create_set, group_label, ingest_root, load_points, results_exist,
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


def test_pes_refine_grid_travels_as_aux_not_a_second_point(aiida_profile_clean, tmp_path):
    """`pes_scan_refine.csv` (written alongside `pes_scan.csv` when
    `pes.grid_n_refine` is set -- src/builders/sheetonsheet/pes_sheet.py
    `_grid_variants`) must not become its own `LoadPoint`: both grids would
    carry the identical identity (load_kind='none', load=angle=speed=0,
    layers=None), so two points would supersede each other on every
    ingest and neither would ever settle. It rides along as an aux file on
    the one point the base grid forms instead."""
    root = tmp_path / 'sim'
    run_dir = make_run(root, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet',
                       tiers=())
    (run_dir / 'results' / 'pes_scan_refine.csv').write_text(
        PES_GRID_CSV['pes_sheet'], encoding='utf-8')
    loc, = discover_runs(root)

    points = load_points(loc)

    assert len(points) == 1
    point = points[0]
    assert point.tiers['window'].name == 'pes_scan.csv'
    assert {Path(p).name for p in point.aux} == {'pes_scan_refine.csv'}

    # Stable across re-ingest: same single point, same hash both times.
    again, = load_points(loc)
    assert again.content_sha256 == point.content_sha256


def test_a_tier_collision_is_warned_not_silently_overwritten(
        aiida_profile_clean, tmp_path, caplog):
    """Two filenames that parse to the SAME (kind, load, angle, speed,
    layers, tier) -- e.g. a stray `.bak` copy, or `f30` alongside `f30.0` --
    must not silently overwrite one another in `tiers`; the second is
    dropped with a warning naming both files."""
    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    results_dir = run_dir / 'results'
    original = results_dir / 'friction_f30_a0_s2_layer2.txt'
    dup = results_dir / 'friction_f30.0_a0_s2_layer2.txt.bak'
    dup.write_text(original.read_text(encoding='utf-8'), encoding='utf-8')
    loc, = discover_runs(root)

    with caplog.at_level('WARNING'):
        point, = load_points(loc)

    assert original.name in caplog.text
    assert dup.name in caplog.text
    assert point.tiers['window'].name in {original.name, dup.name}


def _walk(node):
    out = []
    for dirpath, _dirs, files in node.base.repository.walk():
        out.extend((dirpath / f).as_posix() for f in files)
    return out


def test_ingest_creates_set_run_and_results(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_20260910_120000'
    make_run(root, REL, 'afm', loads=(10.0, 30.0), tiers=('window', 'raw'))

    report = ingest_root(root)

    assert isinstance(report, IngestReport)
    assert report.ok() is True
    assert report.label == 'simulation_20260910_120000'
    assert report.set_uuid
    assert report.created_runs == [REL]
    assert len(report.created_results) == 2
    assert report.skipped_results == []
    assert report.failures == []


def test_second_ingest_creates_nothing(aiida_profile_clean, tmp_path):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(10.0, 30.0))
    ingest_root(root, label='rerun')
    before = QueryBuilder().append(FrictionResultsData).count()

    second = ingest_root(root, label='rerun')

    assert QueryBuilder().append(FrictionResultsData).count() == before
    assert second.created_runs == []
    assert second.created_results == []
    assert len(second.skipped_results) == 2


def test_an_indent_run_is_idempotent_too(aiida_profile_clean, tmp_path):
    """The retract curve is in the hash, so re-ingest must still skip."""
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    rel = 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1'
    root = tmp_path / 'simulation_i'
    run_dir = make_run(root, rel, 'indent', loads=(30.0,))
    (run_dir / 'results' / 'indent_f30N_l1.csv').write_text(
        'step,z_tip,fz_tip,fx_tip,fy_tip,pe\n0,0,5.0,0,0,-1\n1,1,-2.0,0,0,-1\n',
        encoding='utf-8')
    ingest_root(root, label='ind')
    before = QueryBuilder().append(FrictionResultsData).count()

    second = ingest_root(root, label='ind')

    assert QueryBuilder().append(FrictionResultsData).count() == before
    assert second.created_results == []
    assert len(second.skipped_results) == 1


def test_a_changed_load_point_supersedes_rather_than_duplicating(
        aiida_profile_clean, tmp_path):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    root = tmp_path / 'simulation_s'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='sup')
    target = next((run_dir / 'results').glob('friction_f30*.txt'))
    target.write_text(target.read_text(encoding='utf-8') + '99000 30.0 9.9 0.0\n',
                      encoding='utf-8')

    second = ingest_root(root, label='sup')

    assert len(second.superseded_results) == 1
    assert second.created_results == []
    nodes = QueryBuilder().append(FrictionResultsData).all(flat=True)
    assert len(nodes) == 2
    assert sum(1 for n in nodes if n.base.extras.get('superseded', False)) == 1


def test_new_load_points_attach_to_the_existing_run(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm', loads=(10.0,))
    ingest_root(root, label='grow')
    make_run(root, REL, 'afm', loads=(30.0,))

    report = ingest_root(root, label='grow')

    assert report.created_runs == []
    assert len(report.created_results) == 1
    assert run_dir.is_dir()


def test_results_carry_the_summary_and_the_pointers(aiida_profile_clean, tmp_path):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(30.0,), tiers=('window', 'raw'))
    report = ingest_root(root, label='sum')

    node = QueryBuilder().append(FrictionResultsData).first(flat=True)

    assert node.set_uuid == report.set_uuid
    assert node.run_uuid
    assert node.load == 30.0
    assert node.load_kind == 'force'
    assert node.base.attributes.get('mean_cof') is not None
    assert node.base.attributes.get('slip_n_events') is not None


def test_dry_run_creates_no_nodes(aiida_profile_clean, tmp_path):
    from aiida.orm import Group, QueryBuilder

    from src.aiida.data import FrictionSimulationSetData

    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')

    report = ingest_root(root, label='plan', dry_run=True)

    assert report.dry_run is True
    assert report.created_runs == [REL]
    assert QueryBuilder().append(FrictionSimulationSetData).count() == 0
    # R(fix-1c): a fresh store must come out of a dry run with no Group
    # either -- only `find_or_create_*` ever call `_group_for`, and the dry
    # path must never reach them.
    assert QueryBuilder().append(Group).count() == 0
    assert 'DRY RUN' in report.render()


def test_dry_run_after_a_real_ingest_reports_nothing_new(aiida_profile_clean, tmp_path):
    """R(fix-1a): a dry run must classify against what is ALREADY stored, not
    unconditionally claim every point as new."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(10.0, 30.0))
    ingest_root(root, label='plan2')

    report = ingest_root(root, label='plan2', dry_run=True)

    assert report.created_runs == []
    assert report.created_results == []
    assert len(report.skipped_results) == 2
    assert report.superseded_results == []


def test_dry_run_reports_a_changed_point_as_superseded(aiida_profile_clean, tmp_path):
    """R(fix-1b): a point whose bytes changed since the last real ingest is
    planned as a supersession, not a fresh "created" or a false "skipped"."""
    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='plan3')
    target = next((run_dir / 'results').glob('friction_f30*.txt'))
    target.write_text(target.read_text(encoding='utf-8') + '99000 30.0 9.9 0.0\n',
                      encoding='utf-8')

    report = ingest_root(root, label='plan3', dry_run=True)

    assert report.superseded_results == [f'{REL}#force30_a0_s2']
    assert report.created_results == []
    assert report.skipped_results == []


def test_profile_falls_back_to_settings(monkeypatch, tmp_path):
    import src.aiida as aiida_pkg

    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    (tmp_path / 'FrictionSim2D').mkdir()
    (tmp_path / 'FrictionSim2D' / 'settings.yaml').write_text(
        'aiida:\n  profile: friction2d\n', encoding='utf-8')
    seen = []
    monkeypatch.setattr(aiida_pkg, 'LOAD_PROFILE', lambda name: seen.append(name))

    aiida_pkg.load_aiida_profile(None)
    aiida_pkg.load_aiida_profile('explicit')

    assert seen == ['friction2d', 'explicit']


def test_a_broken_run_is_reported_not_raised(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    good = REL
    bad = 'afm/h-WS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1'
    make_run(root, good, 'afm')
    make_run(root, bad, 'afm')
    (root / bad / 'results' / 'friction_f30_a0_s2_layer1.txt').write_text(
        '', encoding='utf-8')

    report = ingest_root(root, label='mixed')

    assert good in report.created_runs
    assert report.ok() is False
    # R14: a per-point failure is keyed 'rel#force30_a0_s2', not the bare rel
    assert report.failures and report.failures[0][0].startswith(bad)
    assert 'FAILED' in report.render()


def test_aux_files_are_stored_under_run_relative_names(aiida_profile_clean, tmp_path):
    """R18/T11 carry-over: `add_result_files`'s old `parent.name/name` guess
    put a direct child of the run root (the slide log) one level too deep,
    filed under the run's own leaf directory name instead of the run root."""
    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    (run_dir / 'log_slide.lammps').write_text('LAMMPS (1 Jan 2026)\n', encoding='utf-8')
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'data' / 'final_f30.data').write_text('x', encoding='utf-8')
    loc, = discover_runs(root)
    point, = load_points(loc)
    set_node = find_or_create_set(root, 'aux')
    run_node, _ = find_or_create_run(set_node, loc)

    node = create_results(run_node, point, loc.run_dir)

    names = set(_walk(node))
    assert 'log_slide.lammps' in names
    assert 'data/final_f30.data' in names
    assert any(n.startswith('results/friction_') for n in names)


def test_an_over_cap_aux_file_is_recorded_not_stored(
        aiida_profile_clean, tmp_path, monkeypatch):
    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    (run_dir / 'data').mkdir(exist_ok=True)
    big = run_dir / 'data' / 'final_f30.data'
    big.write_bytes(b'x' * (2 * 1024 * 1024))
    loc, = discover_runs(root)
    point, = load_points(loc)
    set_node = find_or_create_set(root, 'cap')
    run_node, _ = find_or_create_run(set_node, loc)
    monkeypatch.setattr('src.aiida.ingest._max_stored_file_mb', lambda: 1)

    node = create_results(run_node, point, loc.run_dir)

    assert 'data/final_f30.data' not in _walk(node)
    skipped = node.oversize_files
    assert skipped[0]['path'] == 'data/final_f30.data'
    assert skipped[0]['size'] == 2 * 1024 * 1024
    assert len(skipped[0]['sha256']) == 64


def test_frame_retract_finds_the_stored_indent_csv(aiida_profile_clean, tmp_path):
    rel = 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1'
    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, rel, 'indent', loads=(30.0,))
    (run_dir / 'results' / 'indent_f30N_l1.csv').write_text(
        'step,z_tip,fz_tip,fx_tip,fy_tip,pe\n0,0,5.0,0,0,-1\n1,1,-2.0,0,0,-1\n',
        encoding='utf-8')
    loc, = discover_runs(root)
    point, = load_points(loc)
    set_node = find_or_create_set(root, 'retract')
    run_node, _ = find_or_create_run(set_node, loc)

    node = create_results(run_node, point, loc.run_dir)

    df = node.frame('retract')
    assert list(df.columns) == ['step', 'z_tip', 'fz_tip', 'fx_tip', 'fy_tip', 'pe']
    assert len(df) == 2


def test_importing_ingest_does_not_require_aiida():
    """`src.aiida.ingest` must import without `aiida` installed: every
    `aiida.orm` call in it is deferred into a function body specifically so
    the module loads on a machine without aiida. A module-top import of
    `TIER_KEYS` from `src.aiida.data`/`src.aiida.data.results` breaks this
    -- that package eagerly imports three `aiida.orm.Data` node types.
    `src.aiida.data_keys` (a sibling of `ingest.py`, not a child of the
    `data` package) is the fix; this pins it with `aiida` actually blocked,
    not just absent from this test's own environment.

    Blocked via `sys.modules['aiida'] = None` -- the mechanism `importlib`
    itself special-cases as "not installed" (`import aiida` then raises
    `ImportError`, and `importlib.util.find_spec('aiida')`, which
    `src/__init__.py` itself probes with at import time, returns `None`
    rather than raising). A `sys.meta_path` finder that raises from
    `find_spec` was tried first and broke on exactly that probe: raising
    propagates through `find_spec()` instead of behaving like "not found",
    so `import src` itself blew up before reaching `src.aiida.ingest`.
    """
    code = (
        "import sys\n"
        "sys.modules['aiida'] = None\n"
        "import src.aiida.ingest as ingest\n"
        "assert sys.modules.get('aiida') is None\n"
        "assert hasattr(ingest, 'load_points')\n"
        "assert hasattr(ingest, 'LoadPoint')\n"
        "assert hasattr(ingest, 'TIER_KEYS')\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_list_sets_reports_the_new_shape(aiida_profile_clean, tmp_path):
    from src.aiida.integration import list_simulation_sets

    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='listed')

    rows = list_simulation_sets()

    assert len(rows) == 1
    assert rows[0]['label'] == 'listed'
    assert rows[0]['model'] == 'afm'
    assert rows[0]['n_runs'] == 1
    assert rows[0]['n_results'] == 1
    assert rows[0]['materials'] == ['h-MoS2']


def test_delete_removes_the_whole_group(aiida_profile_clean, tmp_path):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData, FrictionSimulationData
    from src.aiida.integration import delete_simulation_set

    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='doomed')

    n_deleted = delete_simulation_set('doomed')

    assert n_deleted >= 3
    assert QueryBuilder().append(FrictionSimulationData).count() == 0
    assert QueryBuilder().append(FrictionResultsData).count() == 0


def test_find_run_by_rel_path_scopes_to_the_matching_source_root(aiida_profile_clean,
                                                                  tmp_path):
    """Two sets sharing a rel_path (R27 ruling): an unscoped lookup on
    `rel_path` alone would return an arbitrary one of the two runs. Scoping
    to the set whose `source_root` matches the given `root` must return the
    run that actually belongs to that tree.
    """
    from src.aiida.ingest import find_run_by_rel_path

    root_a = tmp_path / 'simulation_a'
    root_b = tmp_path / 'simulation_b'
    make_run(root_a, REL, 'afm', loads=(30.0,))
    make_run(root_b, REL, 'afm', loads=(30.0,))

    ingest_root(root_a, label='set-a')
    ingest_root(root_b, label='set-b')

    node_a = find_run_by_rel_path(root_a / REL, root_a)
    node_b = find_run_by_rel_path(root_b / REL, root_b)

    assert node_a is not None
    assert node_b is not None
    assert node_a.uuid != node_b.uuid
    assert node_a.set_label == 'set-a'
    assert node_b.set_label == 'set-b'


def test_find_run_by_rel_path_returns_none_for_an_uningested_root(aiida_profile_clean,
                                                                   tmp_path):
    from src.aiida.ingest import find_run_by_rel_path

    root = tmp_path / 'simulation_untracked'
    make_run(root, REL, 'afm', loads=(30.0,))

    assert find_run_by_rel_path(root / REL, root) is None


def _enrich(run_dir, **updates):
    import json as _json

    prov = run_dir / 'provenance.json'
    data = _json.loads(prov.read_text(encoding='utf-8'))
    data.update(updates)
    prov.write_text(_json.dumps(data), encoding='utf-8')


def test_post_run_provenance_reaches_the_store_and_the_export(aiida_profile_clean,
                                                              tmp_path):
    """I2: `hpc enrich-provenance` rewrites provenance.json after `run
    --aiida` created the run node. The attribute is immutable, so the newer
    record rides in an extra; export, the bundle and `to_record` read it."""
    import json as _json

    from src.aiida.export import export_provenance_bundle, export_results

    root = _root(tmp_path)
    ingest_root(root, label='enrich')
    _enrich(root / REL, ramp_converged=True, ramp_r=0.012, ramp_iterations=7,
            unloaded_reference_z={'layer_1': 12.5}, lammps_version='29 Aug 2024')

    report = ingest_root(root, label='enrich')

    assert report.ok(), report.render()
    assert report.updated_provenance == [REL]
    assert 'provenance updated:  1' in report.render()
    run_node, _ = find_or_create_run(find_or_create_set(root, 'enrich'),
                                     discover_runs(root)[0])
    assert run_node.effective_provenance['ramp_r'] == 0.012
    assert run_node.effective_provenance['lammps_version'] == '29 Aug 2024'
    assert run_node.provenance.get('ramp_r') is None  # attribute unchanged

    out = tmp_path / 'exported'
    export_results('enrich', out)
    exported = _json.loads((out / REL / 'provenance.json').read_text(encoding='utf-8'))
    assert exported['ramp_converged'] is True
    assert exported['unloaded_reference_z'] == {'layer_1': 12.5}

    bundle = tmp_path / 'bundle'
    export_provenance_bundle(run_node, bundle)
    assert _json.loads((bundle / 'provenance.json').read_text(
        encoding='utf-8'))['ramp_iterations'] == 7

    again = ingest_root(root, label='enrich')
    assert again.updated_provenance == []

    _enrich(root / REL, lammps_version='2 Apr 2025')
    third = ingest_root(root, label='enrich')
    assert third.updated_provenance == [REL]
    assert run_node.effective_provenance['lammps_version'] == '2 Apr 2025'
    assert run_node.effective_provenance['ramp_r'] == 0.012


def test_dry_run_reports_a_pending_provenance_update(aiida_profile_clean, tmp_path):
    root = _root(tmp_path)
    ingest_root(root, label='enrich-dry')
    _enrich(root / REL, lammps_version='29 Aug 2024')

    plan = ingest_root(root, label='enrich-dry', dry_run=True)
    assert plan.updated_provenance == [REL]

    run_node, _ = find_or_create_run(find_or_create_set(root, 'enrich-dry'),
                                     discover_runs(root)[0])
    assert run_node.provenance_update is None  # a dry run records nothing


def _stored_run_relative(node, prefix=''):
    return {prefix + (root / name).as_posix()
            for root, _dirs, files in node.base.repository.walk() for name in files}


@pytest.mark.parametrize('family, rel', [
    ('afm', REL),
    ('sheetonsheet', 'sheetonsheet/h-MoS2/100x_100y/K300'),
    ('pes_sheet', 'pes_sheet/h-MoS2/100x_100y/K300'),
])
def test_every_retained_file_is_stored_somewhere(aiida_profile_clean, tmp_path,
                                                 family, rel):
    """I6: ingest's stored set derives from `retention.retained_paths`, so the
    import path stores what the CalcJob path retrieves -- including
    `data/load_*.data`, `*.out` and results files no load point claims."""
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData
    from src.aiida.retention import retained_paths

    root = tmp_path / 'sim'
    run_dir = make_run(root, rel, family)
    extras = {
        'data/load_30.data': 'atoms\n', 'data/hold_30.txt': '1 2\n',
        'data/final_30.data': 'atoms\n', 'lammps/run_slide.log': 'log\n',
        'slide.in.out': 'stdout\n', 'log.lammps': 'LAMMPS (x)\n',
        'results/notes_unrecognised.txt': 'hello\n',
        'visuals/traj.lammpstrj': 'not retained\n',
    }
    if family == 'pes_sheet':
        extras['results/pes_scan_refine.csv'] = '# family=pes_sheet\ndx,dy,energy_eV\n'
    for name, text in extras.items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_text(text, encoding='utf-8')

    report = ingest_root(root, label=f'retain-{family}')
    assert report.ok(), report.render()

    loc, = discover_runs(root)
    run_node, _ = find_or_create_run(find_or_create_set(root, f'retain-{family}'), loc)
    stored = _stored_run_relative(run_node)
    qb = QueryBuilder()
    qb.append(FrictionResultsData, filters={'attributes.run_uuid': str(run_node.uuid)})
    results = qb.all(flat=True)
    assert results
    for node in results:
        stored |= _stored_run_relative(node)

    resolved = run_dir.resolve()
    retained = {p.relative_to(resolved).as_posix() for p in retained_paths(run_dir)}
    assert 'data/load_30.data' in retained
    assert retained <= stored, sorted(retained - stored)
    assert 'visuals/traj.lammpstrj' not in stored

    # An unclaimed results file must not shadow the real tier on re-read.
    for node in results:
        assert node.frame('window') is not None


@pytest.mark.parametrize('rows, complete', [(50, False), (99, True), (100, True)])
def test_a_truncated_slide_trace_is_not_complete(aiida_profile_clean, tmp_path,
                                                 rows, complete):
    """I7: a walltime-killed run leaves a partial window trace whose COF is
    not the run's COF. 100 000 slide steps at results_frequency 1000 -> 100
    rows expected; one row short is normal, more is truncation."""
    import json as _json

    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', n=rows)
    (run_dir.parent / 'provenance' / 'config.json').write_text(_json.dumps({
        'general': {'temp': 300},
        'settings': {'simulation': {'slide_run_steps': 100000},
                     'output': {'results_frequency': 1000}}}), encoding='utf-8')

    report = ingest_root(root, label=f'trunc-{rows}')

    node, = QueryBuilder().append(FrictionResultsData).all(flat=True)
    assert node.ntimesteps == rows
    assert node.is_complete is complete
    attrs = node.base.attributes
    if complete:
        assert attrs.get('truncated', None) is None
        assert report.incomplete_results == []
    else:
        assert attrs.get('truncated') is True
        assert attrs.get('expected_ntimesteps') == 100
        assert len(report.incomplete_results) == 1


def test_truncation_is_not_guessed_without_the_run_settings(aiida_profile_clean,
                                                            tmp_path):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionResultsData

    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', n=5)  # config.json carries no settings block
    ingest_root(root, label='trunc-unknown')

    node, = QueryBuilder().append(FrictionResultsData).all(flat=True)
    assert node.is_complete is True
