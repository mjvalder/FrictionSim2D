"""Ingest is idempotent and set-scoped (spec §4.2)."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from dataclasses import dataclass, replace  # noqa: E402

from src.aiida.ingest import (  # noqa: E402
    find_or_create_run, find_or_create_set, group_label, results_exist,
)
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def _root(tmp_path, name='simulation_a'):
    root = tmp_path / name
    make_run(root, REL, 'afm')
    return root


@dataclass
class _Point:
    """A minimal duck-typed stand-in for `LoadPoint` (Task 15).

    `find_results`/`results_exist` only read `load_kind`, `load`, `angle`,
    `speed`, `layers` and `content_sha256` off their `point` argument, so a
    plain dataclass with those fields exercises them without depending on a
    node type this task does not create. `test_results_exist_matches_
    conditions_and_bytes` (the brief's version, which imports the real
    `LoadPoint`) moves to Task 15 -- see task-14-report.md.
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
    point = _Point(load_kind='force', load=30.0, angle=0.0, speed=2.0,
                   layers=2, content_sha256='deadbeef')

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
