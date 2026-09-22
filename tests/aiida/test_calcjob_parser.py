"""The CalcJob keeps what ingest keeps, and its parser produces the same nodes."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.calcjob import LammpsFrictionCalcJob  # noqa: E402
from src.aiida.ingest import ingest_retrieved, locator_for  # noqa: E402
from src.aiida.retention import RETAINED_GLOBS  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def test_calcjob_declares_the_new_surface(aiida_profile_clean):
    spec = LammpsFrictionCalcJob.spec()

    assert 'results' in spec.outputs
    assert 330 in {code.status for code in spec.exit_codes.values()}
    assert 'results_folder' not in spec.outputs
    assert 'visuals/*' not in RETAINED_GLOBS


def test_calcjob_retrieve_list_is_the_retention_rule_plus_array_status(
        aiida_profile_clean):
    """Not `retrieve_patterns() == list(RETRIEVE_LIST)` (a tautology that
    compares a function to the constant it returns): this pins what the
    CalcJob itself actually builds -- via its own `_retrieve_list_for` helper,
    the thing `prepare_for_submission` calls -- against `retrieve_patterns`,
    so a change that re-derives or forks the tuples here would fail this
    test even though Task 13's own tests never touch this module.
    """
    from src.aiida.calcjob import _retrieve_list_for  # pylint: disable=import-outside-toplevel
    from src.aiida.retention import retrieve_patterns  # pylint: disable=import-outside-toplevel

    assert _retrieve_list_for(None) == list(retrieve_patterns()) + ['array_status/*']
    assert _retrieve_list_for('build/l2') == (
        list(retrieve_patterns('build/l2')) + ['array_status/*'])


def test_sim_prefix_for_prefers_simulation_node_rel_path(aiida_profile_clean):
    """R1 (fix round): `simulation_node.rel_path` -- when the input is given
    -- must win over `_detect_sim_prefix`'s read_data-line heuristic, which
    for an AFM-family run recovers only the shared K-level ancestor of
    `build/`, one level shallower than the run's own `.../K300/L{n}` where
    `results/`/`data/`/`provenance.json` actually live.
    """
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    from src.aiida.calcjob import _sim_prefix_for  # pylint: disable=import-outside-toplevel

    sim_node = SimpleNamespace(rel_path=REL)
    inputs = {'simulation_node': sim_node}

    # repo/lammps_scripts are never consulted once simulation_node is given.
    assert _sim_prefix_for(inputs, repo=None, lammps_scripts=None) == REL


def test_locator_for_types_a_bare_run_directory(aiida_profile_clean, tmp_path):
    run_dir = make_run(tmp_path / 'sim', REL, 'afm')

    loc = locator_for(run_dir, REL)

    assert loc.tokens.rel_path == REL
    assert loc.tokens.family == 'afm'
    assert loc.run_dir == run_dir
    assert loc.results_dir == run_dir / 'results'


def _retrieved_folder(run_dir, dest):
    """What the engine actually hands the parser, built by the retention rule.

    Not `make_run`'s tree: the parser never sees the original run directory. It
    sees a `FolderData` laid out the way `execmanager` builds it for a
    `(remote, local='.', depth=None)` wildcard entry -- each match's LOCAL
    name is `local` joined with the match's path *relative to the remote run
    root* (`rel_rem`, per `src/aiida/retention.py`'s own module docstring),
    not its bare basename. Flattening to `dest / src.name` here would silently
    reproduce the pre-Task-13 bug this whole retention rule exists to catch
    and pass every test on a tree with no colliding basenames.
    """
    import shutil

    from src.aiida.retention import RETRIEVE_LIST

    for remote, local, _depth in RETRIEVE_LIST:
        for src in sorted(run_dir.glob(remote)):
            if not src.is_file():
                continue
            rel_rem = src.relative_to(run_dir)
            out = dest / rel_rem if local == '.' else dest / local / rel_rem
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    return dest


def test_ingest_retrieved_returns_unstored_nodes(aiida_profile_clean, tmp_path):
    from src.aiida.data import FrictionSimulationData
    from src.run.tree import discover_runs

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()
    retrieved = _retrieved_folder(run_dir, tmp_path / 'retrieved')

    nodes = ingest_retrieved(retrieved, REL, run_node)

    assert (retrieved / 'results').is_dir()      # the tree survived retrieval
    assert (retrieved / 'provenance.json').is_file()
    assert len(nodes) == 2
    assert all(not n.is_stored for n in nodes)
    assert {n.load for n in nodes} == {10.0, 30.0}
    assert all(n.run_uuid == str(run_node.uuid) for n in nodes)
    assert all(n.base.attributes.get('mean_cof') is not None for n in nodes)


def test_a_flat_retrieved_folder_yields_no_points(aiida_profile_clean, tmp_path):
    """The failure mode a plain-string retrieve_list produces, pinned."""
    from src.aiida.data import FrictionSimulationData
    from src.run.tree import discover_runs

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()

    flat = tmp_path / 'flat'
    flat.mkdir()
    for src in run_dir.rglob('*'):
        if src.is_file():
            (flat / src.name).write_bytes(src.read_bytes())

    assert ingest_retrieved(flat, REL, run_node) == []


@pytest.mark.aiida
def test_real_retrieval_then_ingest_under_a_prefix(aiida_profile_clean, tmp_path):
    """R1a (fix round): the real `execmanager` retrieval of a run staged at a
    prefix (the actual AFM shape, `<K-level>/L{n}`), then `ingest_retrieved`
    on what came back, end to end -- no synthetic reproduction of the
    retrieved layout, the genuine transport + retention-list code path.
    """
    import asyncio

    from aiida.engine.daemon.execmanager import retrieve_files_from_list
    from aiida.transports.plugins.local import LocalTransport
    from src.aiida.data import FrictionSimulationData
    from src.aiida.retention import retrieve_patterns
    from src.run.tree import discover_runs

    remote_root = tmp_path / 'remote'
    make_run(remote_root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(remote_root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()

    retrieved_dir = tmp_path / 'retrieved'
    retrieved_dir.mkdir()

    class _FakeCalculation:  # only what retrieve_files_from_list touches
        pk = 1

        @staticmethod
        def get_remote_workdir():
            return str(remote_root)

    transport = LocalTransport()
    transport.open()
    try:
        asyncio.run(retrieve_files_from_list(
            _FakeCalculation(), transport, str(retrieved_dir),
            retrieve_patterns(REL)))
    finally:
        transport.close()

    prefixed_run_dir = retrieved_dir / REL
    assert (prefixed_run_dir / 'results').is_dir()
    assert (prefixed_run_dir / 'provenance.json').is_file()

    nodes = ingest_retrieved(prefixed_run_dir, REL, run_node)

    assert {n.load for n in nodes} == {10.0, 30.0}


def test_ingest_retrieved_degrades_per_point(aiida_profile_clean, tmp_path,
                                             monkeypatch):
    """R3 (fix round): one point's file failing must not lose the others --
    exactly `ingest_run`'s own per-point try/except, replicated here because
    `ingest_retrieved` has no `IngestReport` to record a partial failure
    against and could otherwise let one bad point's exception propagate and
    abort every other point of the same run.
    """
    import src.aiida.ingest as ingest_mod
    from src.aiida.data import FrictionSimulationData
    from src.run.tree import discover_runs

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()

    real_create_results = ingest_mod.create_results

    def _flaky(run_node_, point, run_dir_, **kwargs):
        if point.load == 10.0:
            raise ValueError('synthetic corruption')
        return real_create_results(run_node_, point, run_dir_, **kwargs)

    monkeypatch.setattr(ingest_mod, 'create_results', _flaky)

    nodes = ingest_mod.ingest_retrieved(run_dir, REL, run_node)

    assert len(nodes) == 1
    assert nodes[0].load == 30.0


def test_ingest_retrieved_all_points_bad_yields_no_nodes(aiida_profile_clean,
                                                         tmp_path, monkeypatch):
    """The `ERROR_NO_LOAD_POINTS` case: every point failed, not just some."""
    import src.aiida.ingest as ingest_mod
    from src.aiida.data import FrictionSimulationData
    from src.run.tree import discover_runs

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()

    def _always_fails(*_args, **_kwargs):
        raise ValueError('synthetic corruption')

    monkeypatch.setattr(ingest_mod, 'create_results', _always_fails)

    assert ingest_mod.ingest_retrieved(run_dir, REL, run_node) == []


def test_add_results_to_group_files_stored_nodes(aiida_profile_clean, tmp_path):
    from aiida.orm import Group
    from src.aiida.data import FrictionSimulationData
    from src.aiida.ingest import add_results_to_group, group_label
    from src.run.tree import discover_runs

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1', set_label='my-set')
    run_node.store()
    retrieved = _retrieved_folder(run_dir, tmp_path / 'retrieved')
    nodes = ingest_retrieved(retrieved, REL, run_node)
    for node in nodes:
        node.store()

    add_results_to_group('my-set', nodes)

    group = Group.collection.get(label=group_label('my-set'))
    assert {n.uuid for n in nodes} <= {n.uuid for n in group.nodes}


def test_add_results_to_group_is_a_noop_for_no_nodes(aiida_profile_clean):
    from aiida.orm import Group, QueryBuilder  # pylint: disable=import-outside-toplevel
    from src.aiida.ingest import add_results_to_group, group_label  # pylint: disable=import-outside-toplevel

    add_results_to_group('empty-set', [])  # must not create a Group at all

    qb = QueryBuilder()
    qb.append(Group, filters={'label': group_label('empty-set')})
    assert qb.first() is None


def _make_calc_node(process_type='aiida.calculations:friction2d.lammps',
                    simulation_node=None):
    """A stored, bare `CalcJobNode` for `LammpsFrictionCalcJob`, wired with
    whatever inputs/outputs the caller attaches -- no live `Process`, `Runner`
    or broker needed, exactly like aiida-core's own plugin test suites build
    one for `Parser.parse_from_node`.
    """
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel
    from aiida.orm import CalcJobNode  # pylint: disable=import-outside-toplevel

    node = CalcJobNode(process_type=process_type)
    node.set_option('resources', {'num_machines': 1})
    if simulation_node is not None:
        node.base.links.add_incoming(simulation_node, link_type=LinkType.INPUT_CALC,
                                     link_label='simulation_node')
    node.store()
    return node


def _attach_retrieved(calc_node, files_root):
    """Stage every file under `files_root` into a stored `retrieved`
    `FolderData`, linked as `calc_node`'s output the way the engine would."""
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel
    from aiida.orm import FolderData  # pylint: disable=import-outside-toplevel

    retrieved = FolderData()
    for path in sorted(files_root.rglob('*')):
        if path.is_file():
            retrieved.base.repository.put_object_from_file(
                str(path), str(path.relative_to(files_root).as_posix()))
    retrieved.store()
    retrieved.base.links.add_incoming(calc_node, link_type=LinkType.CREATE,
                                      link_label='retrieved')
    return retrieved


def test_parser_parse_from_node_unprefixed(aiida_profile_clean, tmp_path):
    """`Parser.parse_from_node` end to end, no `sim_prefix`: the layout
    `prepare_for_submission` builds for a sheet-on-sheet-style run whose own
    directory already is the remote work directory.
    """
    from src.aiida.calcjob import LammpsFrictionParser  # pylint: disable=import-outside-toplevel
    from src.aiida.data import FrictionSimulationData  # pylint: disable=import-outside-toplevel
    from src.run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(10.0, 30.0))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()
    retrieved_dir = _retrieved_folder(run_dir, tmp_path / 'retrieved')

    calc_node = _make_calc_node(simulation_node=run_node)
    _attach_retrieved(calc_node, retrieved_dir)

    result, calcfunc_node = LammpsFrictionParser.parse_from_node(calc_node)

    assert not calcfunc_node.exit_status
    results_ns = result['results']
    assert {results_ns['point_0'].load, results_ns['point_1'].load} == {10.0, 30.0}
    assert all(node.is_stored for node in results_ns.values())


def test_parser_parse_from_node_prefixed(aiida_profile_clean, tmp_path):
    """The layout `prepare_for_submission` builds for a real AFM submission:
    everything nested under the run's own `rel_path` (`_sim_prefix_for`).
    """
    from src.aiida.calcjob import LammpsFrictionParser  # pylint: disable=import-outside-toplevel
    from src.aiida.data import FrictionSimulationData  # pylint: disable=import-outside-toplevel
    from src.run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1')
    run_node.store()

    # Same retention-shaped copy `_retrieved_folder` makes for the unprefixed
    # case, just landed under `REL` -- exactly what `retrieve_patterns(REL)`
    # (proven against the real execmanager in test_retention.py) produces.
    retrieved_dir = tmp_path / 'retrieved'
    _retrieved_folder(run_dir, retrieved_dir / REL)
    assert (retrieved_dir / REL / 'results').is_dir()

    calc_node = _make_calc_node(simulation_node=run_node)
    _attach_retrieved(calc_node, retrieved_dir)

    result, calcfunc_node = LammpsFrictionParser.parse_from_node(calc_node)

    assert not calcfunc_node.exit_status
    assert result['results']['point_0'].load == 30.0


def test_parser_missing_simulation_node_gets_its_own_exit_code(
        aiida_profile_clean, tmp_path):
    from src.aiida.calcjob import LammpsFrictionCalcJob, LammpsFrictionParser  # pylint: disable=import-outside-toplevel

    calc_node = _make_calc_node(simulation_node=None)
    _attach_retrieved(calc_node, tmp_path)  # an empty retrieved folder

    parser = LammpsFrictionParser(calc_node)
    exit_code = parser.parse()

    assert exit_code.status == LammpsFrictionCalcJob.exit_codes.ERROR_NO_SIMULATION_NODE.status
    assert exit_code.status != 320  # not the generic "no results" code


def test_group_results_from_node_files_calcjob_outputs(aiida_profile_clean,
                                                        tmp_path):
    """The `on_terminated` hook's own logic, invoked directly against a
    manually-built node with real output links -- no live `Process` needed
    (the controller's own suggested fallback for this fix round)."""
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel
    from src.aiida.calcjob import _group_results_from_node  # pylint: disable=import-outside-toplevel
    from src.aiida.data import FrictionResultsData, FrictionSimulationData  # pylint: disable=import-outside-toplevel
    from src.aiida.ingest import group_label  # pylint: disable=import-outside-toplevel
    from src.run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', loads=(30.0,))
    loc, = discover_runs(root)
    run_node = FrictionSimulationData()
    run_node.set_from_locator(loc, set_uuid='set-1', set_label='my-set')
    run_node.store()

    calc_node = _make_calc_node(simulation_node=run_node)
    result_node = FrictionResultsData()
    result_node.family = 'afm'
    result_node.load_kind = 'force'
    result_node.load = 30.0
    result_node.angle = 0.0
    result_node.speed = 2.0
    result_node.run_uuid = str(run_node.uuid)
    result_node.set_uuid = str(run_node.uuid)
    result_node.store()
    # The REAL link label a finished CalcJob node carries -- `Process.out()`
    # accepts the dotted `'results.point_0'` spelling, but link labels only
    # allow alphanumerics/underscores, so the engine actually stores it
    # joined with `PORT_NAMESPACE_SEPARATOR = '__'` once outputs are attached.
    result_node.base.links.add_incoming(calc_node, link_type=LinkType.CREATE,
                                        link_label='results__point_0')

    _group_results_from_node(calc_node)

    group = Group.collection.get(label=group_label('my-set'))
    assert result_node.uuid in {n.uuid for n in group.nodes}


def test_group_results_from_node_is_a_noop_without_simulation_node(
        aiida_profile_clean):
    from src.aiida.calcjob import _group_results_from_node

    calc_node = _make_calc_node(simulation_node=None)

    _group_results_from_node(calc_node)  # must not raise
