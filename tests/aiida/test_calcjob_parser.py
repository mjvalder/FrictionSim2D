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
