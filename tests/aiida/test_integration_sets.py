"""list-sets / delete / clear over the CalcJob graph (`src/aiida/integration.py`)."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.integration import (  # noqa: E402
    clear_all_nodes, delete_simulation_set, list_simulation_sets,
)
from src.run.tree import discover_runs  # noqa: E402
from tests.aiida.test_import import _calcjobs, _tree  # noqa: E402
from tests.helpers.synthetic_results import write_results  # noqa: E402


@pytest.fixture
def no_profile_switch(monkeypatch):
    """`aiida_profile_clean` is already loaded; do not load another."""
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _content_nodes():
    return orm.QueryBuilder().append(
        orm.Node, filters={'or': [{'extras': {'has_key': k}} for k in (
            'friction2d_block_hash', 'friction2d_sha256', 'friction2d_folder_sha256',
            'friction2d_schema_hash')]}).count()


def test_list_sets_reports_the_new_shape(aiida_profile_clean, no_profile_switch, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    (row,) = list_simulation_sets()
    assert row['label'] == 'camp'
    assert (row['n_decks'], row['n_points'], row['n_runs'], row['n_results']) == (4, 2, 2, 2)
    assert row['source_root'] == str(root.resolve())
    assert row['model'] == 'afm' and row['materials'] == ['h-MoS2', 'h-WS2']


def test_list_sets_counts_only_live_results(aiida_profile_clean, no_profile_switch, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    (loc,) = [r for r in discover_runs(root) if 'h-MoS2' in r.tokens.rel_path]
    write_results(loc.run_dir, 'afm', loads=(10.0,), tiers=('raw',), layers=2)
    import_root(root, label='camp')
    assert len(_calcjobs()) == 5
    (row,) = list_simulation_sets()
    assert (row['n_decks'], row['n_points']) == (4, 2)


def test_delete_removes_the_group_and_its_own_nodes(aiida_profile_clean,
                                                    no_profile_switch, tmp_path):
    import_root(_tree(tmp_path / 'sim'), label='camp')
    content = _content_nodes()

    deleted = delete_simulation_set('camp')

    assert deleted > 4
    assert _calcjobs() == []
    assert not orm.QueryBuilder().append(
        orm.Group, filters={'label': 'friction2d/camp'}).count()
    assert _content_nodes() == content           # shared inputs are kept
    assert orm.QueryBuilder().append(orm.InstalledCode).count() == 1
    assert list_simulation_sets() == []


def test_delete_keeps_calcjobs_another_label_holds(aiida_profile_clean,
                                                   no_profile_switch, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='first')
    import_root(root, label='second')
    calcs = {n.pk for n in _calcjobs()}

    assert delete_simulation_set('first') == 0

    assert {n.pk for n in _calcjobs()} == calcs
    (row,) = list_simulation_sets()
    assert row['label'] == 'second' and row['n_decks'] == 4 and row['n_points'] == 2
    for calc in _calcjobs():
        assert 'points' in calc.outputs or 'structures' in calc.outputs


def test_delete_keeps_shared_and_removes_own(aiida_profile_clean, no_profile_switch,
                                             tmp_path):
    """`second` holds the shared decks plus one deck of its own (a re-import
    that superseded `first`'s slide). Deleting `second` removes only that."""
    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    import_root(root, label='first')
    import_root(root, label='second')
    (loc,) = discover_runs(root)
    write_results(loc.run_dir, 'afm', loads=(10.0,), tiers=('raw',), layers=2)
    import_root(root, label='second')
    shared = {n.pk for n in _calcjobs()
              if n.base.extras.get('superseded', False) or
              n.inputs.parameters['deck_name'] == 'system'}

    delete_simulation_set('second')

    assert {n.pk for n in _calcjobs()} == shared


def test_clear_removes_everything_and_a_reimport_recreates_it(aiida_profile_clean,
                                                              no_profile_switch, tmp_path):
    root = _tree(tmp_path / 'sim')
    first = import_root(root, label='camp')
    import_root(root, label='other')

    assert clear_all_nodes() > 0

    assert _calcjobs() == []
    assert _content_nodes() == 0
    assert orm.QueryBuilder().append(orm.InstalledCode).count() == 0
    assert orm.QueryBuilder().append(orm.Data).count() == 0
    assert not orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': 'friction2d/%'}}).count()

    again = import_root(root, label='camp')
    assert again.ok(), again.render()
    assert len(again.created_decks) == len(first.created_decks) == 4
    assert again.already_imported == [] and again.skipped_decks == []
