"""The GUI database worker against a real (temporary) profile."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('aiida')
# The old set/run/result ingest is gone (plan Task 6, `import_root`);
# Task 11 deletes it (old result nodes).
pytest.skip('old ingest graph; Task 11 deletes it (old result nodes)', allow_module_level=True)
pytestmark = pytest.mark.aiida

from src.aiida.ingest import ingest_root  # noqa: E402
from src.gui import db_worker  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

AFM = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def _populate(tmp_path):
    root = tmp_path / 'simulation_w'
    make_run(root, AFM, 'afm', loads=(10.0, 30.0), tiers=('window',))
    ingest_root(root, label='w1')
    return root


def test_list_reads_the_store(aiida_profile_clean, tmp_path):
    _populate(tmp_path)

    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'profile': None,
                                 'filters': {'set_label': 'w1'}})

    assert result['ok'] is True
    assert result['returned'] == 2
    assert {r['load'] for r in result['records']} == {10.0, 30.0}
    assert result['records'][0]['run_uuid']
    assert 'imported_at' in result['records'][0], \
        'the Added column reads this -- to_dict() already carries it as an extra'
    assert 'h-MoS2' in result['facets']['materials']
    assert result['facets']['set_labels'] == ['w1']


def test_detail_carries_attributes_and_the_stored_file_list(aiida_profile_clean,
                                                            tmp_path):
    _populate(tmp_path)
    listed = db_worker.dispatch({'kind': 'list', 'source': 'local', 'filters': {}})
    row = listed['records'][0]

    detail = db_worker.dispatch({'kind': 'detail', 'source': 'local',
                                 'run_uuid': row['run_uuid'],
                                 'result_uuid': row['result_uuid']})

    assert detail['ok'] is True
    assert detail['run']['uuid'] == row['run_uuid']
    assert detail['run']['attributes']['rel_path'] == AFM
    assert detail['result']['attributes']['load'] in (10.0, 30.0)
    assert detail['files'], 'the run stores at least provenance.json'
    assert all(f['node'] in ('run', 'result') for f in detail['files'])
    assert detail['total_bytes'] == sum(f['bytes'] for f in detail['files'])


def test_export_results_copies_the_stored_files(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    row = db_worker.dispatch({'kind': 'list', 'source': 'local',
                              'filters': {}})['records'][0]
    dest = tmp_path / 'download'

    result = db_worker.dispatch({
        'kind': 'export', 'source': 'local', 'what': 'results', 'dest': str(dest),
        'selection': [{'run_uuid': row['run_uuid'],
                       'result_uuid': row['result_uuid'],
                       'set_label': row['set_label']}]})

    assert result['ok'] is True
    assert result['files'], 'nothing was written'
    assert all(Path(p).is_file() for p in result['files'])
    assert all(str(dest) in p for p in result['files'])


def test_export_provenance_writes_one_bundle_per_run_and_no_results(
        aiida_profile_clean, tmp_path):
    """The bundle is provenance-only: a run's own inputs, not its set."""
    _populate(tmp_path)
    row = db_worker.dispatch({'kind': 'list', 'source': 'local',
                              'filters': {}})['records'][0]
    dest = tmp_path / 'bundle'

    result = db_worker.dispatch({
        'kind': 'export', 'source': 'local', 'what': 'provenance',
        'dest': str(dest),
        'selection': [{'run_uuid': row['run_uuid'],
                       'result_uuid': row['result_uuid'],
                       'set_label': row['set_label']}]})

    assert result['ok'] is True
    written = [Path(f) for f in result['files']]
    assert (dest / AFM / 'provenance.json') in written
    assert (dest / AFM / 'provenance' / 'config.json') in written
    assert (dest / AFM / 'provenance' / 'settings.yaml') in written
    assert not [f for f in written if '/results/' in str(f)], \
        'the provenance bundle must not drag the results files along'


def test_export_run_tree_rebuilds_a_readable_tree(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    dest = tmp_path / 'tree'

    result = db_worker.dispatch({
        'kind': 'export', 'source': 'local', 'what': 'run_tree', 'dest': str(dest),
        'rebuild': False,
        'selection': [{'run_uuid': '', 'result_uuid': '', 'set_label': 'w1'}]})

    assert result['ok'] is True
    assert (dest / 'w1' / AFM / 'provenance.json').is_file()


def test_import_ingests_a_run_directory(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_new'
    make_run(root, AFM, 'afm', loads=(30.0,), tiers=('window',))

    result = db_worker.dispatch({'kind': 'import', 'source': 'local',
                                 'root': str(root), 'label': 'imported',
                                 'dry_run': False})

    assert result['ok'] is True
    assert result['ingest_ok'] is True
    assert result['created_runs'] == [AFM]
    assert result['label'] == 'imported'

    listed = db_worker.dispatch({'kind': 'list', 'source': 'local',
                                 'filters': {'set_label': 'imported'}})
    assert listed['returned'] == 1
