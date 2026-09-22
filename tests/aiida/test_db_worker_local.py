"""The GUI database worker against a real (temporary) profile."""

from __future__ import annotations

import pytest

pytest.importorskip('aiida')
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
