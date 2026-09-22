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
