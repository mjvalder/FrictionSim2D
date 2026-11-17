"""What is in the store, without downloading it.

The panel answers "what exactly does this row hold" -- every attribute of the
point, its selected verdict and summary, the `related` counts and the stored
file list with sizes -- so the user does not have to export a bundle to find
out.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.widgets.database.detail import DetailPanel, format_bytes  # noqa: E402


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


DETAIL = {
    'ok': True, 'kind': 'detail', 'source': 'local',
    'point': {'uuid': '1a2b', 'node_type': 'data.core.folder.FolderData.',
              'attributes': {'rel_path': 'afm/h-MoS2/100x_100y/K300/L2', 'layers': 2,
                             'load': 30.0}},
    'verdict': {'uuid': '9f8e', 'node_type': 'data.core.dict.Dict.',
                'attributes': {'status': 'complete', 'reasons': []}},
    'summary': {'uuid': '4d4d', 'node_type': 'data.core.dict.Dict.',
                'attributes': {'mean_cof': 0.0731}},
    'related': {'thermostat': 0, 'realization': 3, 'load': 2, 'speed': 0},
    'files': [{'node': 'point', 'path': 'results/friction_f30_a0_s2.txt', 'bytes': 812},
              {'node': 'point', 'path': 'results/window.csv', 'bytes': 40213}],
    'total_bytes': 41025,
}


@pytest.mark.parametrize('count,expected', [
    (0, '0 B'), (812, '812 B'), (40213, '39.3 kB'), (41025, '40.1 kB'),
    (5 * 1024 * 1024, '5.0 MB'),
])
def test_format_bytes(count, expected):
    assert format_bytes(count) == expected


@pytest.fixture(name='panel')
def _panel(qt_app):
    del qt_app
    return DetailPanel()


def test_an_empty_panel_has_no_rows(panel):
    assert panel.rows() == []


def test_it_shows_every_point_verdict_and_summary_attribute(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['rel_path'] == 'afm/h-MoS2/100x_100y/K300/L2'
    assert rows['layers'] == '2'
    assert ('load', '30.0') in panel.rows()
    assert rows['status'] == 'complete'
    assert rows['mean_cof'] == '0.0731'


def test_it_shows_the_node_identities(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['point uuid'] == '1a2b'
    assert rows['verdict uuid'] == '9f8e'
    assert rows['summary uuid'] == '4d4d'


def test_it_shows_the_related_counts(panel):
    panel.show_detail(DETAIL)

    rows = panel.rows()
    assert ('realization', '3') in rows
    # `load` is both a point attribute and a related kind: two groups.
    assert ('load', '30.0') in rows and ('load', '2') in rows


def test_it_lists_the_stored_files_with_sizes(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['point: results/friction_f30_a0_s2.txt'] == '812 B'
    assert rows['point: results/window.csv'] == '39.3 kB'
    assert rows['total stored'] == '40.1 kB'


def test_a_point_without_a_verdict_or_summary_says_so(panel):
    panel.show_detail(dict(DETAIL, verdict=None, summary=None, related={}, files=[]))

    rows = dict(panel.rows())
    assert rows['verdict uuid'] == 'none'
    assert rows['summary uuid'] == 'none'
    assert 'total stored' not in rows


def test_showing_a_second_detail_replaces_the_first(panel):
    panel.show_detail(DETAIL)
    panel.show_detail({'point': {'uuid': 'zzz', 'node_type': '', 'attributes': {}},
                       'verdict': None, 'summary': None,
                       'files': [], 'total_bytes': 0})

    rows = dict(panel.rows())
    assert rows['point uuid'] == 'zzz'
    assert 'rel_path' not in rows


def test_clear_empties_it(panel):
    panel.show_detail(DETAIL)
    panel.clear()

    assert panel.rows() == []
