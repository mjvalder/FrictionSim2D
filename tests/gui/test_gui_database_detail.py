"""What is in the store, without downloading it.

The panel answers "what exactly does this row hold" -- every attribute of the
run and result nodes plus the stored file list with sizes -- so the user does
not have to export a bundle to find out.
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
    'run': {'uuid': '1a2b', 'node_type': 'data.friction2d.simulation.X.',
            'attributes': {'rel_path': 'afm/h-MoS2/100x_100y/K300/L2', 'layers': 2}},
    'result': {'uuid': '9f8e', 'node_type': 'data.friction2d.results.X.',
               'attributes': {'load': 30.0, 'mean_cof': 0.0731}},
    'files': [{'node': 'run', 'path': 'provenance.json', 'bytes': 812},
              {'node': 'result', 'path': 'results/window.csv', 'bytes': 40213}],
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


def test_it_shows_every_run_and_result_attribute(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['rel_path'] == 'afm/h-MoS2/100x_100y/K300/L2'
    assert rows['layers'] == '2'
    assert rows['load'] == '30.0'
    assert rows['mean_cof'] == '0.0731'


def test_it_shows_the_node_identities(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['run uuid'] == '1a2b'
    assert rows['result uuid'] == '9f8e'


def test_it_lists_the_stored_files_with_sizes(panel):
    panel.show_detail(DETAIL)

    rows = dict(panel.rows())
    assert rows['run: provenance.json'] == '812 B'
    assert rows['result: results/window.csv'] == '39.3 kB'
    assert rows['total stored'] == '40.1 kB'


def test_a_central_detail_shows_the_record_and_no_file_list(panel):
    panel.show_detail({'ok': True, 'kind': 'detail', 'source': 'central',
                       'run': {'uuid': '', 'node_type': '', 'attributes': {}},
                       'result': {'uuid': '9f8e', 'node_type': '',
                                  'attributes': {'id': 41, 'mean_cof': 0.07}},
                       'files': [], 'total_bytes': 0})

    rows = dict(panel.rows())
    assert rows['id'] == '41'
    assert 'total stored' not in rows


def test_showing_a_second_detail_replaces_the_first(panel):
    panel.show_detail(DETAIL)
    panel.show_detail({'run': {'uuid': 'zzz', 'node_type': '', 'attributes': {}},
                       'result': {'uuid': '', 'node_type': '', 'attributes': {}},
                       'files': [], 'total_bytes': 0})

    rows = dict(panel.rows())
    assert rows['run uuid'] == 'zzz'
    assert 'rel_path' not in rows


def test_clear_empties_it(panel):
    panel.show_detail(DETAIL)
    panel.clear()

    assert panel.rows() == []
