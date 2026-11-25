"""Renderer noise is collapsed; everything else still gets through.

The viewport's picking pass fails to bind a buffer for ovito's thick-line
shader whenever the simulation cell is drawn, at two lines per pointer motion.
Filtering it is only acceptable if the filter is narrow, so these tests pin
both halves: the noise goes, and nothing else does.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import qCritical, qWarning  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from frictionsim2d.gui import qt_logging  # noqa: E402


@pytest.fixture(name='filtered')
def _filtered(capsys):
    app = QApplication.instance() or QApplication([])
    qt_logging._counts.clear()
    qt_logging.install_message_filter()
    yield capsys
    qt_logging._counts.clear()


#: The exact messages seen in the terminal.
NOISE = [
    'OpenGLShaderHelper::bindBuffer() failed for shader "line_thick_picking"',
    "ERROR: Failed to bind OpenGL vertex buffer for shader 'line_thick_picking'.",
]


@pytest.mark.parametrize('message', NOISE)
def test_renderer_noise_is_not_printed(filtered, message):
    qWarning(message)
    assert 'line_thick_picking' not in filtered.readouterr().err


@pytest.mark.parametrize('pattern', [
    'OpenGLShaderHelper::bindBuffer',
    'Failed to bind OpenGL vertex buffer',
    'line_thick_picking',
])
def test_each_pattern_alone_suppresses(filtered, pattern):
    """A message carrying only one pattern is still collapsed."""
    qWarning(f'prefix {pattern} suffix')
    assert pattern not in filtered.readouterr().err
    assert qt_logging.suppressed_summary() is not None


def test_noise_is_counted_rather_than_discarded(filtered):
    for message in NOISE * 3:
        qWarning(message)
    filtered.readouterr()
    summary = qt_logging.suppressed_summary()
    assert summary is not None
    assert '6' in summary


def test_nothing_is_reported_when_there_was_no_noise(filtered):
    assert qt_logging.suppressed_summary() is None


def test_unrelated_warnings_still_reach_the_terminal(filtered):
    qWarning('the potential file could not be read')
    assert 'the potential file could not be read' in filtered.readouterr().err


def test_unrelated_criticals_still_reach_the_terminal(filtered):
    qCritical('something is badly wrong')
    err = filtered.readouterr().err
    assert 'something is badly wrong' in err
    assert 'critical' in err


def test_a_message_merely_mentioning_opengl_is_not_swallowed(filtered):
    """The filter must be narrow: a real OpenGL problem is worth seeing."""
    qWarning('OpenGL context creation failed')
    assert 'OpenGL context creation failed' in filtered.readouterr().err
