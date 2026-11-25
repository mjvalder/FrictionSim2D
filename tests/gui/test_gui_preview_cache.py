"""A failed preview must not be cached.

`PreviewController.start` reused any `preview.json` under the request's cache
key, including `ok: false`. A failure caused by something outside the key --
a path the staging got wrong, a potential file added since -- then replayed
forever for that config, even after the cause was fixed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QProcess  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from frictionsim2d.core.config import parse_config  # noqa: E402
from frictionsim2d.gui.controllers.preview import (  # noqa: E402
    PreviewController, PreviewRequest, cache_key)
from frictionsim2d.gui.workspace import Workspace  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / 'examples'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='request_and_controller')
def _request_and_controller(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES.parent)
    started = []
    monkeypatch.setattr(QProcess, 'start', lambda self, *a: started.append(True))
    controller = PreviewController(Workspace.open(tmp_path / 'ws'))
    request = PreviewRequest(
        config=parse_config(EXAMPLES / 'afm_config.ini'), model='afm', layer=None,
        settings_file=None, referenced_files=[], base_dir=EXAMPLES)
    result_file = controller.workspace.preview_path(request.key()) / 'preview.json'
    result_file.parent.mkdir(parents=True)
    return controller, request, result_file, started


def test_a_cached_failure_is_rebuilt(request_and_controller):
    controller, request, result_file, started = request_and_controller
    result_file.write_text(json.dumps({'ok': False, 'error': 'stale'}), encoding='utf-8')
    failures = []
    controller.failed.connect(failures.append)

    controller.start(request)

    assert failures == []
    assert started


def test_a_cached_success_is_still_reused(request_and_controller):
    controller, request, result_file, started = request_and_controller
    result_file.write_text(json.dumps({'ok': True}), encoding='utf-8')

    controller.start(request)

    assert not started


class TestCacheKey:
    """A preview may only be reused when it would be rebuilt identically."""

    def test_same_input_same_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) == cache_key(config, 'afm', 1)

    def test_config_change_changes_the_key(self):
        assert cache_key({'2D': {'mat': 'MoS2'}}, 'afm', 1) != \
            cache_key({'2D': {'mat': 'WS2'}}, 'afm', 1)

    def test_model_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'indent', 1)

    def test_layer_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'afm', 2)

    def test_referenced_file_change_changes_the_key(self, tmp_path):
        potential = tmp_path / 'p.sw'
        potential.write_text('a', encoding='utf-8')
        config = {'2D': {'mat': 'MoS2'}}
        before = cache_key(config, 'afm', 1, [potential])
        potential.write_text('bb', encoding='utf-8')
        assert cache_key(config, 'afm', 1, [potential]) != before

    def test_the_code_is_part_of_the_key(self, monkeypatch):
        """Otherwise editing a template and re-rendering would show a system
        built by the previous code."""
        import frictionsim2d.gui.controllers.preview as preview_module
        config = {'2D': {'mat': 'MoS2'}}
        monkeypatch.setattr(preview_module, 'code_fingerprint', lambda: 'a')
        before = cache_key(config, 'afm', 1)
        monkeypatch.setattr(preview_module, 'code_fingerprint', lambda: 'b')
        assert cache_key(config, 'afm', 1) != before

    def test_no_layer_is_not_layer_none_text_of_another_layer(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', None) != cache_key(config, 'afm', 0)
