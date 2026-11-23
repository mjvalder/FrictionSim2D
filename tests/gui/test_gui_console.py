"""The GUI keeps the launching terminal quiet unless asked not to."""
import logging
import warnings

import pytest

from frictionsim2d.gui import quiet_console


@pytest.fixture(name='restore_logging')
def _restore_logging():
    names = ('', 'httpx', 'httpcore')
    levels = {name: logging.getLogger(name).level for name in names}
    with warnings.catch_warnings():
        yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_quiet_silences_http_logs_banner_and_ovito_warning(restore_logging, monkeypatch):
    monkeypatch.delenv('PYDANTIC_AI_NO_BANNER', raising=False)
    logging.getLogger().setLevel(logging.INFO)
    quiet_console()
    assert not logging.getLogger('httpx').isEnabledFor(logging.INFO)
    assert not logging.getLogger().isEnabledFor(logging.INFO)
    assert logging.getLogger().isEnabledFor(logging.WARNING)
    assert __import__('os').environ['PYDANTIC_AI_NO_BANNER'] == '1'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')
        quiet_console()
        warnings.warn('Did you accidentally install the OVITO package from the PyPI repository?')
    assert caught == []


def test_verbose_leaves_everything_alone(restore_logging, monkeypatch):
    monkeypatch.delenv('PYDANTIC_AI_NO_BANNER', raising=False)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.NOTSET)
    quiet_console(verbose=True)
    assert logging.getLogger('httpx').isEnabledFor(logging.INFO)
    assert 'PYDANTIC_AI_NO_BANNER' not in __import__('os').environ
