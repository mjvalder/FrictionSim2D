"""The manual describes the window that exists.

`documentation/` is the tracked manual, and its "Not included" paragraph
claimed the results database was Part 3 future work long after it shipped
would be exactly the kind of drift that makes a manual untrustworthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DOC = REPO_ROOT / 'documentation' / 'gui.md'


@pytest.fixture(name='text', scope='module')
def _text():
    return GUI_DOC.read_text(encoding='utf-8')


def test_there_is_a_database_section(text):
    assert '\n## Database\n' in text


def test_it_names_every_download(text):
    section = text.split('\n## Database\n', 1)[1]
    for label in ('Results files', 'Provenance bundle', 'Summary CSV',
                  'Recreate run tree'):
        assert label in section, f"the manual does not mention {label}"


def test_it_names_the_write_action_and_the_check(text):
    section = text.split('\n## Database\n', 1)[1]
    assert 'Import run directory' in section
    assert 'Check connection' in section


def test_it_says_the_central_source_was_removed(text):
    section = text.split('\n## Database\n', 1)[1]
    assert 'Copy owner contact' not in section
    assert 'removed' in section


def test_it_says_where_the_connection_settings_come_from(text):
    section = text.split('\n## Database\n', 1)[1]
    assert 'settings.aiida.profile' in section
    assert 'database.api_url' not in section


def test_not_included_no_longer_claims_the_database_is_missing(text):
    not_included = text.split('\n## Not included\n', 1)[1]
    assert 'results database' not in not_included
    assert 'Job monitoring' in not_included
