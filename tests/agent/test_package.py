"""The agent package's availability flag and result compaction."""
from importlib.util import find_spec

from src.agent import _AGENT_AVAILABLE
from src.agent.tools import compact


def test_flag_matches_install():
    assert _AGENT_AVAILABLE is (find_spec('pydantic_ai') is not None)


def test_compact_keeps_short_text():
    assert compact('a\nb') == 'a\nb'


def test_compact_truncates_by_line_and_says_how_many():
    text = '\n'.join(f'line {i:04d}' for i in range(500))
    out = compact(text, limit=200)
    assert len(out) <= 200
    assert out.splitlines()[-1].startswith('... ')
    assert out.splitlines()[-1].endswith(' more lines')
