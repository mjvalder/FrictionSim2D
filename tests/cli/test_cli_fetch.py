"""`FrictionSim2D fetch model`: the agent-extra guard, usage, and clean errors."""
import pytest
from click.testing import CliRunner

from frictionsim2d import agent
from frictionsim2d.cli import cli


def test_without_the_agent_extra_it_says_how_to_install(monkeypatch):
    monkeypatch.setattr(agent, '_AGENT_AVAILABLE', False)
    result = CliRunner().invoke(cli, ['fetch', 'model', 'qwen3.5-4b'])
    assert result.exit_code == 1 and 'pip install -e ".[agent]"' in result.output


def test_remove_needs_a_tag(monkeypatch):
    monkeypatch.setattr(agent, '_AGENT_AVAILABLE', True)
    result = CliRunner().invoke(cli, ['fetch', 'model', '--remove'])
    assert result.exit_code == 2 and '--remove needs a TAG' in result.output


def test_a_fetch_error_is_one_line_not_a_traceback(monkeypatch):
    pytest.importorskip('httpx')
    from frictionsim2d.agent import fetch
    monkeypatch.setattr(agent, '_AGENT_AVAILABLE', True)

    def _fail(tag, remove, echo):
        raise fetch.FetchError('Ollama is not installed')
    monkeypatch.setattr(fetch, 'fetch_command', _fail)
    result = CliRunner().invoke(cli, ['fetch', 'model', 'qwen3.5-4b'])
    assert result.exit_code == 1 and '❌ Ollama is not installed' in result.output
    assert 'Traceback' not in result.output


def test_the_tag_and_remove_flag_reach_the_package(monkeypatch):
    pytest.importorskip('httpx')
    from frictionsim2d.agent import fetch
    monkeypatch.setattr(agent, '_AGENT_AVAILABLE', True)
    seen = []
    monkeypatch.setattr(fetch, 'fetch_command',
                        lambda tag, remove, echo: seen.append((tag, remove)) or echo('done'))
    result = CliRunner().invoke(cli, ['fetch', 'model', 'qwen3.5-2b', '--remove'])
    assert result.exit_code == 0 and seen == [('qwen3.5-2b', True)] and 'done' in result.output
