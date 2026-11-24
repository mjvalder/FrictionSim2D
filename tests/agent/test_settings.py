"""Agent settings (agent.yaml) and the Gemini key file (agent.secrets)."""
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from frictionsim2d.agent.settings import (AgentSettings, api_key, config_dir,
                                load_agent_settings, save_agent_settings, store_api_key)


def test_defaults_match_phase0():
    s = AgentSettings()
    assert s.provider == 'gemini'
    assert s.gemini.model == 'gemini-3.5-flash-lite'
    assert s.gemini.api_key_env == 'GEMINI_API_KEY'
    assert s.ollama.model == 'fs2d-assistant:qwen3.5-4b'
    assert (s.ollama.context_tokens, s.ollama.keep_alive) == (16384, '2m')
    assert s.ollama.idle_minutes == 10
    assert s.max_requests_per_turn == 50
    assert s.materials_project.enabled is True


def test_config_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    assert config_dir() == tmp_path / 'FrictionSim2D'


def test_missing_file_gives_defaults(tmp_path):
    assert load_agent_settings(tmp_path / 'none.yaml') == AgentSettings()


def test_round_trip(tmp_path):
    s = AgentSettings(provider='ollama')
    s.gemini.consent_given = True
    path = save_agent_settings(s, tmp_path / 'agent.yaml')
    assert load_agent_settings(path) == s


def test_invalid_provider_rejected(tmp_path):
    path = tmp_path / 'agent.yaml'
    path.write_text('provider: groq\n')
    with pytest.raises(ValidationError):
        load_agent_settings(path)


def test_env_key_wins_over_file(monkeypatch, tmp_path):
    secrets = store_api_key('file-key', 'gemini', tmp_path / 'agent.secrets')
    monkeypatch.setenv('GEMINI_API_KEY', ' env-key ')
    assert api_key(AgentSettings(), 'gemini', secrets) == 'env-key'
    monkeypatch.delenv('GEMINI_API_KEY')
    assert api_key(AgentSettings(), 'gemini', secrets) == 'file-key'


def test_no_key_anywhere(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert api_key(AgentSettings(), 'gemini', tmp_path / 'absent') is None


def test_secrets_file_is_owner_only_even_if_it_existed(tmp_path):
    path = tmp_path / 'agent.secrets'
    path.write_text('old')
    path.chmod(0o644)
    store_api_key('k', 'gemini', path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == 'gemini=k\n'


def test_each_provider_keeps_its_own_key(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    path = store_api_key('g', 'gemini', tmp_path / 'agent.secrets')
    store_api_key('a', 'anthropic', path)
    store_api_key('g2', 'gemini', path)
    s = AgentSettings()
    assert (api_key(s, 'gemini', path), api_key(s, 'anthropic', path)) == ('g2', 'a')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'env-a')
    assert api_key(s, 'anthropic', path) == 'env-a'


def test_anthropic_defaults():
    s = AgentSettings(provider='anthropic')
    assert (s.anthropic.model, s.anthropic.api_key_env) == ('claude-opus-5-5', 'ANTHROPIC_API_KEY')
    assert s.anthropic.effort == 'medium' and not s.anthropic.consent_given


def test_openai_defaults_and_its_own_key(monkeypatch, tmp_path):
    from frictionsim2d.agent.settings import OPENAI_MODELS
    s = AgentSettings(provider='openai')
    assert (s.openai.model, s.openai.api_key_env) == ('gpt-5.5', 'OPENAI_API_KEY')
    assert s.openai.model in OPENAI_MODELS and not s.openai.consent_given
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    path = store_api_key('sk-o', 'openai', tmp_path / 'agent.secrets')
    assert api_key(s, 'openai', path) == 'sk-o' and api_key(s, 'gemini', path) is None


def test_key_never_written_to_settings_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'secret-value')
    path = save_agent_settings(AgentSettings(), tmp_path / 'agent.yaml')
    assert 'secret-value' not in path.read_text()


def test_ask_before_apply_defaults_off_and_round_trips(tmp_path):
    assert AgentSettings().ask_before_apply is False
    s = AgentSettings(ask_before_apply=True)
    path = save_agent_settings(s, tmp_path / 'agent.yaml')
    assert load_agent_settings(path).ask_before_apply is True


def test_idle_minutes_and_step_cap_round_trip_and_validate(tmp_path):
    s = AgentSettings(max_requests_per_turn=3)
    s.ollama.idle_minutes = 2
    path = save_agent_settings(s, tmp_path / 'agent.yaml')
    loaded = load_agent_settings(path)
    assert (loaded.ollama.idle_minutes, loaded.max_requests_per_turn) == (2, 3)
    with pytest.raises(ValidationError):
        AgentSettings(max_requests_per_turn=0)
    path.write_text('ollama:\n  idle_minutes: 0\n')
    with pytest.raises(ValidationError):
        load_agent_settings(path)


def test_an_old_agent_yaml_keeps_its_model(tmp_path):
    """No stock fallback and no name migration: the pane offers Download instead."""
    path = tmp_path / 'agent.yaml'
    path.write_text('provider: ollama\nsettings_version: 1\nollama:\n  model: fs2d-qwen3-4b\n'
                    '  max_threads: 8\n')
    assert load_agent_settings(path).ollama.model == 'fs2d-qwen3-4b'


def test_data_dir_honours_xdg(monkeypatch, tmp_path):
    from frictionsim2d.agent.settings import data_dir
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert data_dir() == tmp_path / 'FrictionSim2D'
    monkeypatch.delenv('XDG_DATA_HOME')
    assert data_dir() == Path.home() / '.local' / 'share' / 'FrictionSim2D'


@pytest.mark.parametrize('key', ['abc\ndef', 'abc\rdef', 'abc\r\ngemini=evil'])
def test_key_with_line_break_refused(tmp_path, key):
    path = tmp_path / 'agent.secrets'
    with pytest.raises(ValueError, match='line break'):
        store_api_key(key, 'gemini', path)
    assert not path.exists()


def test_key_file_symlink_not_followed(tmp_path):
    victim = tmp_path / 'victim.txt'
    victim.write_text('keep')
    link = tmp_path / 'agent.secrets'
    link.symlink_to(victim)
    with pytest.raises(OSError):
        store_api_key('k', 'gemini', link)
    assert victim.read_text() == 'keep'


def test_new_config_dir_is_owner_only(tmp_path):
    path = tmp_path / 'cfg' / 'FrictionSim2D' / 'agent.secrets'
    store_api_key('k', 'gemini', path)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
