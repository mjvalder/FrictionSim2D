"""Agent settings (agent.yaml) and the Gemini key file (agent.secrets)."""
import stat

import pytest
from pydantic import ValidationError

from src.agent.settings import (AgentSettings, config_dir, gemini_key,
                                load_agent_settings, save_agent_settings,
                                store_gemini_key)


def test_defaults_match_phase0():
    s = AgentSettings()
    assert s.provider == 'gemini'
    assert s.gemini.model == 'gemini-3.5-flash-lite'
    assert s.gemini.api_key_env == 'GEMINI_API_KEY'
    assert s.ollama.model == 'qwen3:4b'
    assert (s.ollama.max_threads, s.ollama.context_tokens, s.ollama.keep_alive) == (8, 8192, '2m')
    assert s.ollama.idle_minutes == 10
    assert s.max_requests_per_turn == 50


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
    secrets = store_gemini_key('file-key', tmp_path / 'agent.secrets')
    monkeypatch.setenv('GEMINI_API_KEY', ' env-key ')
    assert gemini_key(AgentSettings(), secrets) == 'env-key'
    monkeypatch.delenv('GEMINI_API_KEY')
    assert gemini_key(AgentSettings(), secrets) == 'file-key'


def test_no_key_anywhere(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert gemini_key(AgentSettings(), tmp_path / 'absent') is None


def test_secrets_file_is_owner_only_even_if_it_existed(tmp_path):
    path = tmp_path / 'agent.secrets'
    path.write_text('old')
    path.chmod(0o644)
    store_gemini_key('k', path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == 'gemini=k\n'


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


def test_legacy_derived_model_falls_back_when_not_installed(tmp_path):
    from src.agent.settings import fallback_ollama_model
    path = tmp_path / 'agent.yaml'
    path.write_text('provider: ollama\nollama:\n  model: fs2d-qwen3-4b\n')
    s = load_agent_settings(path)
    assert s.ollama.model == 'fs2d-qwen3-4b'  # kept until we know what is installed
    note = fallback_ollama_model(s, ['qwen3:4b', 'qwen3:8b', 'llama3.1:8b'])
    assert s.ollama.model == 'qwen3:4b' and 'fs2d-qwen3-4b' in note and 'qwen3:4b' in note


def test_legacy_model_kept_when_installed_and_others_untouched():
    from src.agent.settings import fallback_ollama_model
    s = AgentSettings()
    s.ollama.model = 'fs2d-qwen3-4b'
    assert fallback_ollama_model(s, ['fs2d-qwen3-4b:latest', 'qwen3:4b']) is None
    assert s.ollama.model == 'fs2d-qwen3-4b'
    s.ollama.model = 'mistral:7b'  # a user's own choice is never rewritten
    assert fallback_ollama_model(s, ['qwen3:4b']) is None and s.ollama.model == 'mistral:7b'
