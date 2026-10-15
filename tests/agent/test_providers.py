"""Provider construction and the Ollama helpers."""
import httpx
import pytest

pytest.importorskip('pydantic_ai')

from src.agent.providers import (ProviderUnavailable, build_model,  # noqa: E402
                                 ollama_modelfile, ollama_status)
from src.agent.settings import AgentSettings  # noqa: E402


def test_gemini_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Gemini API key'):
        build_model(AgentSettings(), key_lookup=lambda s: None)


def test_gemini_model_built_with_key():
    model = build_model(AgentSettings(), key_lookup=lambda s: 'k')
    assert 'gemini-3.5-flash-lite' in model.model_name


def test_ollama_model_uses_openai_endpoint():
    model = build_model(AgentSettings(provider='ollama'))
    assert model.model_name == 'fs2d-qwen3-4b'


def test_ollama_status_unreachable(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError('down')
    monkeypatch.setattr(httpx, 'get', boom)
    ok, message = ollama_status(AgentSettings(provider='ollama'))
    assert not ok and 'not running' in message


def test_ollama_status_model_missing(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None
        def json(self):
            return {'models': [{'name': 'qwen3:4b'}]}
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: Resp())
    ok, message = ollama_status(AgentSettings(provider='ollama'))
    assert not ok and 'fs2d-qwen3-4b' in message


def test_modelfile_carries_limits():
    text = ollama_modelfile(AgentSettings())
    assert 'PARAMETER num_thread 8' in text and 'PARAMETER num_ctx 8192' in text


def test_ollama_status_not_ollama(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None
        def json(self):
            raise ValueError('not json')
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: Resp())
    ok, message = ollama_status(AgentSettings(provider='ollama'))
    assert not ok and 'did not answer like Ollama' in message
