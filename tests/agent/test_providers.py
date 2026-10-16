"""Provider construction and the Ollama helpers."""
import httpx
import pytest

pytest.importorskip('pydantic_ai')

from src.agent.providers import (ProviderUnavailable, build_model,  # noqa: E402
                                 ollama_models, ollama_status)
from src.agent.settings import AgentSettings  # noqa: E402


def test_gemini_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Gemini API key'):
        build_model(AgentSettings(), key_lookup=lambda s: None)


def test_gemini_model_built_with_key():
    model = build_model(AgentSettings(), key_lookup=lambda s: 'k')
    assert 'gemini-3.5-flash-lite' in model.model_name


def test_ollama_model_uses_openai_endpoint():
    model = build_model(AgentSettings(provider='ollama'))
    assert model.model_name == 'qwen3:4b'


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
            return {'models': [{'name': 'qwen3:8b'}]}
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: Resp())
    ok, message = ollama_status(AgentSettings(provider='ollama'))
    assert not ok and 'qwen3:4b' in message  # qwen3:8b is not qwen3:4b


def _tags(*names):
    def get(url, timeout=None):
        assert url.endswith('/api/tags')
        return httpx.Response(200, json={'models': [{'name': n} for n in names]},
                              request=httpx.Request('GET', url))
    return get


def test_ollama_models_lists_installed_names():
    names = ollama_models(AgentSettings(), http_get=_tags('qwen3:4b', 'llama3.1:8b'))
    assert names == ['qwen3:4b', 'llama3.1:8b']


def test_ollama_models_none_when_down():
    def down(url, timeout=None):
        raise httpx.ConnectError('refused')
    assert ollama_models(AgentSettings(), http_get=down) is None


def test_ollama_status_ready_with_exact_model(monkeypatch):
    monkeypatch.setattr(httpx, 'get', _tags('qwen3:4b'))
    assert ollama_status(AgentSettings(provider='ollama')) == (True, 'Ollama ready (qwen3:4b)')


def test_untagged_model_matches_latest():
    from src.agent.providers import model_installed
    assert model_installed('mymodel', ['mymodel:latest'])
    assert not model_installed('qwen3:4b', ['qwen3:8b'])


def test_ollama_status_not_ollama(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None
        def json(self):
            raise ValueError('not json')
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: Resp())
    ok, message = ollama_status(AgentSettings(provider='ollama'))
    assert not ok and 'did not answer like Ollama' in message


def test_ollama_binary_prefers_path(tmp_path):
    from src.agent.providers import ollama_binary
    assert ollama_binary(which=lambda name: '/usr/bin/ollama', home=tmp_path) == '/usr/bin/ollama'


def test_ollama_binary_falls_back_to_the_user_install(tmp_path):
    from src.agent.providers import ollama_binary
    binary = tmp_path / '.local' / 'ollama' / 'bin' / 'ollama'
    binary.parent.mkdir(parents=True)
    binary.write_text('')
    assert ollama_binary(which=lambda name: None, home=tmp_path) == str(binary)


def test_ollama_binary_plain_name_when_not_found(tmp_path):
    from src.agent.providers import ollama_binary
    assert ollama_binary(which=lambda name: None, home=tmp_path) == 'ollama'


def test_start_command_is_throttled_and_names_the_binary(tmp_path):
    from src.agent.providers import ollama_start_command
    settings = AgentSettings(provider='ollama')
    command = ollama_start_command(settings, which=lambda name: '/opt/x/ollama', home=tmp_path)
    for part in ('OLLAMA_IGPU_ENABLE=1', 'OLLAMA_VULKAN=1', 'OLLAMA_MAX_LOADED_MODELS=1',
                 'OLLAMA_NUM_PARALLEL=1', 'OLLAMA_KEEP_ALIVE=2m', 'OLLAMA_CONTEXT_LENGTH=8192',
                 'nice -n 10'):
        assert part in command
    assert command.endswith('/opt/x/ollama serve')


def test_setup_help_carries_the_start_command(tmp_path):
    from src.agent.providers import ollama_setup_help
    text = ollama_setup_help(AgentSettings(provider='ollama'), which=lambda name: None,
                             home=tmp_path)
    assert 'OLLAMA_IGPU_ENABLE=1' in text and 'ollama serve' in text
    assert 'ollama pull qwen3:4b' in text
    assert 'create' not in text and 'Modelfile' not in text
