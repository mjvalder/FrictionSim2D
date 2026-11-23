"""Provider construction and the Ollama helpers."""
import httpx
import pytest

pytest.importorskip('pydantic_ai')

from frictionsim2d.agent.providers import (ProviderUnavailable, build_model,  # noqa: E402
                                 ollama_models, ollama_status, run_settings)
from frictionsim2d.agent.settings import AgentSettings  # noqa: E402


def test_gemini_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Gemini API key'):
        build_model(AgentSettings(), key_lookup=lambda s, p: None)


def test_gemini_model_built_with_key():
    model = build_model(AgentSettings(), key_lookup=lambda s, p: 'k')
    assert 'gemini-3.5-flash-lite' in model.model_name


def test_anthropic_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Anthropic API key'):
        build_model(AgentSettings(provider='anthropic'), key_lookup=lambda s, p: None)


def test_anthropic_model_built_with_its_own_key():
    pytest.importorskip('anthropic')
    asked = []
    model = build_model(AgentSettings(provider='anthropic'),
                        key_lookup=lambda s, p: asked.append(p) or 'k')
    assert asked == ['anthropic'] and model.model_name == 'claude-opus-5-5'


def test_run_settings_per_provider():
    assert run_settings(AgentSettings()) == {'temperature': 0.0}
    assert run_settings(AgentSettings(provider='ollama')) == {
        'temperature': 0.0, 'max_tokens': 4096, 'openai_reasoning_effort': 'none'}
    claude = run_settings(AgentSettings(provider='anthropic'))
    assert 'temperature' not in claude  # current Claude models reject sampling params
    assert claude['anthropic_effort'] == 'medium' and claude['anthropic_cache_instructions']


@pytest.mark.parametrize('model', ['fs2d-assistant:p6-4b-nt', 'qwen3.5:4b'])
def test_every_ollama_model_is_told_not_to_think(model):
    settings = AgentSettings(provider='ollama', ollama={'model': model})
    assert run_settings(settings)['openai_reasoning_effort'] == 'none'


def test_ollama_model_uses_openai_endpoint():
    model = build_model(AgentSettings(provider='ollama'))
    assert model.model_name == 'fs2d-assistant:qwen3.5-4b'


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
    assert not ok and 'fs2d-assistant:qwen3.5-4b' in message  # qwen3:8b is not it


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
    monkeypatch.setattr(httpx, 'get', _tags('fs2d-assistant:qwen3.5-4b'))
    assert ollama_status(AgentSettings(provider='ollama')) == (
        True, 'Ollama ready (fs2d-assistant:qwen3.5-4b)')


def test_untagged_model_matches_latest():
    from frictionsim2d.agent.providers import model_installed
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
    from frictionsim2d.agent.providers import ollama_binary
    assert ollama_binary(which=lambda name: '/usr/bin/ollama', home=tmp_path) == '/usr/bin/ollama'


def test_ollama_binary_falls_back_to_the_user_install(tmp_path):
    from frictionsim2d.agent.providers import ollama_binary
    binary = tmp_path / '.local' / 'ollama' / 'bin' / 'ollama'
    binary.parent.mkdir(parents=True)
    binary.write_text('')
    assert ollama_binary(which=lambda name: None, home=tmp_path) == str(binary)


def test_ollama_binary_plain_name_when_not_found(tmp_path):
    from frictionsim2d.agent.providers import ollama_binary
    assert ollama_binary(which=lambda name: None, home=tmp_path) == 'ollama'


def test_start_command_is_throttled_and_names_the_binary(tmp_path):
    from frictionsim2d.agent.providers import ollama_start_command
    settings = AgentSettings(provider='ollama')
    command = ollama_start_command(settings, which=lambda name: '/opt/x/ollama', home=tmp_path)
    for part in ('OLLAMA_IGPU_ENABLE=1', 'OLLAMA_VULKAN=1', 'OLLAMA_MAX_LOADED_MODELS=1',
                 'OLLAMA_NUM_PARALLEL=1', 'OLLAMA_KEEP_ALIVE=2m', 'OLLAMA_CONTEXT_LENGTH=16384',
                 'nice -n 10'):
        assert part in command
    assert command.endswith('/opt/x/ollama serve')


def test_ollama_reply_cap_goes_out_as_max_tokens():
    """Ollama ignores max_completion_tokens (checked against 0.x: a cap of 5 gave 1109)."""
    model = build_model(AgentSettings(provider='ollama'))
    assert model.profile.get('openai_chat_supports_max_completion_tokens') is False
