"""Provider construction and the Ollama helpers."""
import httpx
import pytest

pytest.importorskip('pydantic_ai')

from frictionsim2d.agent.providers import (ProviderUnavailable, build_model,  # noqa: E402
                                 ollama_models, run_settings)
from frictionsim2d.agent.settings import AgentSettings  # noqa: E402


def test_gemini_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Gemini API key'):
        build_model(AgentSettings(), key_lookup=lambda s, p: None)


def test_gemini_model_built_with_key():
    settings = AgentSettings()
    model = build_model(settings, key_lookup=lambda s, p: 'k')
    assert settings.gemini.model in model.model_name


def test_anthropic_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='Anthropic API key'):
        build_model(AgentSettings(provider='anthropic'), key_lookup=lambda s, p: None)


def test_anthropic_model_built_with_its_own_key():
    pytest.importorskip('anthropic')
    asked = []
    settings = AgentSettings(provider='anthropic')
    model = build_model(settings, key_lookup=lambda s, p: asked.append(p) or 'k')
    assert asked == ['anthropic'] and model.model_name == settings.anthropic.model


def test_openai_without_key_is_unavailable():
    with pytest.raises(ProviderUnavailable, match='OpenAI API key'):
        build_model(AgentSettings(provider='openai'), key_lookup=lambda s, p: None)


def test_openai_model_built_with_its_own_key():
    """Built only: no request is sent."""
    from pydantic_ai.models.openai import OpenAIChatModel
    asked = []
    settings = AgentSettings(provider='openai')
    model = build_model(settings, key_lookup=lambda s, p: asked.append(p) or 'sk-test')
    assert asked == ['openai'] and isinstance(model, OpenAIChatModel)
    assert model.model_name == settings.openai.model and 'api.openai.com' in model.base_url


def test_run_settings_per_provider():
    assert run_settings(AgentSettings()) == {'temperature': 0.0}
    assert run_settings(AgentSettings(provider='ollama')) == {
        'temperature': 0.0, 'max_tokens': 4096, 'openai_reasoning_effort': 'none'}
    claude = run_settings(AgentSettings(provider='anthropic'))
    assert 'temperature' not in claude  # current Claude models reject sampling params
    assert claude['anthropic_effort'] == 'medium' and claude['anthropic_cache_instructions']
    assert run_settings(AgentSettings(provider='openai')) == {'max_tokens': 16000}


@pytest.mark.parametrize('model', ['fs2d-assistant:p6-4b-nt', 'qwen3.5:4b'])
def test_every_ollama_model_is_told_not_to_think(model):
    settings = AgentSettings(provider='ollama', ollama={'model': model})
    assert run_settings(settings)['openai_reasoning_effort'] == 'none'


def test_ollama_model_uses_openai_endpoint():
    settings = AgentSettings(provider='ollama')
    assert build_model(settings).model_name == settings.ollama.model


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


def test_untagged_model_matches_latest():
    from frictionsim2d.agent.providers import model_installed
    assert model_installed('mymodel', ['mymodel:latest'])
    assert not model_installed('qwen3:4b', ['qwen3:8b'])


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
    from frictionsim2d.agent.ollama_server import ollama_start_command
    settings = AgentSettings(provider='ollama')
    command = ollama_start_command(settings, which=lambda name: '/opt/x/ollama', home=tmp_path)
    assert command == ('OLLAMA_IGPU_ENABLE=1 OLLAMA_VULKAN=1 OLLAMA_MAX_LOADED_MODELS=1 '
                       'OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=2m OLLAMA_CONTEXT_LENGTH=16384 '
                       'nice -n 10 /opt/x/ollama serve')


def test_ollama_reply_cap_goes_out_as_max_tokens():
    """Ollama ignores max_completion_tokens (checked against 0.x: a cap of 5 gave 1109)."""
    model = build_model(AgentSettings(provider='ollama'))
    assert model.profile.get('openai_chat_supports_max_completion_tokens') is False
