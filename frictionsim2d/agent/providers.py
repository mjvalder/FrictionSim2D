"""Building the chat model for the configured provider.

pydantic-ai is imported inside `build_model` only, so the pane can show
provider status without it.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import httpx

from .settings import AgentSettings, api_key

#: Where the user-local install (no root) puts the binary, under $HOME.
OLLAMA_USER_BINARY = Path('.local') / 'ollama' / 'bin' / 'ollama'


def ollama_binary(which: Callable[[str], Optional[str]] = shutil.which,
                  home: Optional[Path] = None) -> str:
    """The ollama binary: on PATH, else the user-local install, else plain ``ollama``."""
    found = which('ollama')
    if found:
        return found
    local = (home if home is not None else Path.home()) / OLLAMA_USER_BINARY
    return str(local) if local.is_file() else 'ollama'


class ProviderUnavailable(RuntimeError):
    """The chosen provider cannot be used right now; the message says why."""


def build_model(settings: AgentSettings, key_lookup=api_key):
    """A pydantic-ai model for ``settings.provider``."""
    # pylint: disable=import-outside-toplevel
    if settings.provider == 'gemini':
        key = key_lookup(settings, 'gemini')
        if not key:
            raise ProviderUnavailable('No Gemini API key: set GEMINI_API_KEY or enter '
                                      'it when the Assistant asks.')
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleModel(settings.gemini.model, provider=GoogleProvider(api_key=key))
    if settings.provider == 'anthropic':
        key = key_lookup(settings, 'anthropic')
        if not key:
            raise ProviderUnavailable('No Anthropic API key: set ANTHROPIC_API_KEY or '
                                      'enter it when the Assistant asks.')
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
        except ImportError as exc:
            raise ProviderUnavailable('Claude needs the anthropic package: '
                                      'pip install "pydantic-ai-slim[anthropic]"') from exc
        return AnthropicModel(settings.anthropic.model,
                              provider=AnthropicProvider(api_key=key))
    from pydantic_ai.models.openai import OpenAIChatModel
    if settings.provider == 'openai':
        key = key_lookup(settings, 'openai')
        if not key:
            raise ProviderUnavailable('No OpenAI API key: set OPENAI_API_KEY or '
                                      'enter it when the Assistant asks.')
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(settings.openai.model, provider=OpenAIProvider(api_key=key))
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.ollama import OllamaProvider
    base = settings.ollama.base_url.rstrip('/') + '/v1'
    provider = OllamaProvider(base_url=base)
    # Ollama ignores max_completion_tokens and honours only max_tokens, so the
    # reply cap (run_settings) must go out under that name.
    profile = OpenAIModelProfile(**(provider.model_profile(settings.ollama.model) or {}),
                                 openai_chat_supports_max_completion_tokens=False)
    return OpenAIChatModel(settings.ollama.model, provider=provider, profile=profile)


def run_settings(settings: AgentSettings) -> dict:
    """Per-request model settings for ``settings.provider``.

    Temperature 0 for Gemini and Ollama, a cap on Ollama's reply length and no
    thinking for Ollama. Current Claude and GPT models reject sampling
    parameters: Claude gets an effort level and caching of the long system
    prompt and tool list plus the growing conversation (re-sent every step);
    GPT only a reply cap (OpenAI caches long prompts by itself).
    """
    if settings.provider == 'ollama':
        # Thinking off: the fine-tunes were trained without it, and thinking
        # spends the reply cap before the model acts.
        return {'temperature': 0.0, 'max_tokens': settings.ollama.max_output_tokens,
                'openai_reasoning_effort': 'none'}
    if settings.provider == 'openai':
        return {'max_tokens': 16000}
    if settings.provider != 'anthropic':
        return {'temperature': 0.0}
    return {'max_tokens': 16000, 'anthropic_effort': settings.anthropic.effort,
            'anthropic_cache_instructions': True, 'anthropic_cache': True}


def model_installed(model: str, installed: Iterable[str]) -> bool:
    """Whether ``model`` is among ``installed`` (an untagged name means ``:latest``)."""
    names = set(installed)
    return model in names or (':' not in model and f'{model}:latest' in names)


def ollama_models(settings: AgentSettings, http_get: Callable = httpx.get,
                  timeout: float = 2.0) -> Optional[List[str]]:
    """Installed model names from /api/tags; None when Ollama does not answer."""
    url = settings.ollama.base_url.rstrip('/') + '/api/tags'
    try:
        response = http_get(url, timeout=timeout)
        response.raise_for_status()
        return [m.get('name', '') for m in response.json().get('models', [])
                if m.get('name')]
    except httpx.HTTPError:
        return None
    except (ValueError, AttributeError, TypeError):
        return []
