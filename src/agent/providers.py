"""Building the chat model for the configured provider.

pydantic-ai is imported inside `build_model` only, so the pane can show
provider status without it.
"""
from __future__ import annotations

from typing import Tuple

import httpx

from .settings import AgentSettings, gemini_key

OLLAMA_SETUP_HELP = (
    'Ollama is the experimental offline option (Phase 0: about a minute per turn '
    'on a Radeon 780M). To use it: install Ollama, start it with\n'
    '  OLLAMA_IGPU_ENABLE=1 OLLAMA_VULKAN=1 OLLAMA_MAX_LOADED_MODELS=1 '
    'OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=2m ollama serve\n'
    '(OLLAMA_VULKAN is not listed by `ollama serve --help` but is honoured), then\n'
    '  ollama pull qwen3:4b\n'
    '  ollama create fs2d-qwen3-4b -f <the Modelfile below>')


class ProviderUnavailable(RuntimeError):
    """The chosen provider cannot be used right now; the message says why."""


def build_model(settings: AgentSettings, key_lookup=gemini_key):
    """A pydantic-ai model for ``settings.provider``."""
    if settings.provider == 'gemini':
        key = key_lookup(settings)
        if not key:
            raise ProviderUnavailable('No Gemini API key: set GEMINI_API_KEY or enter '
                                      'it when the Assistant asks.')
        from pydantic_ai.models.google import GoogleModel  # pylint: disable=import-outside-toplevel
        from pydantic_ai.providers.google import GoogleProvider  # pylint: disable=import-outside-toplevel
        return GoogleModel(settings.gemini.model, provider=GoogleProvider(api_key=key))
    from pydantic_ai.models.openai import OpenAIChatModel  # pylint: disable=import-outside-toplevel
    from pydantic_ai.providers.ollama import OllamaProvider  # pylint: disable=import-outside-toplevel
    base = settings.ollama.base_url.rstrip('/') + '/v1'
    return OpenAIChatModel(settings.ollama.model, provider=OllamaProvider(base_url=base))


def ollama_status(settings: AgentSettings, timeout: float = 2.0) -> Tuple[bool, str]:
    """Whether Ollama answers and has the configured model."""
    url = settings.ollama.base_url.rstrip('/') + '/api/tags'
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return False, f'Ollama is not running at {settings.ollama.base_url}.'
    names = {m.get('name', '').split(':')[0] for m in response.json().get('models', [])}
    if settings.ollama.model.split(':')[0] not in names:
        return False, f'Ollama is running but has no model {settings.ollama.model}.'
    return True, f'Ollama ready ({settings.ollama.model}).'


def ollama_modelfile(settings: AgentSettings, base: str = 'qwen3:4b') -> str:
    """A Modelfile that applies the configured thread and context limits."""
    return (f'FROM {base}\n'
            f'PARAMETER num_thread {settings.ollama.max_threads}\n'
            f'PARAMETER num_ctx {settings.ollama.context_tokens}\n'
            'PARAMETER temperature 0\n')
