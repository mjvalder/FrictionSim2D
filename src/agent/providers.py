"""Building the chat model for the configured provider.

pydantic-ai is imported inside `build_model` only, so the pane can show
provider status without it.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import httpx

from .settings import AgentSettings, gemini_key

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


def ollama_start_command(settings: AgentSettings,
                         which: Callable[[str], Optional[str]] = shutil.which,
                         home: Optional[Path] = None) -> str:
    """The throttled ``ollama serve`` line for the user to run (never run here).

    OLLAMA_IGPU_ENABLE=1 is what lets Ollama use an integrated GPU such as
    the Radeon 780M (Phase 0); OLLAMA_VULKAN is honoured though not listed
    by ``ollama serve --help``.
    """
    env = ('OLLAMA_IGPU_ENABLE=1 OLLAMA_VULKAN=1 OLLAMA_MAX_LOADED_MODELS=1 '
           f'OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE={settings.ollama.keep_alive} '
           f'OLLAMA_CONTEXT_LENGTH={settings.ollama.context_tokens}')
    return f'{env} nice -n 10 {ollama_binary(which, home)} serve'


def ollama_setup_help(settings: AgentSettings,
                      which: Callable[[str], Optional[str]] = shutil.which,
                      home: Optional[Path] = None, start_command: Optional[str] = None) -> str:
    """How to get the local provider going, naming the binary actually found.

    ``start_command`` replaces the `ollama_start_command` line when given.
    """
    binary = ollama_binary(which, home)
    start = start_command or ollama_start_command(settings, which, home)
    return ('Ollama is the experimental offline option (Phase 0: about a minute per turn '
            'on a Radeon 780M). The Assistant starts it when needed; to start it '
            'yourself, run\n'
            f'  {start}\n'
            'and fetch the model once with\n'
            f'  {binary} pull {settings.ollama.model}')


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


def ollama_status(settings: AgentSettings, timeout: float = 2.0) -> Tuple[bool, str]:
    """Whether Ollama answers and has the configured model."""
    url = settings.ollama.base_url.rstrip('/') + '/api/tags'
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return False, f'Ollama is not running at {settings.ollama.base_url}.'
    try:
        names = [m.get('name', '') for m in response.json().get('models', [])]
    except (ValueError, AttributeError, TypeError):
        return False, f'Ollama at {settings.ollama.base_url} did not answer like Ollama.'
    if not model_installed(settings.ollama.model, names):
        return False, f'Ollama is running but has no model {settings.ollama.model}.'
    return True, f'Ollama ready ({settings.ollama.model})'
