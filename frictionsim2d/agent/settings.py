"""Assistant settings and the cloud API keys, under $XDG_CONFIG_HOME/FrictionSim2D/.

Kept apart from `GlobalSettings` (simulation physics) on purpose: these are a
user's GUI preferences, and the key must never reach settings.yaml, a workspace
or the repo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field

SETTINGS_NAME = 'agent.yaml'
SECRETS_NAME = 'agent.secrets'
#: The default local model: the 4B fine-tune (`FrictionSim2D fetch model`).
OLLAMA_DEFAULT_MODEL = 'fs2d-assistant:qwen3.5-4b'


class OllamaSettings(BaseModel):
    """Local provider: a FrictionSim2D fine-tune by default."""

    base_url: str = 'http://localhost:11434'
    model: str = OLLAMA_DEFAULT_MODEL
    context_tokens: int = Field(16384, ge=2048)
    keep_alive: str = '2m'
    #: A server the pane started is stopped after this long without a turn.
    idle_minutes: int = Field(10, ge=1)
    #: Tokens one reply may generate, so a small model that loops fails fast.
    max_output_tokens: int = Field(4096, ge=64)
    #: Corpus passages found for the user's message and attached to it (0: none).
    prefetch_passages: int = Field(0, ge=0, le=3)


class GeminiSettings(BaseModel):
    """Cloud provider: Gemini through the Google API (the default provider)."""

    model: str = 'gemini-3.5-flash-lite'
    api_key_env: str = 'GEMINI_API_KEY'
    consent_given: bool = False


#: The Claude models the model box offers (any other name can still be typed).
ANTHROPIC_MODELS = ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5')


class AnthropicSettings(BaseModel):
    """Cloud provider: Claude through the Anthropic API (a paid key)."""

    model: str = 'claude-opus-5-5'
    api_key_env: str = 'ANTHROPIC_API_KEY'
    consent_given: bool = False
    #: How hard Claude thinks; lower is faster and cheaper per turn.
    effort: Literal['low', 'medium', 'high', 'xhigh', 'max'] = 'medium'


#: The OpenAI models the model box offers (any other name can still be typed).
OPENAI_MODELS = ('gpt-5.5', 'gpt-5.4-mini', 'gpt-5.4-nano')


class OpenAISettings(BaseModel):
    """Cloud provider: GPT through the OpenAI API (a paid key)."""

    model: str = 'gpt-5.5'
    api_key_env: str = 'OPENAI_API_KEY'
    consent_given: bool = False


class MaterialsProjectSettings(BaseModel):
    """Materials Project / 2DMatPedia lookups: only formulas and ids are sent."""

    enabled: bool = True


class AgentSettings(BaseModel):
    """Everything the Assistant pane remembers between sessions."""

    provider: Literal['gemini', 'ollama', 'anthropic', 'openai'] = 'gemini'
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    materials_project: MaterialsProjectSettings = Field(
        default_factory=MaterialsProjectSettings)
    #: Off: config/settings changes land in the Builder at once (with Undo).
    #: On: they wait as cards, like file operations always do.
    ask_before_apply: bool = False
    #: Model requests one turn may make before it stops to save quota.
    max_requests_per_turn: int = Field(50, ge=1)


def config_dir() -> Path:
    """FrictionSim2D's per-user config directory, honouring XDG_CONFIG_HOME."""
    base = os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
    return Path(base) / 'FrictionSim2D'


def data_dir() -> Path:
    """FrictionSim2D's per-user data directory, honouring XDG_DATA_HOME."""
    base = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(base) / 'FrictionSim2D'


def load_agent_settings(path: Optional[Path] = None) -> AgentSettings:
    """Read agent.yaml; defaults when it does not exist."""
    path = path or config_dir() / SETTINGS_NAME
    if not path.is_file():
        return AgentSettings()
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return AgentSettings.model_validate(data)


def save_agent_settings(settings: AgentSettings, path: Optional[Path] = None) -> Path:
    """Write agent.yaml (never the key)."""
    path = path or config_dir() / SETTINGS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump(), sort_keys=False),
                    encoding='utf-8')
    return path


def api_key(settings: AgentSettings, provider: str,
            secrets_path: Optional[Path] = None) -> Optional[str]:
    """A cloud ``provider``'s key: its environment variable first, then agent.secrets."""
    env = os.environ.get(getattr(settings, provider).api_key_env)
    if env and env.strip():
        return env.strip()
    return _read_secrets(secrets_path or config_dir() / SECRETS_NAME).get(provider)


def _read_secrets(path: Path) -> Dict[str, str]:
    """``provider=key`` lines of agent.secrets; empty keys are left out."""
    if not path.is_file():
        return {}
    keys = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        name, sep, value = line.partition('=')
        if sep and value.strip():
            keys[name.strip()] = value.strip()
    return keys


def store_api_key(key: str, provider: str, secrets_path: Optional[Path] = None) -> Path:
    """Write ``provider``'s key to agent.secrets (owner-only), keeping the others.

    ``fchmod`` also narrows a file that already existed; ``O_NOFOLLOW`` refuses a symlink.
    """
    key = key.strip()
    if '\n' in key or '\r' in key:
        raise ValueError('the API key may not contain a line break')
    path = secrets_path or config_dir() / SECRETS_NAME
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, mode=0o700)
        os.chmod(path.parent, 0o700)  # mkdir's mode is masked by the umask
    keys = {} if path.is_symlink() else _read_secrets(path)  # O_NOFOLLOW refuses it below
    keys[provider] = key
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(''.join(f'{name}={value}\n' for name, value in keys.items()))
    return path
