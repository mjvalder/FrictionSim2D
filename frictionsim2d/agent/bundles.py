"""The published assistant fine-tunes and their downloaded bundles.

A bundle is ``model.gguf`` (the Q4_K_M weights), a ``Modelfile`` whose ``FROM``
is relative (so ``ollama create`` works from any copy of it) and a
``manifest.json`` (base model, the tool schemas it fits, the weights' sha256
and size). `fetch` puts one in ``MODELS_DIR/<tag>/`` (`models_dir`) when ``ollama pull`` from
Hugging Face is not enough.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..data.provenance import sha256_file
from .settings import data_dir

logger = logging.getLogger(__name__)

MODEL_PREFIX = 'fs2d-assistant'
GGUF_NAME = 'model.gguf'
#: What a bundle holds, in download order (the manifest first: it names the rest).
BUNDLE_FILES = ('manifest.json', 'Modelfile', GGUF_NAME)

#: The published fine-tunes: tag -> Hugging Face model repository.
PUBLISHED: Dict[str, str] = {
    'qwen3.5-2b': 'mjvalder/fs2d-assistant-qwen3.5-2b',
    'qwen3.5-4b': 'mjvalder/fs2d-assistant-qwen3.5-4b',
    'qwen3.5-9b': 'mjvalder/fs2d-assistant-qwen3.5-9b',
}
#: Tool schemas (`runtime.tool_schema_hash`) each published model was trained or
#: benchmarked on, from its manifest. Update it with every upload that changes them.
PUBLISHED_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    'qwen3.5-2b': ('58d9a9a68b61b3d2',),
    'qwen3.5-4b': ('b8b78921bc6dd497', '58d9a9a68b61b3d2'),
    'qwen3.5-9b': ('b8b78921bc6dd497', '58d9a9a68b61b3d2'),
}

_REQUIRED = ('tag', 'base', 'schema_hash')


def models_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/models``: one bundle folder per tag."""
    return data_dir() / 'models'


def __getattr__(name: str):
    if name == 'MODELS_DIR':  # read at use, so XDG_DATA_HOME set after import counts
        return models_dir()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
_OPTIONAL = ('sha256', 'size', 'verified_schemas')


def model_name(tag: str) -> str:
    """The Ollama model name a fine-tune is installed as."""
    return f'{MODEL_PREFIX}:{tag}'


def published_tag(model: str) -> Optional[str]:
    """The `PUBLISHED` tag of the Ollama name ``model``; None for any other model."""
    prefix = f'{MODEL_PREFIX}:'
    tag = model[len(prefix):] if model.startswith(prefix) else None
    return tag if tag in PUBLISHED else None


def fits_published(tag: str, schema_hash: str) -> bool:
    """Whether the published ``tag`` was trained or benchmarked on ``schema_hash``."""
    return schema_hash in PUBLISHED_SCHEMAS.get(tag, ())


@dataclass(frozen=True)
class FineTune:
    """One downloaded bundle, from its manifest."""

    tag: str
    base: str
    schema_hash: str
    #: Tool schemas besides ``schema_hash`` the model was benchmarked on and passed.
    verified_schemas: Tuple[str, ...] = ()
    sha256: Optional[str] = None
    size: Optional[int] = None
    bundle: Optional[Path] = None

    def fits(self, schema_hash: str) -> bool:
        """Whether the model was trained or benchmarked on the tool schema ``schema_hash``."""
        return schema_hash == self.schema_hash or schema_hash in self.verified_schemas

    @property
    def name(self) -> str:
        """The Ollama model name it is installed as."""
        return model_name(self.tag)


def read_manifest(folder: Path) -> Optional[FineTune]:
    """The bundle in ``folder``; None (and a warning) when its manifest is unusable."""
    try:
        data = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        fields = {k: data[k] for k in _REQUIRED}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning('Skipping model bundle %s: unreadable manifest (%s)', folder, exc)
        return None
    fields.update({k: data[k] for k in _OPTIONAL if data.get(k) is not None})
    if 'verified_schemas' in fields:
        fields['verified_schemas'] = tuple(fields['verified_schemas'])
    return FineTune(**fields, bundle=folder)


def verify(finetune: FineTune) -> Optional[str]:
    """None when the bundle's weights match its manifest, else why not."""
    path = finetune.bundle / GGUF_NAME if finetune.bundle else None
    ok = path is not None and path.is_file()
    if ok and finetune.size is not None:
        ok = path.stat().st_size == finetune.size
    if ok and finetune.sha256 is not None:
        ok = sha256_file(path) == finetune.sha256
    return None if ok else f'{GGUF_NAME} does not match its manifest'
