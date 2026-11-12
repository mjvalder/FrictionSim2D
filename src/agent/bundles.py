"""Portable fine-tune bundles: ``tools/agent_data/models/<tag>/``.

A bundle is ``model.gguf`` (the Q4_K_M weights), a ``Modelfile`` whose ``FROM``
is relative (so ``ollama create`` works from any copy of it) and a
``manifest.json`` saying what the model was trained for. `export.py` writes
bundles; the Assistant pane reads them to offer Install.

`PUBLISHED` is for models pushed to a registry (none yet): an entry there is
installable with ``ollama pull`` on a machine without the bundle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / 'tools' / 'agent_data' / 'models'
MODEL_PREFIX = 'fs2d-assistant'
GGUF_NAME = 'model.gguf'

MODELFILE = """FROM ./{gguf}
{template}
PARAMETER temperature 0
PARAMETER num_ctx 16384
"""

_REQUIRED = ('tag', 'base', 'schema_hash', 'knowledge_hash')
_OPTIONAL = ('sha256', 'size', 'pass1', 'latency_p50', 'registry')


@dataclass(frozen=True)
class FineTune:  # pylint: disable=too-many-instance-attributes
    """One fine-tuned assistant model, from its manifest or `PUBLISHED`."""

    tag: str
    base: str
    schema_hash: str
    knowledge_hash: str
    sha256: Optional[str] = None
    size: Optional[int] = None
    pass1: Optional[float] = None
    latency_p50: Optional[float] = None
    registry: Optional[str] = None
    bundle: Optional[Path] = None

    @property
    def name(self) -> str:
        """The Ollama model name it is installed as."""
        return f'{MODEL_PREFIX}:{self.tag}'

    @property
    def installable(self) -> bool:
        """A registry name to pull, or a bundle whose weights are here."""
        return bool(self.registry) or (self.bundle is not None
                                       and (self.bundle / GGUF_NAME).is_file())


#: Registry-only fine-tunes, by tag (nothing is published yet).
PUBLISHED: Dict[str, FineTune] = {}


def sha256_file(path: Path) -> str:
    """Hex sha256 of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def render_modelfile(template: str) -> str:
    """The bundle's Modelfile around ``template`` (RENDERER/PARSER or TEMPLATE lines)."""
    return MODELFILE.format(gguf=GGUF_NAME, template=template.strip())


def write_bundle(dest: Path, *, gguf: Path, template: str, manifest: dict) -> Path:
    """Write ``dest/`` with the weights (hard-linked when possible), Modelfile, manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / GGUF_NAME
    target.unlink(missing_ok=True)
    try:
        os.link(gguf, target)
    except OSError:
        shutil.copy2(gguf, target)
    (dest / 'Modelfile').write_text(render_modelfile(template), encoding='utf-8')
    data = {**manifest, 'sha256': sha256_file(target), 'size': target.stat().st_size}
    (dest / 'manifest.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return dest


def _read(folder: Path) -> Optional[FineTune]:
    try:
        data = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        fields = {k: data[k] for k in _REQUIRED}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning('Skipping model bundle %s: unreadable manifest (%s)', folder, exc)
        return None
    fields.update({k: data[k] for k in _OPTIONAL if data.get(k) is not None})
    return FineTune(**fields, bundle=folder)


def catalogue(models_dir: Path = MODELS_DIR) -> Dict[str, FineTune]:
    """Known fine-tunes by Ollama name: `PUBLISHED`, then the local bundles over them."""
    found = {tune.name: tune for tune in PUBLISHED.values()}
    folders = sorted(models_dir.iterdir()) if models_dir.is_dir() else []
    for folder in folders:
        if not (folder / 'manifest.json').is_file():
            continue
        tune = _read(folder)
        if tune is None:
            continue
        published = found.get(tune.name)
        if published is not None and published.registry:
            tune = replace(tune, registry=published.registry)
        found[tune.name] = tune
    return found


def verify(finetune: FineTune) -> Optional[str]:
    """None when the bundle's weights match its manifest, else why not."""
    path = finetune.bundle / GGUF_NAME if finetune.bundle else None
    ok = path is not None and path.is_file()
    if ok and finetune.size is not None:
        ok = path.stat().st_size == finetune.size
    if ok and finetune.sha256 is not None:
        ok = sha256_file(path) == finetune.sha256
    return None if ok else f'{GGUF_NAME} does not match its manifest (still syncing?)'
