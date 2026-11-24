"""Fetching a published fine-tune into the user's Ollama, on their request (never a tool).

1. ``ollama pull hf.co/<repo>``, ``ollama cp`` to ``fs2d-assistant:<tag>``, then
   ``ollama show --modelfile`` must still hold the `SERVING` lines: a pull builds
   the chat template from the GGUF and may drop the qwen3.5 renderer and parser.
2. Otherwise (the pull failed, or lost those lines): ``hf_hub_download`` of the
   bundle into ``MODELS_DIR/<tag>/``, `bundles.verify`, then ``ollama create``
   from its Modelfile.

Ollama must be running. Blocking: run it off the GUI thread.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from . import bundles
from .bundles import BUNDLE_FILES, PUBLISHED, model_name, read_manifest, verify
from .providers import model_installed, ollama_binary, ollama_models
from .settings import AgentSettings

#: Modelfile lines the fine-tunes are served with, as ``ollama show --modelfile`` prints them.
SERVING = ('RENDERER qwen3.5', 'PARSER qwen3.5', 'PARAMETER temperature 0',
           'PARAMETER num_ctx 16384')
#: Where the CLI redraws its line (cursor to column 1 / erase line): a new line.
_REDRAW = re.compile(r'\x1b\[(?:1G|2K)|[\r\n]+')
#: Any other ANSI control sequence, and the braille spinner frames.
_NOISE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|[⠀-⣿]')


class FetchError(RuntimeError):
    """The model could not be fetched or installed; the message says why."""


def _run(cmd: List[str], env: dict, progress: Callable[[str], None],
         popen: Callable) -> Tuple[int, str]:
    """Run ``cmd``, passing each new output line on; (exit code, last line).

    The CLI draws a spinner with ANSI codes even into a pipe: each redraw is
    a line, the codes and spinner frames go, and a repeated line is dropped.
    """
    process = popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
    last = ''
    for chunk in process.stdout:
        for line in _REDRAW.split(chunk):
            line = _NOISE.sub('', line).strip()
            if line and line != last:
                last = line
                progress(line)
    return process.wait(), last


def _output(cmd: List[str], env: dict, popen: Callable) -> Tuple[int, str]:
    """Run ``cmd`` quietly; (exit code, its whole output)."""
    process = popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
    text = ''.join(process.stdout)
    return process.wait(), text


def serves_as_trained(modelfile: str) -> bool:
    """Whether a Modelfile holds every `SERVING` line (case and spacing aside)."""
    lines = {' '.join(line.lower().split()) for line in modelfile.splitlines()}
    return all(want.lower() in lines for want in SERVING)


def _published(tag: str) -> str:
    try:
        return PUBLISHED[tag]
    except KeyError:
        raise FetchError(f'unknown model {tag!r}; one of: {", ".join(PUBLISHED)}') from None


class _Ollama:
    """The ollama CLI against the configured server."""

    def __init__(self, settings: AgentSettings, progress: Callable[[str], None],
                 popen: Callable, binary: Optional[str]) -> None:
        self.binary = binary or ollama_binary()
        self.env = {**os.environ, 'OLLAMA_HOST': urlparse(settings.ollama.base_url).netloc}
        self.progress, self.popen = progress, popen

    def run(self, *args: str) -> Tuple[int, str]:
        """Run ``ollama <args>`` with its output as progress; (exit code, last line)."""
        return _run([self.binary, *args], self.env, self.progress, self.popen)

    def output(self, *args: str) -> Tuple[int, str]:
        """Run ``ollama <args>`` quietly; (exit code, output)."""
        return _output([self.binary, *args], self.env, self.popen)


def _pull(ollama: _Ollama, repo: str, name: str) -> bool:
    """Path 1; whether ``name`` is now installed and serves as trained."""
    source = f'hf.co/{repo}'
    code, last = ollama.run('pull', source)
    if code != 0:
        ollama.progress(f'ollama pull failed ({last or f"exit {code}"}); downloading the bundle')
        return False
    code, _ = ollama.run('cp', source, name)
    ollama.output('rm', source)  # the hf.co name only duplicates ``name``
    if code != 0:
        return False
    code, modelfile = ollama.output('show', '--modelfile', name)
    if code == 0 and serves_as_trained(modelfile):
        return True
    ollama.progress('the pulled model lacks the qwen3.5 renderer and parser; '
                    'downloading the bundle')
    return False


def _download(tag: str, repo: str, folder: Path, progress: Callable[[str], None],
              hf_hub_download: Optional[Callable]) -> bundles.FineTune:
    """Path 2's download into ``folder``, verified; a failed one leaves no folder."""
    existing = read_manifest(folder) if (folder / 'manifest.json').is_file() else None
    if existing is not None and existing.tag == tag and existing.sha256:
        progress(f'checking the downloaded {bundles.GGUF_NAME}')
        if verify(existing) is None:
            return existing
    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise FetchError('downloading the model needs huggingface_hub: '
                             'pip install -e ".[agent]"') from exc
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        for filename in BUNDLE_FILES:
            progress(f'downloading {filename} from huggingface.co/{repo}')
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(folder))
        tune = read_manifest(folder)
        if tune is None or tune.tag != tag:
            raise FetchError(f'huggingface.co/{repo} holds no manifest for {tag}')
        if tune.sha256 is None or tune.size is None:
            raise FetchError(f'the manifest of {tag} gives no sha256 and size to check')
        progress(f'checking {bundles.GGUF_NAME}')
        problem = verify(tune)
        if problem:
            raise FetchError(problem)
    except FetchError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise FetchError(f'download from huggingface.co/{repo} failed: {exc}') from exc
    return tune


def fetch_model(tag: str, progress: Callable[[str], None], *,
                settings: Optional[AgentSettings] = None,
                popen: Callable = subprocess.Popen,
                hf_hub_download: Optional[Callable] = None,
                binary: Optional[str] = None,
                models_dir: Optional[Path] = None) -> str:
    """Install the published ``tag`` into Ollama; the local model name.

    Raises `FetchError` when neither path works. Settings are not touched.
    """
    repo, name = _published(tag), model_name(tag)
    ollama = _Ollama(settings or AgentSettings(), progress, popen, binary)
    if _pull(ollama, repo, name):
        return name
    folder = (models_dir or bundles.models_dir()) / tag
    tune = _download(tag, repo, folder, progress, hf_hub_download)
    code, last = ollama.run('create', name, '-f', str(tune.bundle / 'Modelfile'))
    if code != 0:
        raise FetchError(last or f'ollama create failed (exit {code})')
    return name


def available(settings: Optional[AgentSettings] = None, *,
              models: Optional[Iterable[str]] = None,
              models_dir: Optional[Path] = None) -> Dict[str, dict]:
    """Every published tag: ``{repo, installed, size}``.

    ``installed``: Ollama lists it (``models``, else asked; down counts as no).
    ``size``: bytes of a downloaded bundle's weights, else None.
    """
    if models is None:
        models = ollama_models(settings or AgentSettings()) or []
    names, base = list(models), models_dir or bundles.models_dir()
    found = {}
    for tag, repo in PUBLISHED.items():
        folder = base / tag
        tune = read_manifest(folder) if (folder / 'manifest.json').is_file() else None
        found[tag] = {'repo': repo, 'installed': model_installed(model_name(tag), names),
                      'size': tune.size if tune is not None else None}
    return found


def remove_model(tag: str, *, settings: Optional[AgentSettings] = None,
                 popen: Callable = subprocess.Popen, binary: Optional[str] = None,
                 models_dir: Optional[Path] = None) -> None:
    """``ollama rm`` the published ``tag`` and delete its downloaded bundle."""
    _published(tag)
    ollama = _Ollama(settings or AgentSettings(), lambda line: None, popen, binary)
    code, text = ollama.output('rm', model_name(tag))
    if code != 0 and 'not found' not in text.lower():
        raise FetchError(text.strip() or f'ollama rm failed (exit {code})')
    shutil.rmtree((models_dir or bundles.models_dir()) / tag, ignore_errors=True)
