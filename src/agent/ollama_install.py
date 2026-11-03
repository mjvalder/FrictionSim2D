"""Installing a known fine-tune into the user's Ollama, on their click (never a tool).

From a local bundle: check the weights against the manifest, then ``ollama
create`` (the Modelfile's relative FROM makes any copy work). From a registry
(once a model is published): ``ollama pull`` and ``ollama cp`` to the local name.
Blocking: run it off the GUI thread.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

from .bundles import FineTune, verify
from .providers import ollama_binary
from .settings import AgentSettings

#: Where the CLI redraws its line (cursor to column 1 / erase line): a new line.
_REDRAW = re.compile(r'\x1b\[(?:1G|2K)|[\r\n]+')
#: Any other ANSI control sequence, and the braille spinner frames.
_NOISE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|[\u2800-\u28ff]')


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


def install_finetune(settings: AgentSettings, finetune: FineTune,
                     progress: Callable[[str], None], *,
                     popen: Callable = subprocess.Popen,
                     binary: Optional[str] = None) -> Tuple[bool, str]:
    """Install ``finetune`` as ``finetune.name``; (ok, message). Settings are not touched."""
    binary = binary or ollama_binary()
    env = {**os.environ, 'OLLAMA_HOST': urlparse(settings.ollama.base_url).netloc}
    if finetune.bundle is not None:
        progress('checking model.gguf')
        problem = verify(finetune)
        if problem:
            return False, problem
        steps = [[binary, 'create', finetune.name, '-f', str(finetune.bundle / 'Modelfile')]]
    elif finetune.registry:
        steps = [[binary, 'pull', finetune.registry],
                 [binary, 'cp', finetune.registry, finetune.name]]
    else:
        return False, f'{finetune.name} has neither a bundle nor a registry name'
    for cmd in steps:
        code, last = _run(cmd, env, progress, popen)
        if code != 0:
            return False, last or f'{" ".join(cmd[1:3])} failed (exit {code})'
    return True, f'Installed {finetune.name}'
