"""Merge a trained LoRA, convert to GGUF and register it in Ollama (spec §5).

    python tools/agent_data/export.py --run smoke-1p7b --tag smoke

Writes ``runs/<run>/gguf/`` (Q4_K_M) and a Modelfile carrying temperature 0
and the tool-schema hash the model was trained for, then runs
``ollama create fs2d-assistant:<tag>``. Run with the fs2d-train env; Ollama
must be on ``$OLLAMA_BIN`` (default ``~/opt/ollama/bin/ollama``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
OLLAMA_BIN = os.environ.get('OLLAMA_BIN', str(Path.home() / 'opt' / 'ollama' / 'bin' / 'ollama'))

#: The stock Ollama model whose chat template (tool calls, thinking switch) the
#: fine-tune keeps, so it is prompted exactly like the base model.
OLLAMA_BASE = {'unsloth/Qwen3-1.7B': 'qwen3:1.7b', 'unsloth/Qwen3-4B': 'qwen3:4b'}

MODELFILE = """FROM {gguf}
TEMPLATE \"\"\"{template}\"\"\"
PARAMETER temperature 0
PARAMETER num_ctx 8192
# fs2d-assistant trained for tool-schema hash {schema_hash}
# (knowledge {knowledge_hash}; run {run}; base {base})
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--quant', default='q4_k_m')
    args = parser.parse_args(argv)

    # pylint: disable=import-outside-toplevel
    from unsloth import FastLanguageModel
    run = RUNS / args.run
    metrics = json.loads((run / 'metrics.json').read_text())
    model, tokenizer = FastLanguageModel.from_pretrained(str(run / 'adapter'),
                                                         max_seq_length=8192,
                                                         load_in_4bit=False)
    gguf_dir = run / 'gguf'
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=args.quant)
    ggufs = sorted(gguf_dir.rglob('*.gguf'), key=lambda p: p.stat().st_size)
    quantised = [p for p in ggufs if args.quant.lower() in p.name.lower()] or ggufs[:1]
    env = {**os.environ, 'OLLAMA_HOST': os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}
    template = subprocess.run([OLLAMA_BIN, 'show', OLLAMA_BASE[metrics['base']], '--template'],
                              check=True, capture_output=True, text=True, env=env).stdout
    modelfile = run / 'Modelfile'
    modelfile.write_text(MODELFILE.format(gguf=quantised[0], template=template.strip(),
                                          run=args.run, **{
        k: metrics[k] for k in ('schema_hash', 'knowledge_hash', 'base')}))
    name = f'fs2d-assistant:{args.tag}'
    subprocess.run([OLLAMA_BIN, 'create', name, '-f', str(modelfile)], check=True, env=env)
    print(f'{name} <- {quantised[0]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
