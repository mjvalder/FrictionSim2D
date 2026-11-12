"""Merge a trained LoRA, convert to GGUF, register it in Ollama and bundle it (spec §5).

    python tools/agent_data/export.py --run q35-4b --tag q35-4b
    python tools/agent_data/export.py --run q35-4b --tag q35-4b \\
        --bundle-only --eval tests/agent/evals/.results/ollama-fs2d-assistant_q35-4b-*.json

Writes ``runs/<run>/gguf/`` (Q4_K_M) and a Modelfile carrying temperature 0
and the tool-schema hash the model was trained for, then runs
``ollama create fs2d-assistant:<tag>``. Run with the fs2d-train env; Ollama
must be on ``$OLLAMA_BIN`` (default ``~/opt/ollama/bin/ollama``).

Either way it also writes the portable bundle ``models/<tag>/`` (weights,
relative-FROM Modelfile, manifest; `src/agent/bundles.py`), which any machine
installs with ``ollama create`` or the Assistant's Install button.
``--bundle-only`` makes just the bundle, from a run already exported: no
conversion, no Ollama, so it works on a machine without the training stack.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import compare  # noqa: E402
from src.agent.bundles import MODELS_DIR, write_bundle  # noqa: E402

RUNS = HERE / 'runs'
OLLAMA_BIN = os.environ.get('OLLAMA_BIN', str(Path.home() / 'opt' / 'ollama' / 'bin' / 'ollama'))

#: The stock Ollama model whose chat template (tool calls, thinking switch) the
#: fine-tune keeps, so it is prompted exactly like the base model.
OLLAMA_BASE = {'unsloth/Qwen3-1.7B': 'qwen3:1.7b', 'unsloth/Qwen3-4B': 'qwen3:4b',
               'unsloth/Qwen3.5-2B': 'qwen3.5:2b', 'unsloth/Qwen3.5-4B': 'qwen3.5:4b',
               'unsloth/Qwen3.5-9B': 'qwen3.5:9b'}

#: The stock qwen3:4b template opens every reply with ``<think>`` (qwen3:1.7b's
#: does not). Training (thinking off) put the empty think block only before final
#: answers, never before a tool call, so that opening steers the fine-tune to
#: answer instead of calling a tool: 4B 46/73 with it, 58/73 without
#: (2026-10-31). The reply opens bare, as trained. The thinking switch is cut
#: out too (``$.IsThinkSet`` -> false): set off, qwen3:1.7b's template appends
#: ``/no_think`` to the request and an empty think block to the reply, neither of
#: which training had, and the runtime sends every local model "thinking off".
OPEN_THINK = '<|im_start|>assistant\n<think>\n{{ end }}'
OPEN_BARE = '<|im_start|>assistant\n{{ end }}'

MODELFILE = """FROM {gguf}
{template}
PARAMETER temperature 0
PARAMETER num_ctx 16384
# fs2d-assistant trained for tool-schema hash {schema_hash}
# (knowledge {knowledge_hash}; run {run}; base {base})
"""


def find_gguf(run: Path, quant: str) -> List[Path]:
    """The run's quantised weights, never the multimodal projector."""
    return [p for p in sorted(run.rglob(f'*{quant.upper()}*.gguf')) if 'mmproj' not in p.name]


def template_lines(modelfile: str) -> str:
    """A Modelfile's template part: everything but FROM, PARAMETER and comments.

    Lines inside a ``TEMPLATE \"\"\"...\"\"\"`` block are kept whole: Qwen3's
    template has a ``# Tools`` line that is not a comment.
    """
    keep, inside = [], False
    for line in modelfile.splitlines():
        if inside or not line.startswith(('FROM', 'PARAMETER', '#')):
            keep.append(line)
        if line.count('"""') % 2:
            inside = not inside
    return '\n'.join(keep).strip()


def eval_scores(paths: Sequence[Path], name: str) -> dict:
    """pass@1 and median warm latency of ``name`` in the benchmark results."""
    runs = [r for (_, model), records in compare.load(paths).items()
            if model == name for r in records]
    if not runs:
        raise SystemExit(f'no results for {name} in {", ".join(map(str, paths))}')
    score = compare.score(name, runs)
    return {'pass1': score['pass1'], 'latency_p50': score['latency_p50']}


def _convert(run: Path, quant: str) -> List[Path]:
    """Merge the adapter and write the GGUF; Unsloth appends '_gguf' to the folder."""
    from unsloth import FastLanguageModel  # pylint: disable=import-outside-toplevel
    model, tokenizer = FastLanguageModel.from_pretrained(str(run / 'adapter'),
                                                         max_seq_length=8192,
                                                         load_in_4bit=False)
    model.save_pretrained_gguf(str(run / 'gguf'), tokenizer, quantization_method=quant)
    return find_gguf(run, quant)


def _served_template(base: str, env: dict) -> str:
    """The template the stock base model is served with, as the fine-tune needs it."""
    template = subprocess.run([OLLAMA_BIN, 'show', OLLAMA_BASE[base], '--template'],
                              check=True, capture_output=True, text=True, env=env).stdout
    stock = subprocess.run([OLLAMA_BIN, 'show', OLLAMA_BASE[base], '--modelfile'],
                           check=True, capture_output=True, text=True, env=env).stdout
    renderer = [line for line in stock.splitlines() if line.startswith(('RENDERER', 'PARSER'))]
    if renderer:  # Qwen3.5: Ollama's built-in renderer, the HF template's twin
        return '\n'.join(renderer)
    template = template.replace(OPEN_THINK, OPEN_BARE).replace('$.IsThinkSet', 'false')
    return f'TEMPLATE """{template.strip()}"""'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--quant', default='q4_k_m')
    parser.add_argument('--bundle-only', action='store_true',
                        help='only write models/<tag>/ from the run\'s GGUF and Modelfile')
    parser.add_argument('--eval', nargs='+', type=Path, default=[],
                        help='benchmark results JSONs to take pass@1 and p50 latency from')
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    run = RUNS / args.run
    metrics = json.loads((run / 'metrics.json').read_text())
    name = f'fs2d-assistant:{args.tag}'
    scores = (eval_scores(args.eval, name) if args.eval
              else {'pass1': None, 'latency_p50': None})
    ggufs = find_gguf(run, args.quant)
    if args.bundle_only:
        if not ggufs:
            raise SystemExit(f'{run} has no {args.quant.upper()} GGUF; export it first')
        template = template_lines((run / 'Modelfile').read_text())
    else:
        ggufs = ggufs or _convert(run, args.quant)
        env = {**os.environ, 'OLLAMA_HOST': os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}
        template = _served_template(metrics['base'], env)
        modelfile = run / 'Modelfile'
        modelfile.write_text(MODELFILE.format(gguf=ggufs[0], template=template,
                                              run=args.run, **{
            k: metrics[k] for k in ('schema_hash', 'knowledge_hash', 'base')}))
        subprocess.run([OLLAMA_BIN, 'create', name, '-f', str(modelfile)], check=True, env=env)
        print(f'{name} <- {ggufs[0]}')
    bundle = write_bundle(args.models_dir / args.tag, gguf=ggufs[0], template=template,
                          manifest={'tag': args.tag, 'base': metrics['base'],
                                    'schema_hash': metrics['schema_hash'],
                                    'knowledge_hash': metrics['knowledge_hash'],
                                    **scores, 'registry': None})
    print(f'bundle {bundle}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
