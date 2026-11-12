"""Re-render generated conversations for the prompt a local model now gets.

Conversations generated before the knowledge files were part of every model's
prompt (p8 and earlier) carry the role and rules alone; ``--knowledge`` rewrites
their system message to the current prompt. ``--prefetch`` attaches corpus
passages to each request (``ollama.prefetch_passages``). The teacher is not called.

    python tools/agent_data/augment.py data/p5-shard*.jsonl --name p6-aug --knowledge --prefetch 2

Writes ``data/<name>.jsonl`` and a card copied from the first input's card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.knowledge_index import load_knowledge  # noqa: E402
from src.agent.prompts import build_system_prompt  # noqa: E402
from src.agent.tools.knowledge import attach_passages  # noqa: E402

DATA = HERE / 'data'


def augment(record: dict, system: Optional[str], index, prefetch: int) -> dict:
    """``record`` with its system message replaced and passages attached to each request."""
    messages = []
    for message in record['messages']:
        if message['role'] == 'system' and system is not None:
            message = {**message, 'content': system}
        elif message['role'] == 'user' and prefetch:
            message = {**message, 'content': attach_passages(message['content'], index,
                                                             prefetch)}
        messages.append(message)
    return {**record, 'messages': messages}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--name', required=True)
    parser.add_argument('--knowledge', action='store_true',
                        help='system prompt with the knowledge files')
    parser.add_argument('--prefetch', type=int, default=0, help='passages per request')
    args = parser.parse_args(argv)
    system = build_system_prompt('ollama') if args.knowledge else None
    index = load_knowledge() if args.prefetch else None
    if args.prefetch and index is None:
        raise SystemExit('no knowledge corpus installed')
    out = DATA / f'{args.name}.jsonl'
    count = 0
    with out.open('w', encoding='utf-8') as sink:
        for path in args.inputs:
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    sink.write(json.dumps(augment(json.loads(line), system, index,
                                                  args.prefetch), ensure_ascii=False) + '\n')
                    count += 1
    card = json.loads(args.inputs[0].with_suffix('.card.json').read_text())
    card.update(name=args.name, conversations=count,
                augmented={'from': [p.name for p in args.inputs],
                           'knowledge': args.knowledge, 'prefetch': args.prefetch})
    (DATA / f'{args.name}.card.json').write_text(json.dumps(card, indent=1))
    print(f'{count} conversations -> {out.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
