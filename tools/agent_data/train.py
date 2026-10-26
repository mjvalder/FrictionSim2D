"""LoRA fine-tune of a Qwen3 student on generated conversations (spec §5).

Renders each conversation with the student's own chat template (tools included,
thinking off) and trains a LoRA with Unsloth. The loss falls on assistant
messages only -- tool calls and replies -- and never on one marked
``train: false`` (an injected mistake kept for the error that follows it).
Records throughput, peak VRAM and the dataset/schema hashes in
``runs/<run>/metrics.json``. Export to GGUF + Ollama is ``export.py``.

    python tools/agent_data/train.py --data data/smoke.jsonl --run smoke-1p7b
    python tools/agent_data/train.py --data data/smoke.jsonl --run x --render-only

Run with the fs2d-train env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
BASES = {'1.7b': 'unsloth/Qwen3-1.7B', '4b': 'unsloth/Qwen3-4B'}
#: One rendered assistant message: its header, then the trained text through <|im_end|>.
_ASSISTANT = re.compile(r'<\|im_start\|>assistant\n(.*?<\|im_end\|>)', re.DOTALL)
IGNORE = -100


def load(path: Path) -> List[dict]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def render(tokenizer, record: dict, tools: List[dict]) -> str:
    """One conversation as training text in the student's chat format."""
    return tokenizer.apply_chat_template(record['messages'], tools=tools, tokenize=False,
                                         enable_thinking=False)


def trained_spans(text: str, record: dict) -> List[Tuple[int, int]]:
    """Character spans the loss covers: each assistant message not marked
    ``train: false``. The template renders assistant messages in order, one block each."""
    flags = [m.get('train', True) for m in record['messages'] if m['role'] == 'assistant']
    blocks = list(_ASSISTANT.finditer(text))
    if len(blocks) != len(flags):
        raise ValueError(f"{record.get('id')}: {len(blocks)} assistant blocks for "
                         f'{len(flags)} assistant messages')
    return [block.span(1) for block, keep in zip(blocks, flags) if keep]


def tokenize(tokenizer, text: str, spans: List[Tuple[int, int]], max_len: int) -> dict:
    """input_ids and labels: a token counts only when it lies inside a trained span."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=True, max_length=max_len)
    labels = []
    for token, (start, end) in zip(encoded['input_ids'], encoded['offset_mapping']):
        inside = any(lo <= start and end <= hi for lo, hi in spans) and end > start
        labels.append(token if inside else IGNORE)
    return {'input_ids': encoded['input_ids'], 'labels': labels}


def show_mask(tokenizer, example: dict) -> str:
    """The text with trained stretches in [[ ]], to check the mask by eye."""
    out, trained = [], False
    for token, label in zip(example['input_ids'], example['labels']):
        now = label != IGNORE
        if now != trained:
            out.append('[[' if now else ']]')
            trained = now
        out.append(tokenizer.decode([token]))
    return ''.join(out) + (']]' if trained else '')


def main(argv: Optional[List[str]] = None) -> int:  # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, nargs='+', required=True,
                        help='one or more generator outputs (shards of one build)')
    parser.add_argument('--run', required=True)
    parser.add_argument('--base', choices=sorted(BASES), default='1.7b')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-len', type=int, default=8192)  # ~5k: 28 tool schemas
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--four-bit', action='store_true', help='QLoRA (use for 4b)')
    parser.add_argument('--render-only', action='store_true',
                        help='print the first rendered conversation and stop')
    args = parser.parse_args(argv)

    # pylint: disable=import-outside-toplevel
    from unsloth import FastLanguageModel  # first: it patches transformers
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    cards = [json.loads(path.with_suffix('.card.json').read_text()) for path in args.data]
    if len({c['schema_hash'] for c in cards}) != 1:
        raise SystemExit('the shards were generated for different tool schemas')
    card = cards[0]
    records = [r for path in args.data for r in load(path)]
    model, tokenizer = FastLanguageModel.from_pretrained(
        BASES[args.base], max_seq_length=args.max_len, load_in_4bit=args.four_bit,
        dtype=None if args.four_bit else torch.bfloat16)
    texts = [render(tokenizer, r, card['tools']) for r in records]
    examples = [tokenize(tokenizer, t, trained_spans(t, r), args.max_len)
                for t, r in zip(texts, records)]
    if args.render_only:
        shown = next((i for i, r in enumerate(records)
                      if any(m.get('train') is False for m in r['messages'])), 0)
        print(show_mask(tokenizer, examples[shown])[-6000:])
        return 0
    lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    too_long = sum(n > args.max_len for n in lengths)

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
                        'down_proj'],
        use_gradient_checkpointing='unsloth', random_state=0)
    out = RUNS / args.run
    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(examples),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=IGNORE),
        args=TrainingArguments(output_dir=str(out), per_device_train_batch_size=args.batch,
                               gradient_accumulation_steps=args.accum,
                               num_train_epochs=args.epochs, learning_rate=args.lr,
                               lr_scheduler_type='cosine', warmup_ratio=0.05,
                               logging_steps=5, save_strategy='no', bf16=True, seed=0,
                               report_to='none', remove_unused_columns=False))
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    stats = trainer.train()
    seconds = time.time() - start
    model.save_pretrained(str(out / 'adapter'))
    tokenizer.save_pretrained(str(out / 'adapter'))
    trained_tokens = sum(min(n, args.max_len) for n in lengths) * args.epochs
    metrics = {
        'run': args.run, 'base': BASES[args.base], 'four_bit': args.four_bit,
        'rank': args.rank, 'epochs': args.epochs, 'lr': args.lr,
        'conversations': len(texts), 'too_long': too_long,
        'tokens_per_epoch': sum(lengths), 'seconds': round(seconds, 1),
        'tokens_per_second': round(trained_tokens / seconds),
        'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        'final_loss': stats.training_loss,
        'data': [str(p) for p in args.data],
        'data_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in args.data))
        .hexdigest()[:16], 'schema_hash': card['schema_hash'],
        'knowledge_hash': card['knowledge_hash'],
        'log': trainer.state.log_history}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'log'}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
