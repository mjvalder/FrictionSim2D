"""Compare assistant models on the eval suite, as tool-use benchmarks report them.

Reads the result files ``pytest tests/agent/evals -m agent_eval`` writes (one
per run; runs of the same provider/model are pooled, so repeats may be split
across files) and reports per model:

* pass@1 -- mean success over all runs, with a 95 % interval from a bootstrap over
  the cases (repeats of one case are not independent: a local model at temperature
  0 answers all three alike, so an interval over runs would be too narrow);
* pass^k -- share of cases passed in all k repeats (tau-bench's consistency
  metric; k is the smallest repeat count among the pooled cases);
* success per case group;
* median warm latency, mean requests, tokens and list-price cost per task.

    python tools/agent_data/compare.py tests/agent/evals/.results/bench-*.json
    python tools/agent_data/compare.py --latex ...   # rows for the paper's table
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: List prices, USD per million tokens: (input, output, cache read, cache write).
#: Local models cost nothing; the Gemini free tier is billed at zero.
PRICES = {
    'claude-opus-5-5': (4.00, 20.00, 0.20, 5.00),
    'claude-sonnet-5-5': (2.00, 10.00, 0.20, 2.50),
    'claude-haiku-5-5': (0.10, 0.50, 0.01, 0.125),
}
GROUP_ORDER = ('setup', 'analysis', 'domain', 'materials_db', 'materials', 'workspace',
               'declines', 'params', 'sweeps', 'must_ask')


def case_bootstrap(by_case: Dict[str, List[bool]], z: float = 0.95,
                   samples: int = 10000, seed: int = 0) -> tuple:
    """95 % percentile interval of pass@1, resampling cases (with all their runs)."""
    rates = [sum(v) / len(v) for v in by_case.values()]
    if not rates:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(rates, k=len(rates))) for _ in range(samples))
    tail = (1 - z) / 2
    return (means[int(tail * samples)], means[int((1 - tail) * samples) - 1])


def cost(record: dict, model: str) -> Optional[float]:
    """USD for one run at list price; 0 for local models, None when unknown."""
    if record.get('provider') == 'ollama':
        return 0.0
    if model not in PRICES:
        return None
    price_in, price_out, price_read, price_write = PRICES[model]
    read = record.get('cache_read_tokens', 0)
    write = record.get('cache_write_tokens', 0)
    fresh = max(0, record.get('input_tokens', 0) - read - write)
    return (fresh * price_in + read * price_read + write * price_write
            + record.get('output_tokens', 0) * price_out) / 1e6


def load(paths: Sequence[Path]) -> Dict[tuple, List[dict]]:
    """Runs pooled by (provider, model); skipped runs dropped."""
    pooled: Dict[tuple, List[dict]] = defaultdict(list)
    for path in paths:
        data = json.loads(Path(path).read_text())
        for record in data['records']:
            if record['outcome'] != 'skipped':
                pooled[(data['provider'], data['model'])].append(
                    {**record, 'provider': data['provider']})
    return pooled


def score(model: str, runs: List[dict]) -> dict:
    """The benchmark metrics of one model's runs."""
    passed = sum(r['outcome'] == 'passed' for r in runs)
    by_case = defaultdict(list)
    for r in runs:
        by_case[r['case']].append(r['outcome'] == 'passed')
    k = min(len(v) for v in by_case.values())
    groups = defaultdict(lambda: [0, 0])
    for r in runs:
        groups[r['group']][0] += r['outcome'] == 'passed'
        groups[r['group']][1] += 1
    warm = [r['latency_s'] for r in runs if not r.get('cold')]
    costs = [cost(r, model) for r in runs]
    return {'model': model, 'runs': len(runs), 'cases': len(by_case), 'k': k,
            'pass1': passed / len(runs), 'ci': case_bootstrap(by_case),
            'passk': sum(all(v[:k]) for v in by_case.values()) / len(by_case),
            'groups': {g: tuple(v) for g, v in groups.items()},
            'latency_p50': statistics.median(warm) if warm else math.nan,
            'requests': statistics.mean(r['requests'] for r in runs),
            'tokens': statistics.mean(r['input_tokens'] + r['output_tokens'] for r in runs),
            'cost': None if None in costs else statistics.mean(costs)}


def markdown(scores: List[dict]) -> str:
    lines = ['| model | runs | pass@1 (95 % CI) | pass^k | p50 s | tokens/task | $/task |',
             '|---|---|---|---|---|---|---|']
    for s in scores:
        lo, hi = s['ci']
        price = '-' if s['cost'] is None else f"{s['cost']:.4f}"
        lines.append(f"| {s['model']} | {s['runs']} | {s['pass1']:.0%} ({lo:.0%}-{hi:.0%}) "
                     f"| {s['passk']:.0%} (k={s['k']}) | {s['latency_p50']:.1f} "
                     f"| {s['tokens']:.0f} | {price} |")
    lines += ['', '| group | ' + ' | '.join(s['model'] for s in scores) + ' |',
              '|---|' + '---|' * len(scores)]
    for g in GROUP_ORDER:
        cells = [f'{s["groups"][g][0]}/{s["groups"][g][1]}' if g in s['groups'] else '-'
                 for s in scores]
        lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def latex(scores: List[dict]) -> str:
    rows = []
    for s in scores:
        lo, hi = s['ci']
        price = '--' if s['cost'] is None else f"{s['cost'] * 1000:.1f}"
        rows.append(f"{s['model']} & {100 * s['pass1']:.0f} ({100 * lo:.0f}--{100 * hi:.0f}) & "
                    f"{100 * s['passk']:.0f} & {s['latency_p50']:.1f} & "
                    f"{s['tokens'] / 1000:.1f} & {price} \\\\")
    return '\n'.join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--latex', action='store_true', help='table rows for the paper')
    args = parser.parse_args(argv)
    scores = sorted((score(model, runs) for (_, model), runs in load(args.results).items()),
                    key=lambda s: -s['pass1'])
    print(latex(scores) if args.latex else markdown(scores))
    return 0


if __name__ == '__main__':
    sys.exit(main())
