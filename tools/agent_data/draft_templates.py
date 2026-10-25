"""Have the local teacher draft user-request templates per category (source B).

For each category in ``categories.py`` the teacher (an Apache-2.0 Qwen3 model in
the local Ollama) writes varied requests with ``{slot}`` placeholders. Drafts
are checked by code -- every slot used exactly as declared, no unknown slot,
no digits (values come from the slots), sensible length -- deduplicated, and
written to ``sources/templates.draft.yaml`` for the user to review and copy
into ``sources/templates.yaml``. No Claude-written text enters a template.

    python tools/agent_data/draft_templates.py                  # all categories
    python tools/agent_data/draft_templates.py --per-category 40 --only afm_setup
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from categories import BY_NAME, CATEGORIES, SLOTS, Category

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'sources' / 'templates.draft.yaml'
OLLAMA = 'http://127.0.0.1:11434/api/chat'
TEACHER = 'qwen3:14b'
TEACHER_LICENCE = 'Apache-2.0'
_SLOT = re.compile(r'\{([a-z_]+)\}')

INSTRUCTIONS = """You write realistic requests that a materials scientist would type into the
chat assistant of FrictionSim2D, a program that sets up LAMMPS friction
simulations of 2D materials. Write {n} different requests for this situation:

{brief}

Use each of these placeholders, in curly braces, and no others:
{slots}
Never write numbers, units after a placeholder are fine. Vary wording, length,
formality and order: terse commands, polite questions, requests with context
about the user's research goal. Return JSON: {{"requests": ["...", ...]}}"""


def _slot_lines(category: Category) -> str:
    if not category.slots:
        return '(none: the request has no placeholder)'
    return '\n'.join(f'{{{slot}}}: {SLOTS[slot]}' for slot in category.slots)


def ask_teacher(category: Category, n: int, temperature: float = 0.8) -> List[str]:
    """The teacher's raw requests for ``category``."""
    prompt = INSTRUCTIONS.format(n=n, brief=category.brief, slots=_slot_lines(category))
    payload = {'model': TEACHER, 'stream': False, 'format': 'json', 'think': False,
               'options': {'temperature': temperature, 'num_ctx': 4096},
               'messages': [{'role': 'user', 'content': prompt}]}
    request = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        content = json.load(response)['message']['content']
    requests = json.loads(content).get('requests', [])
    return [r for r in requests if isinstance(r, str)]


def valid(text: str, category: Category) -> Optional[str]:
    """None if ``text`` is a usable template for ``category``, else why not."""
    used = _SLOT.findall(text)
    if set(used) != set(category.slots):
        return f'slots {used} != {list(category.slots)}'
    if re.search(r'\d', _SLOT.sub('', text)):
        return 'contains a digit'
    def names(word: str) -> bool:  # at a word start: 'multiple' does not name a tip
        return re.search(rf'\b{re.escape(word)}', text.lower()) is not None
    if category.must_mention and not any(names(w) for w in category.must_mention):
        return f'names none of {category.must_mention}'
    if any(names(w) for w in category.must_not):
        return 'names a simulation family'
    if not 3 <= len(text.split()) <= 60:
        return 'length'
    if '{' in _SLOT.sub('', text) or '}' in _SLOT.sub('', text):
        return 'stray brace'
    return None


def _norm(text: str) -> str:
    return re.sub(r'[^a-z{}_ ]', '', text.lower()).strip()


def draft(category: Category, wanted: int, rounds: int = 4) -> Dict:
    """Up to ``wanted`` distinct valid templates, asking again for what was rejected."""
    kept: Dict[str, str] = {}
    rejected = 0
    reasons: Dict[str, int] = {}
    for _ in range(rounds):
        if len(kept) >= wanted:
            break
        for text in ask_teacher(category, wanted - len(kept) + 5):
            text = ' '.join(text.split())
            why = valid(text, category)
            if why is None:
                kept.setdefault(_norm(text), text)
            else:
                rejected += 1
                reasons[why] = reasons.get(why, 0) + 1
    return {'category': category.name, 'kind': category.kind,
            'slots': list(category.slots), 'rejected': rejected,
            'reasons': reasons,
            'templates': list(kept.values())[:wanted]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--per-category', type=int, default=30)
    parser.add_argument('--only', nargs='*', choices=sorted(BY_NAME))
    parser.add_argument('--rounds', type=int, default=4,
                        help='teacher requests per category, at most')
    args = parser.parse_args(argv)
    chosen = [BY_NAME[n] for n in args.only] if args.only else list(CATEGORIES)
    drafts = []
    for category in chosen:
        result = draft(category, args.per_category, args.rounds)
        drafts.append(result)
        print(f"{category.name}: {len(result['templates'])} kept, "
              f"{result['rejected']} rejected", file=sys.stderr)
    if args.only and OUTPUT.exists():  # redraft some categories, keep the rest
        redrafted = {d['category'] for d in drafts}
        previous = yaml.safe_load(OUTPUT.read_text())['categories']
        drafts = [d for d in previous if d['category'] not in redrafted] + drafts
    OUTPUT.write_text(yaml.safe_dump(
        {'source': 'B', 'teacher': TEACHER, 'teacher_licence': TEACHER_LICENCE,
         'status': 'draft -- review, then copy what you keep into templates.yaml',
         'categories': drafts}, sort_keys=False, allow_unicode=True, width=100))
    print(f'{sum(len(d["templates"]) for d in drafts)} templates -> {OUTPUT.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
