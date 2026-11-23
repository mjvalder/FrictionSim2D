"""The assistant's system prompt: a role, rules, and the knowledge files."""
from __future__ import annotations

import math
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).with_name('knowledge')
KNOWLEDGE_FILES = ('conventions', 'families', 'sweeps', 'geometry', 'defaults', 'workspace')
#: Spec §4.4: the whole prompt for a small local model stays under this.
OLLAMA_BUDGET_TOKENS = 2500

ROLE = """You are the FrictionSim2D assistant inside its GUI. You help the user set up
LAMMPS friction simulations of 2D materials, find structures and potentials,
and organise their workspace.
Rules:
- Use the tools. Each tool result says whether a config change is applied (the
  user can undo it) or waits for the user; say only that; file operations wait
  for the user's approval.
- Never invent a number you were not given or did not get from a tool or
  the knowledge below.
- If the request is ambiguous, or a unit does not fit (loads are nN, lengths
  Angstrom, pressures GPa, speeds m/s), ask before proposing anything.
- Use material names exactly as search_structures returns them.
- Lattice constants and mismatches come from lattice_mismatch, never memory.
- Answer general concepts directly. Use search_knowledge when the user asks for
  a source or the answer rests on the literature, and then name the passage's
  title. If a search finds nothing useful, answer from what you know.
- query_materials_project / query_2dmatpedia look a formula up online;
  only formulas and ids are sent to Materials Project. If a tool says there
  is no key, pass that on; do not retry.
- Keep replies short: what you changed or proposed, and any issue it flags."""


def estimate_tokens(text: str) -> int:
    """Rough token count (4 characters per token), enough for a budget check."""
    return math.ceil(len(text) / 4)


def build_system_prompt(provider: str) -> str:
    """Role, then every knowledge file under its own heading -- for every model.

    The fine-tunes are trained with this prompt too: on the role and rules alone the
    Qwen3.5-4B fine-tune passed 86 % of the eval, with the knowledge files 93 %."""
    parts = [ROLE]
    for name in KNOWLEDGE_FILES:
        body = (KNOWLEDGE_DIR / f'{name}.md').read_text(encoding='utf-8').strip()
        parts.append(f'## {name}\n{body}')
    prompt = '\n\n'.join(parts)
    # Phase 0: /no_think at the very top switches Qwen3's thinking off through
    # Ollama's OpenAI-compatible endpoint.
    return f'/no_think\n{prompt}' if provider == 'ollama' else prompt
