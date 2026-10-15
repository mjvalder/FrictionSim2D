"""The system prompt: role + knowledge, within the small-model budget."""
from src.agent.prompts import (KNOWLEDGE_FILES, OLLAMA_BUDGET_TOKENS,
                               build_system_prompt, estimate_tokens)
from src.gui.schema.simulation_types import FAMILIES


def test_ollama_prompt_within_budget():
    assert estimate_tokens(build_system_prompt('ollama')) <= OLLAMA_BUDGET_TOKENS


def test_ollama_prompt_disables_thinking_first():
    assert build_system_prompt('ollama').startswith('/no_think')
    assert not build_system_prompt('gemini').startswith('/no_think')


def test_every_knowledge_file_is_included():
    prompt = build_system_prompt('gemini')
    for name in KNOWLEDGE_FILES:
        assert f'## {name}' in prompt


def test_families_file_names_every_model():
    prompt = build_system_prompt('gemini')
    for family in FAMILIES:
        for mode in family.modes:
            assert f'[{mode.model}]' in prompt


def test_rules_present():
    prompt = build_system_prompt('gemini')
    for phrase in ('never claim a change is applied', 'ask', 'nN', 'Angstrom'):
        assert phrase in prompt
