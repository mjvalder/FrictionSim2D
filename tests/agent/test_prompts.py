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
    for phrase in ('undo', 'file operations wait', 'ask', 'nN', 'Angstrom'):
        assert phrase in prompt


def test_defaults_knowledge_offers_example_values():
    assert 'defaults' in KNOWLEDGE_FILES
    prompt = build_system_prompt('ollama')
    for phrase in ('example config', '2, 5, 10, 20, 30, 50, 70, 85, 100 nN', '10-20 nN',
                   'poor contact', 'suggest_values',
                   '100 x 100', '25 Angstrom', '12 Angstrom', '2 m/s', '300 K', 'confirm'):
        assert phrase in prompt, phrase


def test_materials_project_privacy_rule():
    prompt = build_system_prompt('ollama')
    assert 'only formulas and ids are sent to Materials Project' in prompt


def test_trained_model_gets_the_short_prompt_it_was_trained_on():
    """Fine-tuned models learned the knowledge; they get role + rules only (spec §7)."""
    from src.agent.prompts import ROLE, TRAINED_PREFIX  # pylint: disable=import-outside-toplevel
    prompt = build_system_prompt('ollama', model=f'{TRAINED_PREFIX}:p4')
    assert prompt == f'/no_think\n{ROLE}'
    assert '## conventions' in build_system_prompt('ollama', model='qwen3:4b')
    assert '## conventions' in build_system_prompt('ollama')
