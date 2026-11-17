"""Every command the manual names exists; every command that exists is named."""
from pathlib import Path

import pytest

from src.cli import cli

DOCS = Path(__file__).resolve().parents[2] / 'documentation'
GONE = ['aiida dump', 'db upload ./', '--use-remote', '--hpc-config',
        'active_profile', 'hpc_mode', 'auto_import_results',
        'postprocess_code_label', 'create_provenance', 'ssh_port']


@pytest.mark.parametrize('phrase', GONE)
def test_no_stale_command_or_setting(phrase):
    for path in DOCS.glob('*.md'):
        assert phrase not in path.read_text(encoding='utf-8'), f"{path.name}: {phrase}"


def _subcommand_names(group):
    """Real subcommand names of *group*, from the Click app itself.

    ``--help``'s free-text docstring can wrap onto a 2-space-indented line
    (e.g. the ``db`` group's "query results\\n  contributed by other users."),
    which a regex over the whole ``--help`` output would mistake for a
    command. Reading ``group.commands`` instead of parsing rendered text
    sidesteps that entirely.
    """
    return sorted(group.commands)


def test_every_aiida_subcommand_is_documented():
    names = _subcommand_names(cli.commands['aiida'])
    text = (DOCS / 'commands.md').read_text(encoding='utf-8')

    missing = [n for n in names if f'aiida {n}' not in text]
    assert missing == []


def test_settings_doc_lists_the_real_fields():
    from src.core.config import AiidaSettings

    text = (DOCS / 'settings.md').read_text(encoding='utf-8')
    for field in AiidaSettings.model_fields:
        assert f'{field}:' in text, field
