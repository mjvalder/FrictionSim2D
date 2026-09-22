"""No central-DB credential ever lands in a run tree (final review I1).

`provenance/settings.yaml` and `provenance/config.json` are stored verbatim in
the AiiDA repository and travel in archives and export bundles, so a password
or API key written there leaks to everyone the set is shared with.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config import GlobalSettings, redact_secrets
from src.run.artifacts import generate_package

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

PASSWORD = 'hunter2-central-db-password'
API_KEY = 'sk-central-api-key-0123456789'


def test_redact_secrets_blanks_only_the_credentials():
    settings = GlobalSettings()
    settings.database.password = PASSWORD
    settings.database.api_key = API_KEY
    settings.database.host = 'db.example.org'

    safe = redact_secrets(settings)

    assert safe.database.password == ''
    assert safe.database.api_key == ''
    assert safe.database.host == 'db.example.org'
    assert settings.database.password == PASSWORD  # the original is untouched


def test_generated_tree_carries_no_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    settings_file = tmp_path / 'settings.yaml'
    settings_file.write_text(yaml.safe_dump({'database': {
        'password': PASSWORD, 'api_key': API_KEY, 'host': 'db.example.org'}}),
        encoding='utf-8')

    result = generate_package(EXAMPLES / 'indent_config.ini', 'indent',
                              output_root=tmp_path / 'out',
                              settings_file=settings_file)
    assert result.ok, result.render()

    written = [p for p in result.simulation_root.rglob('*')
               if p.is_file() and p.suffix in ('.yaml', '.json')]
    assert any(p.name == 'settings.yaml' for p in written)
    assert any(p.name == 'config.json' for p in written)
    for path in written:
        text = path.read_text(encoding='utf-8')
        assert PASSWORD not in text, path
        assert API_KEY not in text, path
    stored = yaml.safe_load(next(p for p in written if p.name == 'settings.yaml')
                            .read_text(encoding='utf-8'))
    assert stored['database']['host'] == 'db.example.org'
