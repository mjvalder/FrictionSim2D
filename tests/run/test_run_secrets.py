"""No central-DB credential ever lands in a run tree (final review I1).

`provenance/settings.yaml` and `provenance/config.json` are stored verbatim in
the AiiDA repository and travel in archives and export bundles, so a password
or API key written there leaks to everyone the set is shared with.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config import GlobalSettings, redact_secrets, redact_secrets_mapping
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


def test_redact_secrets_mapping_blanks_the_flat_shape():
    data = {'thermostat': {'type': 'langevin'},
            'database': {'host': 'db.example.org', 'password': PASSWORD,
                        'api_key': API_KEY}}

    safe = redact_secrets_mapping(data)

    assert safe['database']['password'] == ''
    assert safe['database']['api_key'] == ''
    assert safe['database']['host'] == 'db.example.org'
    assert safe['thermostat']['type'] == 'langevin'
    # the input is not mutated
    assert data['database']['password'] == PASSWORD


def test_redact_secrets_mapping_blanks_the_legacy_nested_shape():
    data = {'database': {
        'active_profile': 'central',
        'local': {'host': 'localhost', 'password': 'local-secret',
                 'api_key': 'local-key'},
        'central': {'host': 'db.example.org', 'password': PASSWORD,
                   'api_key': API_KEY},
    }}

    safe = redact_secrets_mapping(data)

    assert safe['database']['local']['password'] == ''
    assert safe['database']['local']['api_key'] == ''
    assert safe['database']['central']['password'] == ''
    assert safe['database']['central']['api_key'] == ''
    assert safe['database']['local']['host'] == 'localhost'
    assert safe['database']['central']['host'] == 'db.example.org'
    # active_profile names no credential and is left alone
    assert safe['database']['active_profile'] == 'central'


def test_redact_secrets_mapping_recurses_into_a_settings_key():
    """`config.json`'s shape: the whole settings document rides under a
    `'settings'` key rather than at the document's own top level."""
    data = {'general': {'temp': 300},
            'settings': {'database': {'password': PASSWORD, 'api_key': API_KEY,
                                      'host': 'db.example.org'}}}

    safe = redact_secrets_mapping(data)

    assert safe['settings']['database']['password'] == ''
    assert safe['settings']['database']['api_key'] == ''
    assert safe['settings']['database']['host'] == 'db.example.org'
    assert safe['general']['temp'] == 300


def test_redact_secrets_mapping_passes_non_dict_input_through():
    assert redact_secrets_mapping(None) is None
    assert redact_secrets_mapping([1, 2, 3]) == [1, 2, 3]


def test_redact_secrets_mapping_tolerates_a_document_with_no_database():
    data = {'thermostat': {'type': 'langevin'}}
    assert redact_secrets_mapping(data) == data
