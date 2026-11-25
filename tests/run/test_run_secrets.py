"""No central-DB credential ever lands in a run tree (final review I1).

`provenance/settings.yaml` and `provenance/config.json` are stored verbatim in
the AiiDA repository and travel in archives and export bundles, so a password
or API key written there leaks to everyone the set is shared with.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from frictionsim2d.core.config import GlobalSettings, redact_secrets, redact_secrets_mapping
from frictionsim2d.run.artifacts import generate_package

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'tests' / 'fixtures' / 'examples'

PASSWORD = 'hunter2-central-db-password'
API_KEY = 'sk-central-api-key-0123456789'


def test_redact_secrets_keeps_every_non_credential_field():
    """`GlobalSettings` carries no credential since the central database
    settings went; redaction is a faithful copy, never the original."""
    settings = GlobalSettings()
    settings.hpc.hpc_host = 'cluster.example.org'

    safe = redact_secrets(settings)

    assert safe == settings and safe is not settings
    assert safe.hpc.hpc_host == 'cluster.example.org'


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
    assert 'database' not in stored   # the block is ignored, never written


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


# --- 2026-09-22 review round: five shapes that got a credential past the
# original database/settings-path-only implementation ------------------

def test_redact_secrets_mapping_blanks_a_yaml_anchor_alias():
    """`base: &b {password: X}` then `database: *b` -- `yaml.safe_load`
    expands the alias into an ordinary (shared) nested dict before this ever
    runs, so `base` itself, not just wherever `database` is checked, must be
    walked and blanked."""
    text = (
        "base: &b\n"
        "  host: db.example.org\n"
        f"  password: {PASSWORD}\n"
        "database: *b\n"
    )
    data = yaml.safe_load(text)

    safe = redact_secrets_mapping(data)

    assert safe['base']['password'] == ''
    assert safe['database']['password'] == ''
    assert safe['base']['host'] == 'db.example.org'


def test_redact_secrets_mapping_blanks_an_unanticipated_database_subkey():
    """`database.extra.password` -- an arbitrary nesting depth under
    `database`, not one of the two named legacy profile keys."""
    data = {'database': {'extra': {'password': PASSWORD}, 'host': 'x'}}

    safe = redact_secrets_mapping(data)

    assert safe['database']['extra']['password'] == ''
    assert safe['database']['host'] == 'x'


def test_redact_secrets_mapping_blanks_a_deeply_nested_config_json_shape():
    """`general.settings.database.password` -- `config.json`'s `database`
    block need not sit at `settings.database`; any nesting must be caught."""
    data = {'general': {'settings': {'database': {'password': PASSWORD}}}}

    safe = redact_secrets_mapping(data)

    assert safe['general']['settings']['database']['password'] == ''


def test_redact_secrets_mapping_blanks_a_stray_hpc_api_key():
    """A credential-named key outside `database` entirely."""
    data = {'hpc': {'api_key': API_KEY, 'queue': 'batch'}}

    safe = redact_secrets_mapping(data)

    assert safe['hpc']['api_key'] == ''
    assert safe['hpc']['queue'] == 'batch'


def test_redact_secrets_mapping_blanks_a_bare_top_level_password():
    data = {'password': PASSWORD, 'other': 'kept'}

    safe = redact_secrets_mapping(data)

    assert safe['password'] == ''
    assert safe['other'] == 'kept'


def test_redact_secrets_mapping_blanks_case_insensitively_and_by_suffix():
    data = {'Password': PASSWORD, 'API_KEY': API_KEY, 'db_password': 'x',
           'central_api_key': 'y', 'secret': 'z', 'passwd': 'w', 'apikey': 'v'}

    safe = redact_secrets_mapping(data)

    assert all(value == '' for value in safe.values())


def test_redact_secrets_mapping_leaves_non_secret_lookalikes_alone():
    """Sanity check for the audit note in `frictionsim2d.core.config`: nothing in
    `GlobalSettings` is legitimately named like a secret, so ordinary keys
    that merely CONTAIN (rather than end in) a secret substring must be
    left alone -- only an exact name or a `_`-joined suffix counts."""
    data = {'tokenizer': 'unaffected', 'my_secretary': 'unaffected',
           'passwordless': 'unaffected'}

    assert redact_secrets_mapping(data) == data


def test_redact_secrets_mapping_blanks_a_credential_inside_a_list():
    """A secret in a list of mappings (e.g. `env_lines`-style entries) must not
    slip past because the walk stops at the list."""
    data = {'hpc': {'frameworks': [
        {'name': 'a', 'password': PASSWORD},
        [{'api_key': API_KEY, 'host': 'x.example.org'}],
    ]}}

    safe = redact_secrets_mapping(data)

    assert safe['hpc']['frameworks'][0] == {'name': 'a', 'password': ''}
    assert safe['hpc']['frameworks'][1] == [{'api_key': '', 'host': 'x.example.org'}]
    assert data['hpc']['frameworks'][0]['password'] == PASSWORD   # input untouched
