"""No credential ever enters the AiiDA store (follow-up A).

`FrictionSimulationSetData.set_from_root` stores a root `settings.yaml`
verbatim, and `FrictionSimulationData.add_run_files` stores `provenance/**`
(including `provenance/settings.yaml` and `provenance/config.json`) verbatim.
Both are user documents that can carry `database.password`/`database.api_key`
(current flat shape) or the pre-Task-31 `database.local.*`/`database.central.*`
nested shape -- and, once stored, they travel unchanged through
`export_results`, `export_archive` and `rebuild_set`. Every settings/config
document staged into a friction2d node repository must be redacted first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.data import FrictionSimulationSetData  # noqa: E402
from src.aiida.export import export_archive, export_results, import_archive, rebuild_set  # noqa: E402
from src.aiida.ingest import group_label, ingest_root  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'

PASSWORD = 'hunter2-central-db-password'
API_KEY = 'sk-central-api-key-0123456789'
LOCAL_PASSWORD = 'hunter2-local-db-password'
LOCAL_API_KEY = 'sk-local-api-key-0123456789'

_SECRETS = (PASSWORD, API_KEY, LOCAL_PASSWORD, LOCAL_API_KEY)

#: The flat (current, Task 31) `DatabaseSettings` shape.
FLAT_DATABASE = {
    'host': 'db.example.org', 'password': PASSWORD, 'api_key': API_KEY,
}

#: The legacy (pre-Task 31) nested shape: `active_profile` + `local`/`central`
#: profile sub-dicts, each with its own `password`/`api_key`.
LEGACY_DATABASE = {
    'active_profile': 'central',
    'local': {'host': 'localhost', 'password': LOCAL_PASSWORD,
             'api_key': LOCAL_API_KEY},
    'central': {'host': 'db.example.org', 'password': PASSWORD,
               'api_key': API_KEY},
}


def _prov_dir(root: Path, rel: str) -> Path:
    """The provenance dir `make_run` writes for `rel` (AFM nests it one level
    up from `L{n}`, mirroring `src.run.tree.provenance_dir_for`)."""
    run_dir = root / rel
    layers = int(rel.rsplit('/L', 1)[1]) if '/L' in rel else None
    return (run_dir.parent if layers is not None else run_dir) / 'provenance'


def _write_secret_provenance(root: Path, rel: str, *, database: dict) -> None:
    """Overwrite `make_run`'s stub `provenance/settings.yaml` and
    `provenance/config.json` with credential-bearing content, in the given
    `database` shape."""
    prov_dir = _prov_dir(root, rel)
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / 'settings.yaml').write_text(
        yaml.safe_dump({'thermostat': {'type': 'langevin'}, 'database': database},
                       sort_keys=False),
        encoding='utf-8')
    (prov_dir / 'config.json').write_text(
        json.dumps({'general': {'temp': 300}, 'settings': {'database': database}}),
        encoding='utf-8')


def _assert_no_secrets(text: str, path) -> None:
    for secret in _SECRETS:
        assert secret not in text, f"{secret!r} leaked into {path}"


def _assert_group_has_no_secrets(label: str) -> None:
    """Walk every repository file of every node in `friction2d/<label>` and
    assert none of `_SECRETS` occurs anywhere -- the store-wide guard."""
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel

    group = Group.collection.get(label=group_label(label))
    assert len(group.nodes) > 0
    for node in group.nodes:
        for dirpath, _dirs, files in node.base.repository.walk():
            for fname in files:
                rel = (dirpath / fname).as_posix()
                content = node.base.repository.get_object_content(rel, mode='rb')
                _assert_no_secrets(content.decode('utf-8', errors='replace'),
                                  f'{node.__class__.__name__}:{rel}')


def test_set_from_root_strips_credentials_from_a_root_settings_file(
        aiida_profile_clean, tmp_path):
    """`set_from_root` prefers a root-level `settings.yaml` (a user file, e.g.
    from `settings init` run in the root) over the per-run one -- exactly the
    file this test plants with credentials."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    (root / 'settings.yaml').write_text(
        yaml.safe_dump({'thermostat': {'type': 'langevin'},
                        'database': FLAT_DATABASE}, sort_keys=False),
        encoding='utf-8')

    node = FrictionSimulationSetData()
    node.set_from_root(root, label='secrets-set')
    node.store()

    stored = node.base.repository.get_object_content('settings.yaml', mode='r')
    _assert_no_secrets(stored, 'settings.yaml (set node)')
    parsed = yaml.safe_load(stored)
    assert parsed['database']['host'] == 'db.example.org'
    assert parsed['thermostat']['type'] == 'langevin'


@pytest.mark.parametrize('database', [FLAT_DATABASE, LEGACY_DATABASE],
                         ids=['flat', 'legacy-nested'])
def test_ingest_strips_credentials_from_provenance_documents(
        aiida_profile_clean, tmp_path, database):
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    _write_secret_provenance(root, REL, database=database)

    report = ingest_root(root, label='secrets-ingest')
    assert report.ok(), report.render()

    _assert_group_has_no_secrets('secrets-ingest')

    exported = export_results('secrets-ingest', tmp_path / 'exported')
    for path in exported:
        _assert_no_secrets(path.read_text(encoding='utf-8', errors='replace'), path)

    archive_path = tmp_path / 'archive.aiida'
    export_archive('secrets-ingest', archive_path)

    aiida_profile_clean.reset_storage()
    n_new = import_archive(archive_path)
    assert n_new > 0
    _assert_group_has_no_secrets('secrets-ingest')


def test_group_wide_guard_no_secret_string_anywhere(aiida_profile_clean, tmp_path):
    """Store-wide guard: after ingesting a tree with BOTH database shapes
    (one run flat, one run legacy-nested), no repository file of any node in
    the set's group contains any secret string."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    _write_secret_provenance(root, REL, database=FLAT_DATABASE)
    rel2 = 'afm/h-WS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'
    make_run(root, rel2, 'afm')
    _write_secret_provenance(root, rel2, database=LEGACY_DATABASE)
    (root / 'settings.yaml').write_text(
        yaml.safe_dump({'database': FLAT_DATABASE}), encoding='utf-8')

    ingest_root(root, label='secrets-guard')

    _assert_group_has_no_secrets('secrets-guard')


def test_reingest_of_an_unredacted_tree_is_idempotent(aiida_profile_clean, tmp_path):
    """Re-ingesting the SAME (still unredacted on disk) tree a second time
    creates nothing new: the first ingest's sanitised, stored bytes are what
    every subsequent comparison is keyed against."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    _write_secret_provenance(root, REL, database=FLAT_DATABASE)

    first = ingest_root(root, label='secrets-reingest')
    assert first.ok(), first.render()
    assert first.created_runs

    second = ingest_root(root, label='secrets-reingest')
    assert second.ok(), second.render()
    assert second.created_runs == []
    assert second.created_results == []
    assert second.updated_provenance == []


def _minimal_afm_config_with_secrets(tmp_path):
    """A real, validating `AFMSimulationConfig` whose embedded `settings`
    carries central-DB credentials -- built the way `tests/aiida/test_export.
    py`'s `_minimal_sheet_flake_config` builds its sheet-on-sheet analogue."""
    from src.core.config import AFMSimulationConfig, load_settings

    dummy_pot = tmp_path / 'dummy.sw'
    dummy_cif = tmp_path / 'dummy.cif'
    dummy_pot.write_text('# dummy potential\n', encoding='utf-8')
    dummy_cif.write_text('# dummy cif\n', encoding='utf-8')

    settings = load_settings()
    settings.database.password = PASSWORD
    settings.database.api_key = API_KEY

    return AFMSimulationConfig(
        general={'temp': 300.0},
        tip={'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': str(dummy_pot),
            'cif_path': str(dummy_cif), 'r': 25.0},
        **{'2D': {'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': str(dummy_pot),
                  'cif_path': str(dummy_cif), 'x': 200.0, 'y': 200.0,
                  'layers': [2], 'stack_type': 'AB'}},
        sub={'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': str(dummy_pot),
            'cif_path': str(dummy_cif), 'thickness': 12.0, 'amorph': 'a'},
        settings=settings,
    )


def test_rebuild_set_reloads_a_redacted_config_json(aiida_profile_clean, tmp_path,
                                                     monkeypatch):
    """`rebuild_set`'s exported `config.json` -- redacted at ingest time -- must
    still validate through the real `AFMSimulationConfig` and reach
    `generate_package`, i.e. blanking the credentials does not break the
    rebuild path (only `database.password`/`database.api_key`, plain `str`
    fields, are touched)."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')

    cfg = _minimal_afm_config_with_secrets(tmp_path)
    prov_dir = _prov_dir(root, REL)
    (prov_dir / 'config.json').write_text(cfg.model_dump_json(indent=2),
                                          encoding='utf-8')

    ingest_root(root, label='secrets-rebuild')
    seen = {}

    class _FakeResult:
        def __init__(self):
            self.created = []
            self.simulation_root = None
            self.failures = []

    def _fake(config_file, model, **kwargs):  # pylint: disable=unused-argument
        text = Path(config_file).read_text(encoding='utf-8')
        _assert_no_secrets(text, config_file)
        from src.core.config import AFMSimulationConfig  # pylint: disable=import-outside-toplevel
        seen['config_obj'] = AFMSimulationConfig(**json.loads(text))
        return _FakeResult()

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    rebuild_set('secrets-rebuild', tmp_path / 'rebuilt')

    assert seen['config_obj'].settings.database.password == ''
    assert seen['config_obj'].settings.database.api_key == ''
