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
from src.aiida.export import import_archive, rebuild_set  # noqa: E402
from src.aiida.ingest import group_label, import_root  # noqa: E402
from tests.helpers.results_tree import make_family_run  # noqa: E402
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


def _assert_node_has_no_secrets(node) -> None:
    """Walk every repository file of `node` and assert none of `_SECRETS`
    occurs anywhere."""
    for dirpath, _dirs, files in node.base.repository.walk():
        for fname in files:
            rel = (dirpath / fname).as_posix()
            content = node.base.repository.get_object_content(rel, mode='rb')
            _assert_no_secrets(content.decode('utf-8', errors='replace'),
                              f'{node.__class__.__name__}:{rel}')


def _assert_group_has_no_secrets(label: str) -> None:
    """Walk every repository file of every CalcJob in `friction2d/<label>`
    and of every node linked to it (inputs and outputs) and assert none of
    `_SECRETS` occurs anywhere -- the store-wide guard."""
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel

    group = Group.collection.get(label=group_label(label))
    assert len(group.nodes) > 0
    seen = set()
    for calc in group.nodes:
        linked = (calc.base.links.get_incoming().all_nodes()
                  + calc.base.links.get_outgoing().all_nodes())
        for node in [calc, *linked]:
            if node.pk not in seen:
                seen.add(node.pk)
                _assert_node_has_no_secrets(node)


def _secret_tree(root: Path, material: str, database: dict) -> str:
    """An AFM run (system + slide deck) with credential-bearing provenance."""
    run_dir = make_family_run(root, 'afm', material=material)
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    rel = run_dir.relative_to(root).as_posix()
    _write_secret_provenance(root, rel, database=database)
    return rel


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


def test_set_from_root_also_finds_a_root_settings_yml(aiida_profile_clean, tmp_path):
    """`.yml` is as valid a YAML extension as `.yaml`; `set_from_root` must
    fall back to it when there is no `settings.yaml`, and still stage the
    result under the canonical `settings.yaml` name (what `load_settings`/
    `rebuild_set` look for)."""
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    (root / 'settings.yml').write_text(
        yaml.safe_dump({'database': FLAT_DATABASE}, sort_keys=False),
        encoding='utf-8')

    node = FrictionSimulationSetData()
    node.set_from_root(root, label='secrets-set-yml')
    node.store()

    assert 'settings.yaml' in node.base.repository.list_object_names()
    stored = node.base.repository.get_object_content('settings.yaml', mode='r')
    _assert_no_secrets(stored, 'settings.yaml (from a root settings.yml)')
    assert yaml.safe_load(stored)['database']['host'] == 'db.example.org'


@pytest.mark.parametrize('database', [FLAT_DATABASE, LEGACY_DATABASE],
                         ids=['flat', 'legacy-nested'])
def test_import_strips_credentials_from_provenance_documents(
        aiida_profile_clean, tmp_path, database):
    """Export and archive round trips of the new graph are Task 10's."""
    root = tmp_path / 'simulation_x'
    _secret_tree(root, 'h-MoS2', database)

    report = import_root(root, label='secrets-import')
    assert report.ok(), report.render()
    assert report.created_decks

    _assert_group_has_no_secrets('secrets-import')


def test_group_wide_guard_no_secret_string_anywhere(aiida_profile_clean, tmp_path):
    """Store-wide guard: after importing a tree with BOTH database shapes
    (one run flat, one run legacy-nested), no repository file of any node in
    or linked to the set's group contains any secret string."""
    root = tmp_path / 'simulation_x'
    _secret_tree(root, 'h-MoS2', FLAT_DATABASE)
    _secret_tree(root, 'h-WS2', LEGACY_DATABASE)
    (root / 'settings.yaml').write_text(
        yaml.safe_dump({'database': FLAT_DATABASE}), encoding='utf-8')

    report = import_root(root, label='secrets-guard')
    assert report.ok(), report.render()
    assert len(report.created_decks) == 4

    _assert_group_has_no_secrets('secrets-guard')


def test_reimport_of_an_unredacted_tree_is_idempotent(aiida_profile_clean, tmp_path):
    """Re-importing the SAME (still unredacted on disk) tree creates nothing
    new: the fingerprint is over the files, the stored bytes are redacted."""
    root = tmp_path / 'simulation_x'
    _secret_tree(root, 'h-MoS2', FLAT_DATABASE)

    first = import_root(root, label='secrets-reimport')
    assert first.ok(), first.render()
    assert first.created_decks

    second = import_root(root, label='secrets-reimport')
    assert second.ok(), second.render()
    assert second.created_decks == []
    assert len(second.skipped_decks) == len(first.created_decks)


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
    assert make_family_run(root, 'afm').relative_to(root).as_posix() == REL

    cfg = _minimal_afm_config_with_secrets(tmp_path)
    prov_dir = _prov_dir(root, REL)
    (prov_dir / 'config.json').write_text(cfg.model_dump_json(indent=2),
                                          encoding='utf-8')

    assert import_root(root, label='secrets-rebuild').created_decks
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


@pytest.mark.skip(reason='removed in Task 11')
def test_calcjob_folder_data_staging_strips_credentials(aiida_profile_clean, tmp_path):
    """`prepare_simulation_folder`/`_stage_sim_dir` -- used by
    `submit_simulation` to build a CalcJob's `simulation_dir`
    input -- stage `provenance/**` the same way `FrictionSimulationData.
    add_run_files` does: a pre-redaction tree's credentials must not survive
    THIS staging path either. The LAMMPS input file it also stages must stay
    byte-identical -- only settings/config documents are sanitised.
    """
    from src.aiida.calcjob import prepare_simulation_folder  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm')
    _write_secret_provenance(root, REL, database=FLAT_DATABASE)
    original_system_in = (run_dir / 'lammps' / 'system.in').read_bytes()

    folder = prepare_simulation_folder(run_dir)

    _assert_node_has_no_secrets(folder)
    stored_config = folder.base.repository.get_object_content(
        'provenance/config.json', mode='r')
    assert json.loads(stored_config)['settings']['database']['host'] == \
        FLAT_DATABASE['host']
    assert folder.base.repository.get_object_content(
        'lammps/system.in', mode='rb') == original_system_in


@pytest.mark.skip(reason='removed in Task 11')
def test_archive_export_of_a_calcjobs_input_folder_carries_no_secret(
        aiida_profile_clean, tmp_path):
    """`aiida.tools.archive.create_archive`'s default traversal rules follow
    a results-like node BACKWARD through the calculation that created it to
    that calculation's own inputs -- so `export_archive`, which names only
    the set/run/results nodes of a group, can still pull a CalcJob's
    `simulation_dir` `FolderData` into the archive via that traversal even
    though it never names the `FolderData` or the `CalcJobNode` itself. Built
    the way aiida-core's own plugin test suites build a bare `CalcJobNode`
    (no live Process/Runner/broker needed) -- mirrors `tests/aiida/
    test_calcjob_parser.py`'s `_make_calc_node`, since actually running the
    CalcJob through the engine needs a daemon/broker this suite does not have.
    """
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel
    from aiida.orm import CalcJobNode, FolderData, Int, QueryBuilder  # pylint: disable=import-outside-toplevel
    from aiida.tools.archive import create_archive  # pylint: disable=import-outside-toplevel

    from src.aiida.calcjob import prepare_simulation_folder  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'simulation_x'
    run_dir = make_run(root, REL, 'afm')
    _write_secret_provenance(root, REL, database=FLAT_DATABASE)

    sim_dir_folder = prepare_simulation_folder(run_dir)
    sim_dir_folder.store()

    calc = CalcJobNode(process_type='aiida.calculations:friction2d.lammps')
    calc.set_option('resources', {'num_machines': 1})
    calc.base.links.add_incoming(sim_dir_folder, link_type=LinkType.INPUT_CALC,
                                 link_label='simulation_dir')
    calc.store()

    output = Int(1)
    output.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='result')
    output.store()
    calc.seal()  # create_archive refuses an unsealed ProcessNode; must
                 # outlive every outgoing link, which a sealed node refuses

    archive_path = tmp_path / 'calcjob_archive.aiida'
    create_archive([output], filename=archive_path)

    aiida_profile_clean.reset_storage()
    n_new = import_archive(archive_path)
    assert n_new > 0

    folders = QueryBuilder().append(FolderData).all(flat=True)
    assert folders, (
        "the archive's default traversal rules did not pull the CalcJob's "
        "simulation_dir input in alongside its output -- this test no longer "
        "exercises the risk it is meant to guard")
    for folder in folders:
        _assert_node_has_no_secrets(folder)
