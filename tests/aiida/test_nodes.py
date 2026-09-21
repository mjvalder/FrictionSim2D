"""The set node (spec §4.1).

`test_node_types_import_the_vocabulary`'s run/results-node `family`/`model`
assertions need setters that only land in Tasks 10/11
(`FrictionSimulationData`/`FrictionResultsData`); this task keeps only the
set-node half plus the plain vocabulary imports (controller ruling R10).
"""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.data import FrictionSimulationData, FrictionSimulationSetData  # noqa: E402
from src.core.observables import FAMILIES  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402


def test_set_node_reads_a_root(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_20260910_120000'
    make_run(root, 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2', 'afm')
    make_run(root, 'afm/h-WS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'afm')
    (root / 'settings.yaml').write_text('thermostat:\n  type: langevin\n',
                                        encoding='utf-8')

    node = FrictionSimulationSetData()
    node.set_from_root(root, label='260910-afm')
    node.store()

    assert node.label == '260910-afm'
    assert node.model == 'afm'
    assert node.materials == ['h-MoS2', 'h-WS2']
    assert node.n_runs == 2
    assert node.git_commit == 'deadbeef'
    assert node.git_dirty is False
    assert node.source_host
    assert node.created
    assert 'settings.yaml' in node.base.repository.list_object_names()


def test_set_node_stores_the_ini_when_one_is_present(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    make_run(root, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet')
    ini = tmp_path / 'sheet_config.ini'
    ini.write_text('[general]\ntemp = 300\n', encoding='utf-8')

    node = FrictionSimulationSetData()
    node.set_from_root(root, label='260910-sheet')
    node.add_file(ini, 'config.ini')
    node.store()

    assert 'config.ini' in node.base.repository.list_object_names()
    assert node.to_dict()['label'] == '260910-sheet'


def test_node_types_import_the_vocabulary(aiida_profile_clean):
    from src.core.observables import FAMILIES
    from src.run.naming import MODELS

    assert set(FAMILIES) >= {'hetero', 'flake'}
    assert 'sheetonsheet' in MODELS
    with pytest.raises(ValueError, match='model must be one of'):
        FrictionSimulationSetData().model = 'flake'


def test_set_node_no_longer_knows_only_two_types(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_y'
    make_run(root, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet')

    node = FrictionSimulationSetData()
    node.set_from_root(root, label='260910-pes')

    assert node.model == 'pes_sheet'
    assert node.model in FAMILIES
    assert not hasattr(FrictionSimulationSetData, 'SIMULATION_TYPES')


def test_empty_label_is_refused(aiida_profile_clean):
    node = FrictionSimulationSetData()

    with pytest.raises(ValueError, match='label is required'):
        node.label = '  '


def _one_run(tmp_path, rel, family):
    root = tmp_path / 'simulation_z'
    make_run(root, rel, family)
    return root, discover_runs(root)[0]


def test_run_node_carries_every_token(aiida_profile_clean, tmp_path):
    # R3: an AFM heterostructure run (path afm/mat1+mat2/...) has family
    # 'afm' -- its files follow the AFM schema. 'hetero' is only the sheet-
    # family hetero stack.
    root, loc = _one_run(
        tmp_path, 'afm/h-MoS2+h-WS2+flake-h-WSe2/100x_50y/sub_aSi_tip_Si_r25/K77/L3',
        'afm')

    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid='set-uuid-1')
    node.store()

    assert node.model == 'afm'
    assert node.family == 'afm'
    assert node.materials == ['h-MoS2', 'h-WS2']
    assert node.flake_material == 'h-WSe2'
    assert node.size_x == 100.0 and node.size_y == 50.0
    # RunTokens.substrate is the bare material name with the amorphous marker
    # stripped (see naming.py's _split_amorph_token / RunTokens docstring,
    # Task 3b, 2026-09-21) -- 'sub_aSi_...' parses to substrate='Si',
    # substrate_amorphous=True, not substrate='aSi' as an earlier brief draft
    # (written before that decision landed) assumed.
    assert node.substrate == 'Si' and node.substrate_amorphous is True
    assert node.tip_material == 'Si' and node.tip_radius == 25.0
    assert node.temperature == 77.0
    assert node.layers == 3
    assert node.pot_token is None
    assert node.rel_path == loc.tokens.rel_path
    assert node.set_uuid == 'set-uuid-1'
    assert node.tokens() == loc.tokens


def test_run_node_stores_the_full_provenance_record(aiida_profile_clean, tmp_path):
    _root, loc = _one_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300',
                          'sheetonsheet')

    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid='s')
    node.store()

    prov = node.provenance
    assert prov['schema_version'] == '2.0.0'
    assert prov['config_hash'] == 'ff' * 32
    assert prov['materials'][0]['pot_file_sha256'] == 'ab' * 32


def test_run_node_stages_inputs_and_provenance_files(aiida_profile_clean, tmp_path):
    _root, loc = _one_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300',
                          'sheetonsheet')

    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid='s')
    stored = node.add_run_files(loc)
    node.store()

    names = set(node.base.repository.list_object_names())
    assert 'provenance.json' in names
    assert stored == sorted(stored)
    assert any(n.startswith('provenance/') for n in _walk(node))
    assert any(n.endswith('.in') for n in _walk(node))


def _walk(node):
    out = []
    for root, _dirs, files in node.base.repository.walk():
        out.extend((root / f).as_posix() for f in files)
    return out


def test_oversize_files_are_recorded_not_stored(aiida_profile_clean, tmp_path):
    _root, loc = _one_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300',
                          'sheetonsheet')
    big = loc.provenance_dir / 'potentials'
    big.mkdir(parents=True, exist_ok=True)
    (big / 'huge.sw').write_bytes(b'x' * (2 * 1024 * 1024))

    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid='s')
    node.add_run_files(loc, max_mb=1)
    node.store()

    assert 'provenance/potentials/huge.sw' not in _walk(node)
    skipped = node.oversize_files
    assert skipped[0]['path'] == 'provenance/potentials/huge.sw'
    assert skipped[0]['size'] == 2 * 1024 * 1024
    assert len(skipped[0]['sha256']) == 64
    assert skipped[0]['source'].endswith('huge.sw')


def test_mutable_state_lives_in_extras(aiida_profile_clean, tmp_path):
    _root, loc = _one_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300',
                          'sheetonsheet')
    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid='s')
    node.store()
    assert node.imported_at == ''

    node.status = 'imported'
    node.job_id = 4321

    assert node.status == 'imported'
    assert node.base.extras.get('status') == 'imported'
    assert node.imported_at.startswith('20')
    assert node.job_id == '4321'
    assert 'status' not in node.base.attributes.all
    exported = node.to_dict()
    assert exported['imported_at'] == node.imported_at
    assert exported['status'] == 'imported'
    assert exported['job_id'] == '4321'
    with pytest.raises(ValueError, match='status must be one of'):
        node.status = 'nonsense'
