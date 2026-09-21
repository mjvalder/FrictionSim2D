"""The set node (spec §4.1).

`test_node_types_import_the_vocabulary`'s run/results-node `family`/`model`
assertions need setters that only land in Tasks 10/11
(`FrictionSimulationData`/`FrictionResultsData`); this task keeps only the
set-node half plus the plain vocabulary imports (controller ruling R10).
"""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.data import FrictionSimulationSetData  # noqa: E402
from src.core.observables import FAMILIES  # noqa: E402
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
