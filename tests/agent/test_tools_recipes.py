"""Recipe tools: one coherent proposal per simulation family."""
import pytest
from pydantic import ValidationError

from src.agent.changes import ToolInputError
from src.agent.tools.setup import (AFMSetupRequest, PESTipSetupRequest,
                                   SheetSetupRequest, setup_afm, setup_pes_tip,
                                   setup_sheet)
from tests.helpers.agent import make_deps


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


def _ops(deps):
    return {(op.section, op.key): op.value for op in deps.pending[-1].ops if op.kind == 'set'}


def test_afm_sweep_recipe(deps):
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2', 'h-WS2'], potential_type='sw',
                                    loads_nN=[5, 10, 20], speed_m_s=2))
    cs = deps.pending[-1]
    assert _ops(deps)[('general', 'force')] == [5, 10, 20]
    assert _ops(deps)[('general', 'scan_speed')] == 2
    assert [op.kind for op in cs.ops][:2] == ['material_dirs', 'materials']
    assert any('h-MoS2' in n for n in cs.notes)


def test_switches_model_first(deps):
    setup_sheet(deps, SheetSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                        pressures_GPa=[1]))
    first = deps.pending[-1].ops[0]
    assert (first.kind, first.value) == ('set_model', 'sheetonsheet')
    assert _ops(deps)[('general', 'pressure')] == [1]


def test_pes_tip_sets_grid_and_single_load(deps):
    setup_pes_tip(deps, PESTipSetupRequest(materials=['h-WS2'], potential_type='sw',
                                           load_nN=2, grid_n=12))
    assert deps.pending[-1].ops[0].value == 'pes_tip'
    assert _ops(deps)[('general', 'force')] == 2
    assert _ops(deps)[('pes', 'grid_n')] == 12


def test_size_sets_x_and_y(deps):
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw', size_A=(120, 80)))
    assert (_ops(deps)[('2D', 'x')], _ops(deps)[('2D', 'y')]) == (120, 80)


def test_negative_load_rejected_by_schema():
    with pytest.raises(ValidationError):
        AFMSetupRequest(materials=['h-MoS2'], loads_nN=[-5])


def test_unknown_material_is_a_retry_hint(deps):
    with pytest.raises(ToolInputError):
        setup_afm(deps, AFMSetupRequest(materials=['unobtainium'], potential_type='sw'))
