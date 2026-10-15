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


@pytest.mark.parametrize('materials', [['h-MoS2', 'h-WS2'], ['h-MoS2']])
def test_afm_fills_tip_and_substrate_from_blank(deps, materials):
    setup_afm(deps, AFMSetupRequest(materials=materials, potential_type='sw', loads_nN=[5, 10]))
    cs = deps.pending[-1]
    ops = _ops(deps)
    for section in ('tip', 'sub'):
        assert ops[(section, 'mat')] == 'Si' and ops[(section, 'pot_type')] == 'sw'
        assert ops[(section, 'cif_path')].endswith('materials/Si.cif')
        assert ops[(section, 'pot_path')].endswith('Si.sw')
    assert ops[('sub', 'amorph')] == 'a'
    assert not [i for i in cs.issues if i.startswith(('tip.mat', 'tip.pot', 'tip.cif',
                                                       'sub.mat', 'sub.pot', 'sub.cif'))]
    assert any('tip: Si (sw) from examples:/materials/Si.cif' in n
               and 'substrate: amorphous Si' in n for n in cs.notes)


def test_indent_and_pes_tip_fill_the_tip(deps):
    from src.agent.tools.setup import IndentSetupRequest, setup_indent
    setup_indent(deps, IndentSetupRequest(materials=['h-MoS2'], potential_type='sw'))
    assert _ops(deps)[('sub', 'mat')] == 'Si'
    setup_pes_tip(deps, PESTipSetupRequest(materials=['h-MoS2'], potential_type='sw'))
    ops = _ops(deps)  # switching from afm keeps its required [sub]: fill it too
    assert ops[('tip', 'mat')] == 'Si' and ops[('sub', 'mat')] == 'Si'
    assert not [i for i in deps.pending[-1].issues if i.startswith(('tip.mat', 'sub.mat'))]


def test_pes_tip_leaves_an_absent_substrate_off(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings, model='pes_tip')
    setup_pes_tip(deps, PESTipSetupRequest(materials=['h-MoS2'], potential_type='sw'))
    ops = _ops(deps)
    assert ops[('tip', 'mat')] == 'Si' and ('sub', 'mat') not in ops
    assert not [i for i in deps.pending[-1].issues if i.startswith(('tip.', 'sub.'))
                and not i.startswith('tip.r')]


def test_tip_left_alone_when_set_unless_named(deps):
    for section in ('tip', 'sub'):
        deps.document.set(section, 'mat', 'Mine')
        for key in ('cif_path', 'pot_path', 'pot_type'):
            deps.document.set(section, key, 'x')
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw'))
    assert ('tip', 'mat') not in _ops(deps)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    tip_material='Si'))
    assert _ops(deps)[('tip', 'mat')] == 'Si' and ('sub', 'mat') not in _ops(deps)
