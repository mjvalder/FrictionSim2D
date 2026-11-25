"""Recipe tools: one coherent proposal per simulation family."""
import pytest
from pydantic import ValidationError

from frictionsim2d.agent.changes import ToolInputError
from frictionsim2d.agent.tools.setup import (AFMSetupRequest, PESTipSetupRequest,
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
    from frictionsim2d.agent.tools.setup import IndentSetupRequest, setup_indent
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


_GEOMETRY = ('tip.r', 'sub.thickness', '2D.x', '2D.y', '2D.layers')


def test_the_afm_note_lists_what_it_filled_in(deps):
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    loads_nN=[5, 10, 20, 30, 50]))
    cs = deps.pending[-1]
    note = next(n for n in cs.notes if 'examples/afm/basic.ini' in n)
    for text in ('2D.x=100', '2D.y=100', '2D.layers=[1]', 'tip.r=25', 'sub.thickness=12',
                 'change if needed'):
        assert text in note


def test_request_and_document_values_beat_the_example_defaults(deps):
    deps.document.set('tip', 'r', 40)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    size_A=(150, 120), layers=[2]))
    ops = _ops(deps)
    assert (ops[('2D', 'x')], ops[('2D', 'y')], ops[('2D', 'layers')]) == (150, 120, [2])
    assert ('tip', 'r') not in ops
    note = next(n for n in deps.pending[-1].notes if 'examples/afm/basic.ini' in n)
    assert '2D.x' not in note and 'tip.r' not in note and 'sub.thickness=12' in note


def test_request_tip_radius_beats_the_default(deps):
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw', tip_radius_A=30))
    assert _ops(deps)[('tip', 'r')] == 30


@pytest.mark.parametrize('setup, request_, source, expected', [
    ('setup_afm', AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                  loads_nN=[5, 10, 20, 30, 50]),
     'afm/basic.ini', {('2D', 'x'): 100, ('2D', 'y'): 100, ('2D', 'layers'): [1],
                        ('tip', 'r'): 25, ('sub', 'thickness'): 12}),
    ('setup_sheet', SheetSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                      pressures_GPa=[1]),
     'sheetonsheet/basic.ini', {('2D', 'x'): 100, ('2D', 'layers'): [3]}),
    ('setup_indent', None, 'indent/basic.ini', {('2D', 'x'): 60, ('tip', 'r'): 15}),
    ('setup_pes_tip', PESTipSetupRequest(materials=['h-MoS2'], potential_type='sw', load_nN=2),
     'pes-tip/basic.ini', {('2D', 'x'): 100, ('2D', 'y'): 100, ('tip', 'r'): 25}),
    ('setup_pes_sheet', None, 'pes-sheet/basic.ini', {('2D', 'x'): 30, ('2D', 'layers'): [2]}),
])
def test_each_model_uses_its_own_example(deps, setup, request_, source, expected):
    from frictionsim2d.agent.tools import setup as tools
    if request_ is None:
        cls = {'setup_indent': tools.IndentSetupRequest,
               'setup_pes_sheet': tools.PESSheetSetupRequest}[setup]
        request_ = cls(materials=['h-MoS2'], potential_type='sw')
    getattr(tools, setup)(deps, request_)
    cs = deps.pending[-1]
    ops = _ops(deps)
    for key, value in expected.items():
        assert ops[key] == value
    assert any(f'examples/{source}' in n for n in cs.notes)
    assert not [i for i in cs.issues if i.startswith(_GEOMETRY)], cs.issues


def test_example_defaults_match_the_example_files():
    """The table is copied from examples/*.ini: keep it honest."""
    import configparser
    import json
    from pathlib import Path
    from frictionsim2d.agent.tools.setup import EXAMPLE_DEFAULTS
    examples = Path(__file__).resolve().parents[2] / 'examples'
    for _model, (source, values) in EXAMPLE_DEFAULTS.items():
        parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
        parser.read(examples.parent / source)
        for (section, key), value in values.items():
            raw = json.loads(parser[section][key])
            if key == 'layers':
                assert set(value) <= set(raw), (source, key)
            else:
                assert raw == value, (source, section, key)


def test_pes_tip_override_is_explained(deps):
    """examples/pes-tip/basic.ini's 60 x 60 sheet breaks the 2x-tip rule; the
    assistant raises it and says why instead of copying a failing value."""
    from frictionsim2d.agent.tools import setup as tools
    tools.setup_pes_tip(deps, tools.PESTipSetupRequest(materials=['h-MoS2'],
                                                       potential_type='sw', load_nN=2))
    cs = deps.pending[-1]
    assert cs.ok, cs.issues
    assert any('2x the tip diameter' in n for n in cs.notes)


def test_layers_must_be_positive():
    with pytest.raises(ValidationError):
        AFMSetupRequest(materials=['h-MoS2'], layers=[0])
    with pytest.raises(ValidationError):
        SheetSetupRequest(materials=['h-MoS2'], layers=[3, -1])
