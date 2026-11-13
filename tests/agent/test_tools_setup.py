"""Setup tools that read the Builder's documents or propose field changes."""
import pytest

from src.agent.changes import ToolInputError
from src.agent.tools.setup import (FieldChange, explain_current, find_parameter,
                                   list_modes, make_sweep, make_sweep_tool,
                                   set_fields, suggest_values)
from tests.helpers.agent import make_deps


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


def test_list_modes_names_every_model(deps):
    text = list_modes(deps)
    for model in ('afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet'):
        assert f'[{model}]' in text


def test_explain_current_reports_model_and_validation(deps):
    text = explain_current(deps)
    assert text.startswith('model: afm')
    assert 'issues' in text


def test_find_parameter_finds_scan_speed(deps):
    text = find_parameter(deps, 'sliding speed')
    assert 'config:general.scan_speed' in text


def test_find_parameter_searches_settings(deps):
    section = next(iter(deps.settings_document.values))
    key = next(iter(deps.settings_document.values[section]))
    assert f'settings:{section}.{key}' in find_parameter(deps, key)


def test_find_parameter_no_match(deps):
    assert find_parameter(deps, 'zzzqqq') == 'no matching parameter'


def test_set_fields_proposes_without_touching(deps):
    handle = set_fields(deps, [FieldChange(section='general', key='scan_speed', value=7)])
    assert 'proposed' in handle
    assert deps.pending[0].ops[-1].key == 'scan_speed'
    assert deps.pending[0].ops[-1].value == 7
    assert deps.document.get('general', 'scan_speed') == 2.0


def test_set_fields_accepts_dotted_key(deps):
    set_fields(deps, [FieldChange(section='general', key='general.scan_speed', value=7)])
    assert deps.pending[0].ops[-1].key == 'scan_speed'


def test_set_fields_unknown_key_suggests(deps):
    with pytest.raises(ToolInputError, match='scan_speed'):
        set_fields(deps, [FieldChange(section='general', key='scan_sped', value=2)])


def test_set_fields_unknown_section(deps):
    with pytest.raises(ToolInputError, match='find_parameter'):
        set_fields(deps, [FieldChange(section='nope', key='x', value=1)])


def test_set_fields_enables_absent_optional_section(deps):
    absent = [s.key for s in deps.document.sections()
              if s.optional and not deps.document.has_section(s.key)
              and not deps.document.is_section_locked(s.key)]
    if not absent:
        pytest.skip('no optional section absent in a blank afm document')
    spec = deps.document.section(absent[0])
    set_fields(deps, [FieldChange(section=spec.key, key=spec.fields[0].key, value=spec.fields[0].default)])
    assert deps.pending[0].ops[0].kind == 'enable_section'


@pytest.mark.parametrize('args, expected', [
    ((2, 10, 5, 'linear'), [2.0, 4.0, 6.0, 8.0, 10.0]),
    ((1, 100, 3, 'log'), [1.0, 10.0, 100.0]),
    ((5, 5, 1, 'linear'), [5.0]),
])
def test_make_sweep(args, expected):
    assert make_sweep(*args) == pytest.approx(expected)


def test_make_sweep_rejects_bad_input():
    with pytest.raises(ToolInputError):
        make_sweep(0, 10, 3, 'log')
    with pytest.raises(ToolInputError):
        make_sweep(1, 2, 0)


def test_make_sweep_tool_text(deps):
    assert make_sweep_tool(deps, 2, 10, 5) == 'values: 2, 4, 6, 8, 10'


def test_suggest_values_low_load_uses_reference_band(deps):
    deps.document.set('tip', 'r', 25)
    text = suggest_values(deps, 'load', 'low', 5)
    assert 'values: 10, 12.5, 15, 17.5, 20' in text
    assert 'nN' in text and 'baseline loads' in text and 'taxonomy' not in text
    assert 'warning' not in text.lower() and 'measured' not in text.lower()
    assert len(text) <= 1500


def test_suggest_values_has_no_spacing_argument_and_lists_synonyms():
    import inspect
    assert 'spacing' not in inspect.signature(suggest_values).parameters
    for word in ('small', 'light', 'moderate', 'medium', 'large', 'heavy'):
        assert word in suggest_values.__doc__


def test_suggest_values_accepts_synonyms(deps):
    deps.document.set('tip', 'r', 25)
    assert 'values: 20, 35, 50' in suggest_values(deps, 'load', 'moderate', 3)


def test_suggest_values_scales_with_the_tip_radius(deps):
    deps.document.set('tip', 'r', 50)
    text = suggest_values(deps, 'load', 'low', 5)
    assert 'values: 40, 50, 60, 70, 80' in text
    assert 'R = 50 A' in text and 'GPa' in text


def test_suggest_values_without_rule_offers_examples(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings, model='sheetonsheet')
    text = suggest_values(deps, 'pressure', 'low', 5)
    assert 'No scaling rule' in text and 'make_sweep' in text
    assert 'example-config values' in text and 'starting point' in text
    assert 'confirm' in text and 'range' in text


@pytest.mark.parametrize('section, key, value', [
    ('simulation', 'minimization_command', 'minimize 0 0 1 1'),
    ('simulation', 'neigh_modify_command', 'neigh_modify every 2'),
    ('hpc', 'num_cpus', 64),
    ('hpc', 'modules', ['lammps']),
    ('hpc', 'conda_env', '/opt/env'),
    ('hpc', 'scratch_dir', '/scratch'),
])
def test_set_fields_refuses_executed_settings(deps, section, key, value):
    with pytest.raises(ToolInputError, match='Settings tab yourself'):
        set_fields(deps, [FieldChange(scope='settings', section=section, key=key, value=value)])
    assert deps.pending == []


def test_refused_settings_constant_covers_the_rule(deps):
    from src.agent.tools.setup import REFUSED_SETTINGS
    for spec in deps.settings_document.sections():
        for spec_field in spec.fields:
            key = spec_field.key
            if (spec.key == 'hpc' or key.endswith(('_command', '_dir'))
                    or key in ('modules', 'conda_env')):
                assert (spec.key, key) in REFUSED_SETTINGS


@pytest.mark.parametrize('scope, section, key, value', [
    ('config', 'general', 'scan_speed', '2\nprint hi'),
    ('settings', 'simulation', 'min_style', 'cg\nshell rm -rf x'),
    ('config', 'general', 'force', ['1', '2\n3']),
])
def test_set_fields_refuses_newlines(deps, scope, section, key, value):
    with pytest.raises(ToolInputError, match='line break'):
        set_fields(deps, [FieldChange(scope=scope, section=section, key=key, value=value)])


def test_compact_cuts_a_single_huge_line():
    from src.agent.tools import compact
    text = compact('x' * 5000 + '\nmore', limit=100)
    assert len(text) <= 100 and text.startswith('x' * 50) and text.endswith('(truncated)')


@pytest.mark.parametrize('args', [(0, 1, 201), (float('nan'), 1, 3), (0, float('inf'), 3),
                                  (float('-inf'), 0, 3)])
def test_make_sweep_rejects_huge_or_non_finite(args):
    with pytest.raises(ToolInputError):
        make_sweep(*args)


def test_make_sweep_allows_200():
    assert len(make_sweep(0, 1, 200)) == 200


def test_direct_mode_keeps_the_snapshot_in_step_with_the_gui(deps):
    import copy
    from src.agent.changes import apply_changeset
    live = copy.deepcopy(deps.document), copy.deepcopy(deps.settings_document)
    deps.direct_apply = True
    set_fields(deps, [FieldChange(section='general', key='scan_speed', value=7)])
    assert 'general.scan_speed = 7' in explain_current(deps)
    set_fields(deps, [FieldChange(section='general', key='temp', value=250)])
    first, second = deps.pending
    assert apply_changeset(first, *live).applied
    result = apply_changeset(second, *live)
    assert result.applied and not result.stale
    assert live[0].get('general', 'scan_speed') == 7 and live[0].get('general', 'temp') == 250



def test_make_sweep_keeps_tiny_values():
    values = make_sweep(1e-9, 1e-7, 3, 'log')
    assert values == pytest.approx([1e-9, 1e-8, 1e-7], rel=1e-8)
    assert make_sweep(0, 2e-8, 3) == pytest.approx([0, 1e-8, 2e-8], rel=1e-8)
    assert make_sweep(0.1, 0.3, 3) == [0.1, 0.2, 0.3]


def test_set_fields_matches_section_case_insensitively(deps):
    set_fields(deps, [FieldChange(section='GENERAL', key='temp', value=300)])
    assert any(op.section == 'general' and op.key == 'temp' for op in deps.pending[-1].ops)


def test_set_fields_unknown_section_suggests(deps):
    with pytest.raises(ToolInputError, match="did you mean general"):
        set_fields(deps, [FieldChange(section='genral', key='temp', value=300)])


def test_set_fields_key_case_hint(deps):
    with pytest.raises(ToolInputError, match='did you mean scan_speed'):
        set_fields(deps, [FieldChange(section='general', key='SCAN_SPEED', value=1)])


def test_set_fields_enables_a_section_once(deps):
    absent = [s for s in deps.document.sections()
              if s.optional and not deps.document.has_section(s.key)
              and not deps.document.is_section_locked(s.key) and len(s.fields) >= 2]
    if not absent:
        pytest.skip('no optional section with two fields absent in a blank afm document')
    spec = absent[0]
    set_fields(deps, [FieldChange(section=spec.key, key=f.key, value=f.default)
                      for f in spec.fields[:2]])
    kinds = [op.kind for op in deps.pending[-1].ops]
    assert kinds.count('enable_section') == 1
