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


def test_suggest_values_asks_until_scaling_exists(deps):
    text = suggest_values(deps, 'load', 'low', 5)
    assert 'ask the user' in text and 'make_sweep' in text


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
