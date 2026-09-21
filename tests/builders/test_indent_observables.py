"""The indent template is on the registry, not its own column list (spec §3 F4)."""
from pathlib import Path

from src.core.observables import channels_for, rename_map

TEMPLATE = (Path(__file__).resolve().parents[2]
            / 'src' / 'templates' / 'afm' / 'indent.lmp')


def test_indent_no_longer_hand_rolls_its_fix():
    text = TEMPLATE.read_text(encoding='utf-8')

    assert 'fix             fc_ave all ave/time' not in text
    assert 'v_fx_react' not in text
    assert 'v_fy_react' not in text
    assert '{{ emit_output_fixes(obs_ctx) }}' in text
    assert '{{ emit_observables(obs_ctx) }}' in text
    assert "{% import 'common/observables.lmp' as obs %}" in text or \
           "from 'common/observables.lmp' import emit_observables, emit_output_fixes" in text


def test_the_registry_can_serve_the_indent_family():
    names = [ob.name for ob in channels_for('indent', 'window')]

    assert {'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz'} <= set(names)
    assert rename_map('indent', 'window')['v_fz_tip'] == 'nf'
    assert rename_map('indent', 'window')['v_fx_spr'] == 'lfx'


def test_template_defines_the_native_names_the_registry_expects():
    text = TEMPLATE.read_text(encoding='utf-8')

    assert 'variable        fx_spr equal f_forcetip[1]*{{ ev_a_to_nn }}' in text
    assert 'variable        fy_spr equal f_forcetip[2]*{{ ev_a_to_nn }}' in text
    assert 'variable        fz_tip equal f_forcetip[3]*{{ ev_a_to_nn }}' in text
