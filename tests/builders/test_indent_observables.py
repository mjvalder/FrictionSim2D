"""The indent template is on the registry, not its own column list (spec §3 F4)."""
from frictionsim2d.core.observables import channels_for, rename_map

def test_the_registry_can_serve_the_indent_family():
    names = [ob.name for ob in channels_for('indent', 'window')]

    assert {'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz'} <= set(names)
    assert rename_map('indent', 'window')['v_fz_tip'] == 'nf'
    assert rename_map('indent', 'window')['v_fx_spr'] == 'lfx'
