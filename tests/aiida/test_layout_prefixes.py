"""The Group label prefixes are stored in every profile and archive: pinned literally;
`aiida_setting` falls back to its default."""
from frictionsim2d.aiida.layout import (
    ANALYSIS_GROUP_PREFIX, DATASET_PREFIX, GROUP_PREFIX, group_label, is_set_label)


def test_group_label_prefixes_are_pinned():
    assert (GROUP_PREFIX, DATASET_PREFIX, ANALYSIS_GROUP_PREFIX) == (
        'friction2d/', 'friction2d/datasets/', 'friction2d/analyses/')
    assert group_label('260421-afm') == 'friction2d/260421-afm'


def test_only_a_set_label_is_a_set():
    assert is_set_label('friction2d/camp')
    assert not is_set_label('friction2d/datasets/d')
    assert not is_set_label('friction2d/analyses/d')
    assert not is_set_label('Friction2D/camp')


def test_aiida_setting_reads_falls_back_and_defaults(monkeypatch):
    from types import SimpleNamespace

    import frictionsim2d.core.config as config
    from frictionsim2d import aiida as aiida_pkg
    settings = SimpleNamespace(aiida=SimpleNamespace(machine='m1', profile=None))
    monkeypatch.setattr(config, 'load_settings', lambda settings_file=None: settings)
    assert aiida_pkg.aiida_setting('machine', 'x') == 'm1'
    assert aiida_pkg.aiida_setting('profile', 'dflt') == 'dflt'          # None -> default
    assert aiida_pkg.aiida_setting('missing', 7) == 7                    # no such field

    def unreadable(settings_file=None):
        raise OSError('gone')
    monkeypatch.setattr(config, 'load_settings', unreadable)
    assert aiida_pkg.aiida_setting('machine', 'x') == 'x'
