"""The Group label prefixes are stored in every profile and archive: pinned literally."""
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
