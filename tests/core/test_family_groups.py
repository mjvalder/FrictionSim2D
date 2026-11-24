"""The AFM-stack / sheet-stack family groups have one home."""
from frictionsim2d.core.observables import AFM_LIKE, FAMILIES, SHEET_LIKE


def test_groups_partition_the_families():
    assert set(AFM_LIKE) | set(SHEET_LIKE) == set(FAMILIES)
    assert not set(AFM_LIKE) & set(SHEET_LIKE)
