"""`_material_id` must resolve to the right implementation by MRO.

`IndentSimulation` once carried a `_material_id` byte-identical to the one it
inherits from `AFMSimulation`. That no-op override was load-bearing by
accident: it sits before `HeteroAFMSimulation` in `HeteroIndentSimulation`'s
MRO, so it shadowed the pair token, and a *second* identical override existed
on `HeteroIndentSimulation` purely to undo it.

Both were deleted. These tests pin the resolutions that deletion relies on, so
re-adding either override fails here rather than silently renaming every hetero
indent job after material 1 alone.
"""
import pytest

from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.builders.afm.hetero import HeteroAFMSimulation, HeteroIndentSimulation
from frictionsim2d.builders.afm.indent import IndentSimulation


def _owner(cls, name):
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    raise AssertionError(f"{name} not found on {cls.__name__}")


@pytest.mark.parametrize("cls,expected", [
    (AFMSimulation, AFMSimulation),               # single material
    (IndentSimulation, AFMSimulation),            # inherits the single-material one
    (HeteroAFMSimulation, HeteroAFMSimulation),   # pair token
    (HeteroIndentSimulation, HeteroAFMSimulation),  # inherits the pair token
])
def test_material_id_resolves_to(cls, expected):
    assert _owner(cls, "_material_id") is expected, (
        f"{cls.__name__}._material_id resolves to "
        f"{_owner(cls, '_material_id').__name__}, expected {expected.__name__}. "
        "A redundant override was probably re-added."
    )


def test_no_redundant_override_reintroduced():
    """An override identical to the inherited one is the bug, not the fix."""
    for cls in (IndentSimulation, HeteroIndentSimulation):
        assert "_material_id" not in cls.__dict__, (
            f"{cls.__name__} defines _material_id again. If it must differ, say "
            "how in the docstring; if it does not, delete it -- an identical "
            "override shadows the correct implementation further up the MRO."
        )
