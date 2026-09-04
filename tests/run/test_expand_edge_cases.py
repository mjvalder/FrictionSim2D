"""Edge cases for ``expand_config_sweeps`` itself (Phase 2b review, task 6,
step 1): the shapes a sweep dict can take that the rest of ``tests/run/``
never exercises directly. Every assertion is on the produced config dicts
and/or ``naming.output_dir_for``'s output, never on a mock.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.run.expand import _validate_runtime_sweep_ordering, expand_config_sweeps
from src.run.naming import output_dir_for

ROOT = Path('examples/potentials')


def _base_config(**general_overrides) -> dict:
    return {
        "general": {"temp": 300.0, **general_overrides},
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": "dummy.sw",
            "cif_path": "dummy.cif",
            "x": 20.0,
            "y": 20.0,
            "layers": [2],
        },
    }


def test_empty_sweep_list_produces_zero_configs() -> None:
    """A list-valued sweep param with no members expands to NOTHING -- there
    is no value left to build with. ``itertools.product`` over an empty
    iterable yields zero combinations, and expand_config_sweeps has no
    special case for it. This is current, real behaviour, pinned here
    because it is easy to trip over silently (an empty ``[general] temp =``
    reads as "use the default", not "build nothing"): a caller that doesn't
    check for an empty result gets a silent no-op run, not an error.
    """
    base_config = _base_config(temp=[])

    expanded = expand_config_sweeps(base_config)

    assert expanded == []


def test_single_value_sweep_produces_exactly_one_config() -> None:
    """A one-member list is still a sweep axis, but with only one value to
    take -- itertools.product must not degenerate into zero or duplicate
    configs."""
    base_config = _base_config(temp=[310.0])

    expanded = expand_config_sweeps(base_config)

    assert len(expanded) == 1
    assert expanded[0]["general"]["temp"] == 310.0


def test_two_sweeps_produce_the_cartesian_product_in_documented_order() -> None:
    """Two independent sweep axes (general.temp and 2D.pot_path) combine into
    their full cartesian product. The order is a direct consequence of
    ``sweep_params``'s insertion order plus ``itertools.product``'s standard
    nested-loop semantics (rightmost/last-inserted key varies fastest):
    the general-section loop inserts ``('general', 'temp')`` first, and the
    potential-sweep block appends ``(primary_key, '_pot_expand')`` after it
    -- so temp is the outer (slow) axis and pot_path is the inner (fast) one.
    """
    base_config = {
        "general": {"temp": [300.0, 350.0]},
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": ["sw/sw_lammps/h-MoS2.sw", "sw/mos2_jiang2.sw"],
            "cif_path": "examples/materials/h-MoS2.cif",
            "x": 100, "y": 100, "layers": [2],
        },
    }

    expanded = expand_config_sweeps(base_config, base_dir=ROOT)

    assert len(expanded) == 4
    combos = [(c["general"]["temp"], Path(c["2D"]["pot_path"]).name) for c in expanded]
    assert combos == [
        (300.0, "h-MoS2.sw"), (300.0, "mos2_jiang2.sw"),
        (350.0, "h-MoS2.sw"), (350.0, "mos2_jiang2.sw"),
    ]


def test_scan_angle_force_without_its_pair_raises() -> None:
    """A parameter that only makes sense as a per-value selector on force or
    pressure (the loopability rule: scan_angle_force is 'in_file', paired
    with one of those two) cannot stand alone as a multi-value list --
    ``_validate_runtime_sweep_ordering`` (run by ``run_simulations`` right
    after ``expand_config_sweeps``, over each expanded config's ``general``
    section) rejects it. This is run.md finding 5 (previously untested)."""
    with pytest.raises(ValueError, match="scan_angle_force list requires force or pressure values"):
        _validate_runtime_sweep_ordering({"scan_angle_force": [1.0, 2.0]})


def test_a_sweep_with_all_identical_values_collides() -> None:
    """Documents current behaviour for a degenerate sweep list -- every value
    the same. expand_config_sweeps does not deduplicate and has no notion of
    "this sweep changed nothing": it happily produces one config per list
    member, and since general.temp is exactly what output_dir_for keys the
    K{temp} segment on, two identical temps collide on the SAME directory --
    the second build would silently overwrite the first, the same class of
    bug as R-3 (run.md's LOOPABILITY invariant), but from a different
    cause (a repeated value, not a misclassified parameter) and NOT covered
    by this task's fix. Flagged here rather than silently pinned: this is a
    real, separate gap for a future task, not a deliberate design choice.
    """
    base_config = _base_config(temp=[300.0, 300.0])

    expanded = expand_config_sweeps(base_config)

    assert len(expanded) == 2
    dirs = {output_dir_for(c, 'sheetonsheet') for c in expanded}
    assert len(dirs) == 1   # collision, not a raise -- see docstring
