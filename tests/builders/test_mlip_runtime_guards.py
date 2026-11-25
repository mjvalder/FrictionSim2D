"""Guards for the two production-scale failures found by generate->run (F2, F3).

Both were invisible to component-level tests: the emitted deck was individually
valid, and only failed when a real builder output met a real solver.
"""

import pytest


def test_d3_cutoff_vs_box_is_validated():
    """F3: d3's 16 A cutoff needs a box bigger than cutoff + neighbour skin.

    Measured: an x=y=15 sheet builds a 15.9 x 16.5 A box and dies with
    "Communication cutoff ... cannot exceed periodic box length"; x=y=20
    (19.1 x 22.1 A) clears it. Small screening cells therefore need a smaller
    d3_cutoff, and the failure must be diagnosed up front rather than deep in a
    LAMMPS run.
    """
    from frictionsim2d.core.mlip import check_d3_fits_box

    # 16.0 cutoff + 0.3 skin = 16.3 required
    with pytest.raises(ValueError, match='(?i)d3_cutoff'):
        check_d3_fits_box(box_x=15.9, box_y=16.5, d3_cutoff=16.0, skin=0.3)

    # a box between the cutoff (16.0) and cutoff + skin (16.3) is still refused
    with pytest.raises(ValueError, match='(?i)d3_cutoff'):
        check_d3_fits_box(box_x=16.1, box_y=20.0, d3_cutoff=16.0, skin=0.3)

    # exactly cutoff + skin fits
    check_d3_fits_box(box_x=16.3, box_y=16.3, d3_cutoff=16.0, skin=0.3)

    # comfortably larger box is fine
    check_d3_fits_box(box_x=19.1, box_y=22.1, d3_cutoff=16.0, skin=0.3)

    # a reduced cutoff rescues the small cell
    check_d3_fits_box(box_x=15.9, box_y=16.5, d3_cutoff=10.0, skin=0.3)

    # the message must say what to do about it
    with pytest.raises(ValueError, match='(?i)reduce|smaller|increase'):
        check_d3_fits_box(box_x=12.0, box_y=12.0, d3_cutoff=16.0, skin=0.3)
