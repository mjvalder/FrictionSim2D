"""Axis labels come from the registry (Q3).

`Observable.units` and `role_key` had zero consumers in `postprocessing/`:
every axis label was hand-written in the plot JSON, so a sheet plot could
say "Lateral Force (nN)" over a column the template emitted in eV/A.
"""

from __future__ import annotations

import pytest

from frictionsim2d.postprocessing.labels import display_name, label_for, unit_for


@pytest.mark.parametrize('column, family, expected', [
    ('nf', 'afm', 'Normal Load (nN)'),
    ('nf', 'sheetonsheet', 'Normal Load (nN)'),
    # a flake slide applies a real normal load (pressure x projected area), so
    # its COF is lateral force over applied load, exactly as the sheet's is
    ('nf', 'flake', 'Normal Load (nN)'),
    ('cof', 'flake', 'Coefficient of Friction'),
    ('lfx', 'afm', 'Lateral Force x (nN)'),
    ('lfy', 'sheetonsheet', 'Lateral Force y (nN)'),
    ('cof', 'afm', 'Coefficient of Friction'),
    ('lf', 'afm', 'Lateral Force (nN)'),
    ('lateral_force', 'afm', 'Lateral Force (nN)'),
    ('cof_slope', 'afm', 'COF (slope of LF vs NF)'),
    ('time', 'afm', 'Timestep (step)'),
    ('etotal', 'afm', 'Total Energy (eV)'),
    ('temp_sheet_3d', 'afm', 'Group Temperature (K)'),
    ('n_contact', 'afm', 'Contact Count (count)'),
    ('gap_min', 'afm', 'Minimum Gap (A)'),
])
def test_label_for_known_columns(column, family, expected):
    assert label_for(column, family) == expected


def test_a_dimensionless_column_gets_no_parenthetical():
    assert '(' not in label_for('cof', 'afm')


def test_layer_columns_are_labelled_for_any_layer():
    assert label_for('comz_L4', 'sheetonsheet') == 'Layer 4 COM z (A)'
    assert label_for('comx_L1', 'afm') == 'Layer 1 COM x (A)'


def test_the_same_name_gets_the_right_family_entry():
    """`lfx` is v_fx_spr for AFM and v_xfrict for the sheet -- two registry
    entries, one name. Both are nN, so the label agrees; the lookup must
    still find the family's own entry rather than the first match."""
    assert unit_for('lfx', 'afm') == 'nN'
    assert unit_for('lfx', 'flake') == 'nN'
    assert unit_for('raw_lfx', 'afm') == 'nN'


def test_an_unknown_column_is_prettified_not_an_error():
    assert label_for('wobble_factor', 'afm') == 'Wobble Factor'
    assert unit_for('wobble_factor', 'afm') is None
    assert display_name('wobble_factor', 'afm') == 'Wobble Factor'


def test_a_column_absent_from_a_family_falls_back_to_its_own_name():
    """`tip_min_z` is declared for 'afm' only."""
    assert unit_for('tip_min_z', 'afm') == 'A'
    assert unit_for('tip_min_z', 'sheetonsheet') is None
