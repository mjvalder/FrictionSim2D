"""The one LAMMPS type-list expander and group parser the other modules share."""
from frictionsim2d.analysis.sheet_geometry import read_settings
from frictionsim2d.postprocessing.snapshot_geometry import expand_lammps_types, parse_type_groups


def test_expand_lammps_types():
    assert expand_lammps_types('7 8 11*14') == {7, 8, 11, 12, 13, 14}
    assert expand_lammps_types('3') == {3}
    assert expand_lammps_types('') == set()


def test_parse_type_groups_first_definition_wins():
    text = 'group layer_1 type 1*2\ngroup tip type 9\ngroup layer_1 type 5\n'

    assert parse_type_groups(text) == {'layer_1': {1, 2}, 'tip': {9}}


def test_sheet_geometry_reads_element_and_layer_types_through_it():
    text = ('mass 1*2 32.06 #S\nmass 3 95.95 #Mo\n'
            'group layer_1 type 1*3\ngroup layer_2 type 4 5\n')

    elem, sigma, layers = read_settings(text)

    assert elem == {1: 'S', 2: 'S', 3: 'Mo'}
    assert sigma == {} and layers == {1: {1, 2, 3}, 2: {4, 5}}
