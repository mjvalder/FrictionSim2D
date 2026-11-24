"""Small naming helpers: layer-aware ordering and numbered-section keys."""
from frictionsim2d.run.naming import (
    flake_material_token, layer_aware_path_sort_key, material_token, primary_sheet,
)


def test_layer_segments_sort_numerically():
    paths = ['a/L10', 'a/L2', 'a/L1', 'a/L3']
    assert sorted(paths, key=layer_aware_path_sort_key) == ['a/L1', 'a/L2', 'a/L3', 'a/L10']


def test_non_layer_segments_sort_after_layer_segments_at_the_same_depth():
    paths = ['a/zz', 'a/L2', 'a/aa']
    assert sorted(paths, key=layer_aware_path_sort_key) == ['a/L2', 'a/aa', 'a/zz']


def test_numbered_sheet_sections_name_the_material_in_numeric_order():
    run = {'2D-10': {'mat': 'C'}, '2D-2': {'mat': 'B'}, '2D-1': {'mat': 'A'}}
    assert material_token(run) == 'A+B+C'
    assert primary_sheet(run) == {'mat': 'A'}


def test_a_non_numeric_numbered_suffix_sorts_last():
    run = {'2D-x': {'mat': 'Z'}, '2D-1': {'mat': 'A'}}
    assert material_token(run) == 'A+Z'


def test_numbered_flake_sections_name_their_materials_in_numeric_order():
    run = {'Flake-2': {'mat': 'B'}, 'Flake-10': {'mat': 'C'}, 'Flake-1': {'mat': 'A'}}
    assert flake_material_token(run) == 'A+B+C'
