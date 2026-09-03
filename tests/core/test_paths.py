from pathlib import Path

import pytest

from src.core.utils.paths import get_material_path, get_potential_path, search_paths


def test_search_paths_tries_base_dir_before_climbing_to_parents(tmp_path):
    """search_paths' documented order is: the name as given (cwd-relative or
    absolute) first, then base_dir, then base_dir's parents up to
    PATH_SEARCH_MAX_PARENTS -- base_dir is NOT tried before the bare name.
    (Adapted from the brief's `test_search_paths_prefers_base_dir_over_cwd`,
    which asserted `cands[0] == tmp_path / 'h-MoS2.cif'`; that fails because
    candidates[0] is always `Path(name)` per paths.py:19-50's own docstring
    -- a wrong assumption in the test, not a src bug.)"""
    cands = search_paths('h-MoS2.cif', base_dir=tmp_path)
    assert cands[0] == Path('h-MoS2.cif')
    assert cands[1] == tmp_path / 'h-MoS2.cif'


def test_get_material_path_finds_a_file_relative_to_base_dir(tmp_path):
    (tmp_path / 'cif').mkdir()
    f = tmp_path / 'cif' / 'x.cif'; f.write_text('data_x')
    assert get_material_path('cif/x.cif', base_dir=tmp_path) == f


def test_get_material_path_missing_file_names_the_material():
    with pytest.raises(FileNotFoundError, match="'no-such-material'"):
        get_material_path('no-such-material')


def test_get_potential_path_missing_file_mentions_examples_potentials():
    with pytest.raises(FileNotFoundError, match='examples/potentials'):
        get_potential_path('nothing.sw')


def test_get_potential_path_absolute_path_is_returned_as_is(tmp_path):
    f = tmp_path / 'p.sw'; f.write_text('# sw')
    assert get_potential_path(str(f)) == f
