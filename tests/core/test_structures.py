import pytest

from frictionsim2d.core.utils.structures import cifread, count_atomtypes


def test_cifread_returns_lattice_and_elements(materials_path):
    data = cifread(materials_path / 'h-MoS2.cif')
    assert set(data['elements']) == {'Mo', 'S'}
    assert data['lat_a'] == pytest.approx(3.16, abs=0.1)


def test_count_atomtypes_single_type_potentials_map_each_element_to_one(tmp_path):
    pot = tmp_path / 'ffield.reaxff'; pot.write_text('anything')
    assert count_atomtypes(pot, ['Mo', 'S'], 'reaxff') == {'Mo': 1, 'S': 1}


def test_count_atomtypes_reads_the_highest_numbered_element_from_sw_triplets(tmp_path):
    pot = tmp_path / 'x.sw'
    pot.write_text('# comment\n\nMo1 S1 S1 1.0\nMo2 S2 S2 1.0\nMo1 S3 S1 1.0\n')
    assert count_atomtypes(pot, ['Mo', 'S'], 'sw') == {'Mo': 2, 'S': 3}


def test_count_atomtypes_skips_comments_and_blank_lines(tmp_path):
    pot = tmp_path / 'x.sw'; pot.write_text('#Mo9 S9 S9\n\n   \nMo S S 1\n')
    assert count_atomtypes(pot, ['Mo', 'S'], 'sw') == {'Mo': 1, 'S': 1}

