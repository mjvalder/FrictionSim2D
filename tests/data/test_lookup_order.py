"""Library lookup order: package data, materials pack, user library, legacy examples/."""
import logging
from pathlib import Path

import pytest

from frictionsim2d.core.utils import paths
from frictionsim2d.core.utils.paths import (
    LEGACY_EXAMPLES_DIR, find_existing, get_material_path, get_potential_path, library_roots,
    materials_pack_dir, packaged_dir)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    """An empty data home and a fresh warned-once set per test."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))
    monkeypatch.setattr(paths, '_WARNED', set())
    monkeypatch.chdir(tmp_path)


def _put(folder: Path, name: str, text: str = 'x') -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def test_the_library_lives_in_the_package():
    assert packaged_dir('materials') == REPO / 'frictionsim2d' / 'data' / 'materials'
    assert (packaged_dir('materials') / 'h-MoS2.cif').is_file()
    assert (packaged_dir('potentials') / 'sw' / 'MoS2_wen.sw').is_file()
    assert not (REPO / 'examples' / 'materials').exists()
    assert not (REPO / 'examples' / 'potentials').exists()


def test_root_order_is_packaged_pack_user_legacy():
    assert [label for label, _ in library_roots('materials')] == [
        'packaged', 'pack', 'user', 'legacy']
    # The pack and the user library hold materials (amorphous blocks) only.
    assert [label for label, _ in library_roots('potentials')] == ['packaged', 'legacy']
    assert [label for label, _ in library_roots('materials', legacy=False)] == [
        'packaged', 'pack', 'user']


def test_the_quench_library_reads_the_same_roots():
    from frictionsim2d.builders.construction import amorphous
    assert amorphous.library_roots() == library_roots('materials', legacy=False)
    assert amorphous.materials_pack_dir() == materials_pack_dir()


@pytest.mark.parametrize('name', ['materials/h-MoS2.cif', 'h-MoS2.cif'])
def test_a_library_name_resolves_to_package_data(name, caplog):
    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        assert get_material_path(name) == packaged_dir('materials') / 'h-MoS2.cif'
    assert not caplog.records


def test_a_potential_library_name_resolves_with_or_without_its_kind():
    want = packaged_dir('potentials') / 'sw' / 'Si.sw'
    assert get_potential_path('potentials/sw/Si.sw') == want
    assert get_potential_path('sw/Si.sw') == want


def test_legacy_examples_path_resolves_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        found = get_material_path('examples/materials/h-MoS2.cif')
        get_material_path('examples/materials/h-MoS2.cif')     # warned once only
        pot = get_potential_path('examples/potentials/sw/sw_lammps/h-WS2.sw')
    assert found == packaged_dir('materials') / 'h-MoS2.cif'
    assert pot == packaged_dir('potentials') / 'sw' / 'sw_lammps' / 'h-WS2.sw'
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert all('Deprecated path' in m for m in messages)
    assert "'examples/materials/h-MoS2.cif'" in messages[0]


def test_an_absolute_path_into_the_old_examples_dir_resolves_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        found = get_material_path(str(LEGACY_EXAMPLES_DIR / 'materials' / 'Si.cif'))
    assert found == packaged_dir('materials') / 'Si.cif'
    assert 'Deprecated path' in caplog.text


def test_a_file_only_in_the_legacy_dir_is_found_last_with_warning(tmp_path, monkeypatch, caplog):
    legacy = tmp_path / 'old_examples'
    monkeypatch.setattr(paths, 'LEGACY_EXAMPLES_DIR', legacy)
    own = _put(legacy / 'materials', 'mine.cif')
    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        assert get_material_path('materials/mine.cif') == own
    assert 'Deprecated path' in caplog.text


def test_a_legacy_dir_hit_as_given_still_loses_to_the_library(tmp_path, monkeypatch):
    """examples/materials/h-MoS2.cif left in a checkout: the packaged file wins."""
    legacy = tmp_path / 'examples'
    monkeypatch.setattr(paths, 'LEGACY_EXAMPLES_DIR', legacy)
    _put(legacy / 'materials', 'h-MoS2.cif', 'stale')
    assert find_existing('examples/materials/h-MoS2.cif', None) == \
        packaged_dir('materials') / 'h-MoS2.cif'


def test_the_pack_comes_after_package_data_and_before_the_user_library(tmp_path):
    pack = materials_pack_dir()
    user = tmp_path / 'xdg' / 'FrictionSim2D' / 'amorphous'
    _put(pack, 'h-MoS2.cif', 'pack copy')                  # shadowed by package data
    in_pack = _put(pack, 'amor_X__0.lmp', 'pack')
    _put(user, 'amor_X__0.lmp', 'user')
    in_user = _put(user, 'amor_Y__0.lmp', 'user')
    assert get_material_path('h-MoS2.cif') == packaged_dir('materials') / 'h-MoS2.cif'
    assert get_material_path('amor_X__0.lmp') == in_pack
    assert get_material_path('materials/amor_Y__0.lmp') == in_user


def test_a_users_own_file_beats_the_library(tmp_path):
    mine = _put(tmp_path, 'h-MoS2.cif', 'my own')
    assert find_existing('h-MoS2.cif', None) == Path('h-MoS2.cif')
    assert get_material_path('h-MoS2.cif', base_dir=tmp_path).read_text() == mine.read_text()


def test_a_missing_name_is_still_missing():
    assert find_existing('materials/not_a_material.cif', None) is None
    with pytest.raises(FileNotFoundError, match='material library'):
        get_material_path('not_a_material.cif')
