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


def test_a_legacy_copy_identical_to_the_packaged_file_yields_the_library_file(
        tmp_path, monkeypatch):
    """examples/materials/h-MoS2.cif left in a checkout, same bytes: the packaged file."""
    legacy = tmp_path / 'examples'
    monkeypatch.setattr(paths, 'LEGACY_EXAMPLES_DIR', legacy)
    packaged = packaged_dir('materials') / 'h-MoS2.cif'
    _put(legacy / 'materials', 'h-MoS2.cif', packaged.read_text(encoding='utf-8'))
    assert find_existing('examples/materials/h-MoS2.cif', None) == packaged


def test_a_users_own_file_in_the_legacy_dir_wins_with_warning(tmp_path, monkeypatch, caplog):
    """Different bytes under a packaged name are the user's own file: never replaced."""
    legacy = tmp_path / 'examples'
    monkeypatch.setattr(paths, 'LEGACY_EXAMPLES_DIR', legacy)
    own = _put(legacy / 'materials', 'h-MoS2.cif', 'my edited cell')
    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        found = find_existing('examples/materials/h-MoS2.cif', None)
    assert found is not None and found.resolve() == own.resolve()
    assert 'Deprecated path' in caplog.text


def _fake_install(root: Path) -> Path:
    """A package directory at ``root`` holding the two library files used below."""
    package = root / 'frictionsim2d'
    for rel in ('materials/Si.cif', 'potentials/sw/Si.sw'):
        src = packaged_dir(rel.split('/')[0]) / rel.split('/', 1)[1]
        dest = package / 'data' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    return package


def test_a_config_dumped_on_one_install_rebuilds_on_another(tmp_path, monkeypatch):
    """config.json names library files portably, so a stored config re-resolves
    with the package somewhere else, and config_hash does not move."""
    from frictionsim2d.core.config import ComponentConfig
    from frictionsim2d.data.provenance import config_hash

    machine_a = _fake_install(tmp_path / 'a' / 'site-packages')
    monkeypatch.setattr(paths, '_PACKAGE_DIR', machine_a)
    built = ComponentConfig(mat='Si', pot_type='sw', pot_path='potentials/sw/Si.sw',
                            cif_path='materials/Si.cif')
    assert built.cif_path == str(machine_a / 'data' / 'materials' / 'Si.cif')
    stored = built.model_dump(mode='json')
    json_on_a = built.model_dump_json()
    assert stored['cif_path'] == 'materials/Si.cif'
    assert stored['pot_path'] == 'potentials/sw/Si.sw'

    machine_b = _fake_install(tmp_path / 'b' / 'env' / 'lib')
    monkeypatch.setattr(paths, '_PACKAGE_DIR', machine_b)
    rebuilt = ComponentConfig.model_validate(stored)
    assert rebuilt.pot_path == str(machine_b / 'data' / 'potentials' / 'sw' / 'Si.sw')
    assert config_hash(rebuilt.model_dump(mode='json')) == config_hash(stored)
    assert rebuilt.model_dump_json() == json_on_a


def test_an_absolute_path_from_another_install_still_resolves(tmp_path, monkeypatch):
    """A config.json written before the portable spelling holds an absolute
    library path; once that install is gone it maps onto this one's library."""
    here = _fake_install(tmp_path / 'here')
    monkeypatch.setattr(paths, '_PACKAGE_DIR', here)
    gone = '/old/env/lib/python3.11/site-packages/frictionsim2d/data/materials/Si.cif'
    assert get_material_path(gone) == here / 'data' / 'materials' / 'Si.cif'
    assert paths.library_name(here / 'data' / 'potentials' / 'sw' / 'Si.sw') == \
        'potentials/sw/Si.sw'
    assert paths.library_name(tmp_path / 'elsewhere.cif') is None


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


def test_a_differing_legacy_file_keeps_its_real_path(tmp_path, monkeypatch):
    """A file in the legacy examples/ folder that differs from the library file of
    the same name is stored by its real path, never as ``materials/<name>`` (which
    would reload the packaged file), and reloads to its own bytes."""
    from frictionsim2d.core.config import ComponentConfig

    legacy = tmp_path / 'examples'
    monkeypatch.setattr(paths, 'LEGACY_EXAMPLES_DIR', legacy)
    own = _put(legacy / 'materials', 'Si.cif', 'my edited Si cell')
    assert paths.library_name(own) is None
    built = ComponentConfig(mat='Si', pot_type='sw', pot_path='potentials/sw/Si.sw',
                            cif_path=str(own))
    stored = built.model_dump(mode='json')
    assert stored['cif_path'] == str(own)
    rebuilt = ComponentConfig.model_validate(stored)
    assert Path(rebuilt.cif_path).read_bytes() == own.read_bytes()


def test_a_pack_file_shadowed_by_a_different_packaged_one_keeps_its_real_path():
    shadowed = _put(materials_pack_dir(), 'h-MoS2.cif', 'pack copy, not the packaged cell')
    assert paths.library_name(shadowed) is None
    unique = _put(materials_pack_dir(), 'amor_Z__0.lmp', 'pack only')
    assert paths.library_name(unique) == 'materials/amor_Z__0.lmp'
