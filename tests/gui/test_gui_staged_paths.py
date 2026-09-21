"""``absolutize_staged_paths`` fixes the sweep/staging path bug.

``ConfigDocument.apply_material_directories`` deliberately stores relative
paths (``cif/{mat}.cif``) so a saved config travels with its run directory.
But ``PreviewController``/``GenerateController`` copy the config into a
*different* directory (a preview cache dir, a ``_generate`` staging dir)
before building it, and ``search_paths`` resolves a relative name against
the config file's own directory -- which, once staged, is no longer where
the user's cif/pot files live.

``absolutize_staged_paths`` fixes this at the staging boundary: it resolves
path-like values against the *original* document's directory, using the same
search the builders use, and returns a copy -- the input mapping (and
whatever the document has saved to disk) is left untouched.
"""

from __future__ import annotations

from pathlib import Path

from src.gui.controllers.staged_paths import absolutize_staged_paths


def _make_material_dirs(tmp_path: Path) -> Path:
    """A config directory with a `cif/` and `sw_lammps/` subdirectory."""
    config_dir = tmp_path / 'myruns'
    (config_dir / 'cif').mkdir(parents=True)
    (config_dir / 'cif' / 'h_MoS2.cif').write_text('cif contents')
    (config_dir / 'sw_lammps').mkdir()
    (config_dir / 'sw_lammps' / 'h_MoS2.sw').write_text('pot contents')
    (config_dir / 'materials.txt').write_text('h_MoS2\n')
    return config_dir


class TestPlainRelativePaths:
    """A plain (non-templated) relative path resolves against base_dir."""

    def test_cif_path_resolves_to_the_original_directory(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'cif_path': 'cif/h_MoS2.cif'}}

        result = absolutize_staged_paths(config, config_dir)

        resolved = Path(result['2D']['cif_path'])
        assert resolved.is_absolute()
        assert resolved == (config_dir / 'cif' / 'h_MoS2.cif').resolve()

    def test_pot_path_resolves_to_the_original_directory(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'pot_path': 'sw_lammps/h_MoS2.sw'}}

        result = absolutize_staged_paths(config, config_dir)

        resolved = Path(result['2D']['pot_path'])
        assert resolved.is_absolute()
        assert resolved == (config_dir / 'sw_lammps' / 'h_MoS2.sw').resolve()

    def test_resolves_even_when_staged_into_an_unrelated_directory(self, tmp_path):
        """The whole point: the result must not depend on where the config is
        about to be written -- only on where it actually lived."""
        config_dir = _make_material_dirs(tmp_path)
        unrelated = tmp_path / 'somewhere' / 'else' / 'entirely'
        unrelated.mkdir(parents=True)

        config = {
            '2D': {
                'cif_path': 'cif/h_MoS2.cif',
                'pot_path': 'sw_lammps/h_MoS2.sw',
            },
        }
        result = absolutize_staged_paths(config, config_dir)

        assert Path(result['2D']['cif_path']).exists()
        assert Path(result['2D']['pot_path']).exists()
        # And, crucially, resolving the *staged* value against `unrelated`
        # (what search_paths would do once written there) must still work,
        # because it is now absolute.
        from src.core.utils.paths import _find_existing  # noqa: PLC2701
        assert _find_existing(result['2D']['cif_path'], unrelated) is not None
        assert _find_existing(result['2D']['pot_path'], unrelated) is not None


class TestTemplatedRelativePaths:
    """A `{mat}`-templated relative path resolves its directory only."""

    def test_templated_cif_path_resolves_directory_and_keeps_placeholder(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'cif_path': 'cif/{mat}.cif'}}

        result = absolutize_staged_paths(config, config_dir)

        resolved = result['2D']['cif_path']
        assert '{mat}' in resolved, 'the placeholder must survive -- it is not substituted yet'
        assert Path(resolved).is_absolute()
        assert Path(resolved.replace('{mat}', 'h_MoS2')) == \
            (config_dir / 'cif' / 'h_MoS2.cif').resolve()

    def test_templated_pot_path_resolves_directory_and_keeps_placeholder(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'pot_path': 'sw_lammps/{mat}.sw'}}

        result = absolutize_staged_paths(config, config_dir)

        resolved = result['2D']['pot_path']
        assert '{mat}' in resolved
        assert Path(resolved).is_absolute()
        assert Path(resolved.replace('{mat}', 'h_MoS2')) == \
            (config_dir / 'sw_lammps' / 'h_MoS2.sw').resolve()

    def test_templated_path_still_resolves_once_staged_elsewhere(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        unrelated = tmp_path / 'staging' / 'xyz'
        unrelated.mkdir(parents=True)

        config = {'2D': {'cif_path': 'cif/{mat}.cif'}}
        result = absolutize_staged_paths(config, config_dir)
        substituted = result['2D']['cif_path'].replace('{mat}', 'h_MoS2')

        from src.core.utils.paths import _find_existing  # noqa: PLC2701
        assert _find_existing(substituted, unrelated) is not None


class TestMaterialsList:
    """A relative `materials_list` has the same staging problem."""

    def test_materials_list_resolves_to_the_original_directory(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'materials_list': 'materials.txt'}}

        result = absolutize_staged_paths(config, config_dir)

        resolved = Path(result['2D']['materials_list'])
        assert resolved.is_absolute()
        assert resolved == (config_dir / 'materials.txt').resolve()

    def test_materials_list_still_resolves_once_staged_elsewhere(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        unrelated = tmp_path / 'staging' / 'xyz'
        unrelated.mkdir(parents=True)

        config = {'2D': {'materials_list': 'materials.txt'}}
        result = absolutize_staged_paths(config, config_dir)

        from src.core.utils.paths import _find_existing  # noqa: PLC2701
        assert _find_existing(result['2D']['materials_list'], unrelated) is not None


class TestPassThroughs:
    """Values this module must never touch."""

    def test_absolute_path_is_left_unchanged(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        absolute = str((config_dir / 'cif' / 'h_MoS2.cif').resolve())
        config = {'2D': {'cif_path': absolute}}

        result = absolutize_staged_paths(config, config_dir)

        assert result['2D']['cif_path'] == absolute

    def test_absolute_templated_path_is_left_unchanged(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        absolute = str((config_dir / 'cif').resolve()) + '/{mat}.cif'
        config = {'2D': {'cif_path': absolute}}

        result = absolutize_staged_paths(config, config_dir)

        assert result['2D']['cif_path'] == absolute

    def test_unresolvable_path_is_left_unchanged(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'cif_path': 'nonexistent_dir/h_MoS2.cif'}}

        result = absolutize_staged_paths(config, config_dir)

        assert result['2D']['cif_path'] == 'nonexistent_dir/h_MoS2.cif'

    def test_unresolvable_templated_path_is_left_unchanged(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'cif_path': 'nonexistent_dir/{mat}.cif'}}

        result = absolutize_staged_paths(config, config_dir)

        assert result['2D']['cif_path'] == 'nonexistent_dir/{mat}.cif'

    def test_none_base_dir_leaves_relative_values_untouched(self, tmp_path):
        del tmp_path
        config = {'2D': {'cif_path': 'cif/h_MoS2.cif', 'materials_list': 'materials.txt'}}

        result = absolutize_staged_paths(config, None)

        assert result['2D']['cif_path'] == 'cif/h_MoS2.cif'
        assert result['2D']['materials_list'] == 'materials.txt'

    def test_missing_keys_and_non_string_values_are_ignored(self):
        config = {'2D': {'x': 60, 'y': 60}, 'general': {'temp': 300}}

        result = absolutize_staged_paths(config, Path('/somewhere'))

        assert result == config


class TestInputIsNotMutated:
    """The original document's in-memory config must survive untouched --
    only the copy handed to the staged config.ini may change."""

    def test_original_mapping_is_not_mutated(self, tmp_path):
        config_dir = _make_material_dirs(tmp_path)
        config = {'2D': {'cif_path': 'cif/h_MoS2.cif'}}

        absolutize_staged_paths(config, config_dir)

        assert config['2D']['cif_path'] == 'cif/h_MoS2.cif'

    def test_multiple_sections_each_resolve_independently(self, tmp_path):
        """tip/sub/flake sections carry the same `cif_path`/`pot_path` keys as
        `2D` -- each must resolve on its own."""
        config_dir = _make_material_dirs(tmp_path)
        config = {
            '2D': {'cif_path': 'cif/h_MoS2.cif'},
            'sub': {'cif_path': 'cif/h_MoS2.cif'},
        }

        result = absolutize_staged_paths(config, config_dir)

        assert Path(result['2D']['cif_path']).is_absolute()
        assert Path(result['sub']['cif_path']).is_absolute()


class TestLaunchedFromTheConfigDirectory:
    """The GUI started from the config's own directory.

    `search_paths` tries a name against the cwd first, so there `Si.sw` is
    "found" as the relative `Si.sw` -- and staged unchanged. The build
    subprocess runs from elsewhere and cannot find it.
    """

    def test_plain_value_found_via_cwd_is_staged_absolute(self, tmp_path, monkeypatch):
        config_dir = _make_material_dirs(tmp_path)
        (config_dir / 'Si.sw').write_text('pot contents')
        monkeypatch.chdir(config_dir)

        result = absolutize_staged_paths({'tip': {'pot_path': 'Si.sw'}}, config_dir)

        assert result['tip']['pot_path'] == str((config_dir / 'Si.sw').resolve())

    def test_templated_value_found_via_cwd_is_staged_absolute(self, tmp_path, monkeypatch):
        config_dir = _make_material_dirs(tmp_path)
        monkeypatch.chdir(config_dir)

        result = absolutize_staged_paths({'2D': {'cif_path': 'cif/{mat}.cif'}}, config_dir)

        assert result['2D']['cif_path'] == str((config_dir / 'cif').resolve() / '{mat}.cif')
