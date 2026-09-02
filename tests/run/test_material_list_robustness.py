"""Material lists are hand-maintained text files, so reading them must tolerate
the ways people actually write them -- and one unresolvable material must not
take down a sweep of a hundred.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.run.placeholders import has_template_marker, substitute_materials
from src.run.expand import (
    expand_config_sweeps,
    preflight_materials,
    read_material_entries,
    read_material_list,
)


@pytest.fixture(name='list_file')
def _list_file(tmp_path):
    def write(content: str) -> Path:
        path = tmp_path / 'materials.txt'
        path.write_text(content, encoding='utf-8')
        return path
    return write


class TestReadMaterialList:

    def test_plain_one_per_line(self, list_file):
        assert read_material_list(list_file("MoS2\nWS2\n")) == ['MoS2', 'WS2']

    def test_blank_lines_and_surrounding_whitespace(self, list_file):
        content = "\n  MoS2  \n\n\t WS2\t\n   \n"
        assert read_material_list(list_file(content)) == ['MoS2', 'WS2']

    def test_whole_line_and_trailing_comments(self, list_file):
        content = "# TMDs\nMoS2  # the good one\n#WS2\nWSe2\n"
        assert read_material_list(list_file(content)) == ['MoS2', 'WSe2']

    def test_comma_separated_entries(self, list_file):
        assert read_material_list(list_file("MoS2, WS2\nWSe2\n")) == ['MoS2', 'WS2', 'WSe2']

    def test_duplicates_are_dropped_preserving_order(self, list_file):
        assert read_material_list(list_file("WS2\nMoS2\nWS2\n")) == ['WS2', 'MoS2']

    def test_missing_file_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='not found'):
            read_material_list(tmp_path / 'nope.txt')

    def test_base_dir_resolves_a_relative_path(self, list_file, tmp_path, monkeypatch):
        list_file("MoS2\n")
        monkeypatch.chdir(tmp_path.parent)
        assert read_material_list('materials.txt', base_dir=tmp_path) == ['MoS2']

    def test_empty_file_yields_no_materials(self, list_file):
        assert read_material_list(list_file("\n#only a comment\n")) == []


class TestPreflightMaterials:

    @pytest.fixture(name='section')
    def _section(self, tmp_path):
        (tmp_path / 'MoS2.cif').write_text('', encoding='utf-8')
        (tmp_path / 'MoS2.sw').write_text('', encoding='utf-8')
        (tmp_path / 'WS2.cif').write_text('', encoding='utf-8')
        # WS2.sw deliberately absent.
        return {
            'cif_path': str(tmp_path / '{mat}.cif'),
            'pot_path': str(tmp_path / '{mat}.sw'),
        }

    def test_resolvable_material_is_usable(self, section):
        result = preflight_materials(['MoS2'], section)
        assert result.usable == ['MoS2']
        assert result.skipped == []

    def test_material_missing_one_file_is_skipped_with_a_reason(self, section):
        result = preflight_materials(['MoS2', 'WS2'], section)
        assert result.usable == ['MoS2']
        assert [m for m, _ in result.skipped] == ['WS2']
        assert 'pot_path' in result.skipped[0][1]

    def test_summary_names_every_skipped_material(self, section):
        result = preflight_materials(['MoS2', 'WS2'], section)
        assert 'WS2' in result.summary()

    def test_section_without_templates_skips_nothing(self):
        result = preflight_materials(['MoS2'], {})
        assert result.usable == ['MoS2']


class TestExpansionSkipsUnusableMaterials:

    @pytest.fixture(name='config')
    def _config(self, tmp_path):
        for name in ('MoS2', 'WS2'):
            (tmp_path / f'{name}.cif').write_text('', encoding='utf-8')
        (tmp_path / 'MoS2.sw').write_text('', encoding='utf-8')  # WS2.sw missing
        (tmp_path / 'materials.txt').write_text("MoS2\nWS2\n", encoding='utf-8')
        return {
            '2D': {
                'materials_list': str(tmp_path / 'materials.txt'),
                'mat': '{mat}',
                'cif_path': str(tmp_path / '{mat}.cif'),
                'pot_path': str(tmp_path / '{mat}.sw'),
            },
            'general': {'temp': 300},
        }

    def test_unusable_material_is_dropped_and_the_rest_expand(self, config, caplog):
        expanded = expand_config_sweeps(config)
        assert len(expanded) == 1
        assert expanded[0]['2D']['mat'] == 'MoS2'

    def test_a_warning_names_the_skipped_material(self, config, caplog):
        with caplog.at_level('WARNING'):
            expand_config_sweeps(config)
        assert 'WS2' in caplog.text

    def test_all_unusable_raises_rather_than_generating_nothing(self, config, tmp_path):
        (tmp_path / 'MoS2.sw').unlink()
        with pytest.raises(ValueError, match='could not be resolved|None of the'):
            expand_config_sweeps(config)

    def test_missing_list_file_warns_and_falls_back(self, config, caplog):
        config['2D']['materials_list'] = 'does-not-exist.txt'
        with caplog.at_level('WARNING'):
            expanded = expand_config_sweeps(config)
        assert 'not found' in caplog.text
        # No material expansion, but the config itself still passes through.
        assert len(expanded) == 1


class TestOrderedPairEntries:
    """A '+' line is one ordered 2-tuple, additive to the existing
    comma-means-multiple-singles behaviour, not a replacement for it."""

    def test_read_material_entries_pairs_and_singles(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2\nh-MoS2 + h-WS2\n# comment\nMoS2, WS2\nh-WS2+h-MoTe2\n')
        assert read_material_entries(f) == [
            ('h-MoS2',), ('h-MoS2', 'h-WS2'), ('MoS2',), ('WS2',),
            ('h-WS2', 'h-MoTe2')]

    def test_comma_lines_still_mean_single_materials(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('MoS2, WS2\nWSe2\n')
        assert read_material_list(f) == ['MoS2', 'WS2', 'WSe2']   # pinned behaviour

    def test_read_material_list_rejects_a_pair_line(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\n')
        with pytest.raises(ValueError, match='h-MoS2 \\+ h-WS2'):
            read_material_list(f)

    def test_a_comma_on_a_plus_line_is_rejected(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2, h-MoTe2\n')
        with pytest.raises(ValueError, match='h-MoS2 \\+ h-WS2, h-MoTe2'):
            read_material_entries(f)

    def test_more_than_two_members_is_rejected(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2 + h-MoTe2\n')
        with pytest.raises(ValueError, match='h-MoS2 \\+ h-WS2 \\+ h-MoTe2'):
            read_material_entries(f)

    def test_duplicate_pairs_dedupe_and_warn(self, tmp_path, caplog):
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\nh-MoS2 + h-WS2\n')
        with caplog.at_level('WARNING'):
            entries = read_material_entries(f)
        assert entries == [('h-MoS2', 'h-WS2')]
        assert 'h-MoS2' in caplog.text and 'h-WS2' in caplog.text


class TestTemplateMarkers:

    def test_template_markers_recognised(self):
        assert has_template_marker('cif/{mat1}.cif')
        assert substitute_materials('c/{mat1}.cif', 'a', 'b') == 'c/a.cif'
        assert substitute_materials('c/{mat2}.sw', 'a', 'b') == 'c/b.sw'
        assert substitute_materials('c/{mat}.cif', 'a') == 'c/a.cif'

    def test_plain_path_has_no_marker(self):
        assert not has_template_marker('cif/h-MoS2.cif')
