"""The preview parsers, checked against real generated packages.

The preview's correctness claim is that it shows what LAMMPS will read, which
holds only if these parsers agree with what the builders actually wrote. So the
important tests here generate packages and parse the result, rather than
parsing hand-written fixtures.

Neither parser imports Qt or ovito, so this module runs anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from frictionsim2d.run.artifacts import discover_preview_target, generate_package
from frictionsim2d.gui.preview.init_script import (
    Placement,
    parse_assembly,
    read_data_file_box,
)
from frictionsim2d.gui.preview.settings_file import expand_type_range, parse_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

#: (config, model, expected component groups)
CASES = [
    ('indent_config.ini', 'indent', {'sub', 'tip', '2D'}),
    ('afm_flake_config.ini', 'afm', {'sub', 'tip', '2D', 'flake'}),
    ('pes_tip_config.ini', 'pes_tip', {'sub', 'tip', '2D'}),
    ('pes_sheet_config.ini', 'pes_sheet', {'2D'}),
]


@pytest.fixture(scope='module', name='previews')
def _previews(tmp_path_factory):
    """Generate each case once and locate its preview target."""
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        out = {}
        for name, model, _ in CASES:
            result = generate_package(
                EXAMPLES / name, model,
                output_root=tmp_path_factory.mktemp(model), generate_hpc=False,
            )
            assert result.ok, result.render()
            target = discover_preview_target(result.simulation_root, result.created[0])
            out[name] = target
        return out
    finally:
        os.chdir(cwd)


class TestAssemblyParsing:

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_every_component_resolves(self, name, model, groups, previews):
        target = previews[name]
        assembly = parse_assembly(target.init_script, target.simulation_root)
        resolved = assembly.resolved_placements()
        assert resolved, "no component files resolved"
        assert len(resolved) == len(assembly.placements)

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_a_cell_is_always_obtainable(self, name, model, groups, previews):
        """From the region line, or from the data file header when there is none."""
        target = previews[name]
        assembly = parse_assembly(target.init_script, target.simulation_root)
        cell = assembly.effective_cell()
        assert cell is not None
        xlo, xhi, ylo, yhi, zlo, zhi = cell
        assert xhi > xlo and yhi > ylo and zhi > zlo

    def test_placed_form_has_groups_shifts_and_offsets(self, previews):
        assembly = parse_assembly(previews['indent_config.ini'].init_script,
                                  previews['indent_config.ini'].simulation_root)
        by_group = {p.group: p for p in assembly.placements}
        assert set(by_group) == {'sub', 'tip', '2D'}
        # The substrate anchors the box and is never shifted.
        assert by_group['sub'].shift == (0.0, 0.0, 0.0)
        assert by_group['sub'].type_offset == 0
        # The tip is centred laterally and lifted clear of the sheet.
        assert by_group['tip'].shift[0] > 0 and by_group['tip'].shift[2] > 0
        assert by_group['tip'].type_offset > 0
        # Types must not collide: each component starts after the previous.
        offsets = [p.type_offset for p in assembly.placements]
        assert offsets == sorted(offsets)

    def test_preassembled_form_is_recognised(self, previews):
        target = previews['pes_sheet_config.ini']
        assembly = parse_assembly(target.init_script, target.simulation_root)
        assert assembly.is_preassembled
        assert assembly.cell is None
        assert assembly.effective_cell() is not None, "cell must come from the data file"

    def test_runtime_state_files_are_flagged_not_resolved(self, previews):
        """slide scripts read data/load_$(v_find)N.data -- a LAMMPS variable."""
        target = previews['indent_config.ini']
        slide = next(target.init_script.parent.glob('slide_*.in'))
        assembly = parse_assembly(slide, target.simulation_root)
        assert assembly.placements
        assert all(p.is_runtime_state for p in assembly.placements)
        assert assembly.resolved_placements() == []


class TestPlacementGrammar:
    """Directive parsing, isolated from generation."""

    def _parse(self, tmp_path, body):
        script = tmp_path / 'x.in'
        script.write_text(body, encoding='utf-8')
        return parse_assembly(script, tmp_path)

    def test_bare_read_data(self, tmp_path):
        a = self._parse(tmp_path, 'read_data build/x.lmp\n')
        assert a.placements == [Placement(file='build/x.lmp')]

    def test_full_directive(self, tmp_path):
        a = self._parse(
            tmp_path,
            'read_data b/t.lmp add append shift 1.5 2.5 3.5 group tip offset 3 0 0 0 0\n')
        p = a.placements[0]
        assert (p.file, p.shift, p.group, p.type_offset) == \
            ('b/t.lmp', (1.5, 2.5, 3.5), 'tip', 3)

    def test_extra_keywords_before_add_are_ignored(self, tmp_path):
        a = self._parse(
            tmp_path,
            'read_data b/x.lmp extra/atom/types 1 add append group bot nocoeff\n')
        assert a.placements[0].group == 'bot'
        assert a.placements[0].file == 'b/x.lmp'

    def test_region_with_units_suffix(self, tmp_path):
        a = self._parse(tmp_path, 'region box block 0 10 0 20 -5 30 units box\n')
        assert a.cell == (0.0, 10.0, 0.0, 20.0, -5.0, 30.0)

    def test_comments_are_ignored(self, tmp_path):
        a = self._parse(tmp_path, '# read_data not/real.lmp\nread_data real.lmp\n')
        assert [p.file for p in a.placements] == ['real.lmp']

    def test_first_region_wins(self, tmp_path):
        a = self._parse(tmp_path,
                        'region box block 0 1 0 1 0 1\nregion other block 0 9 0 9 0 9\n')
        assert a.cell == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


class TestIdentityParsing:

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_component_groups_match_the_model(self, name, model, groups, previews):
        identity = parse_identity(previews[name].settings_file)
        assert set(identity.component_groups) == groups

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_mass_spans_are_disjoint(self, name, model, groups, previews):
        """Overlapping spans mean an element's types get another's mass; fixed in
        9d6829f, pinned here."""
        identity = parse_identity(previews[name].settings_file)
        assert identity.overlaps == [], identity.overlaps

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_every_grouped_type_has_an_element(self, name, model, groups, previews):
        identity = parse_identity(previews[name].settings_file)
        for types in identity.component_groups.values():
            for type_id in types:
                assert type_id in identity.elements, type_id

    def test_flake_types_keep_their_own_element(self, previews):
        """The case that exposed the overlapping-span bug: Mo = {7, 9, 10}."""
        identity = parse_identity(previews['afm_flake_config.ini'].settings_file)
        flake_types = identity.component_groups['flake']
        elements = {identity.elements[t] for t in flake_types}
        assert 'Mo' in elements and 'S' in elements

    def test_group_of_maps_types_to_components(self, previews):
        identity = parse_identity(previews['indent_config.ini'].settings_file)
        sub_types = identity.component_groups['sub']
        assert identity.group_of(sub_types[0]) == 'sub'

    def test_region_groups_are_not_components(self, previews):
        """sub_fix and sub_thermo are subsets of sub_all, not separate parts."""
        identity = parse_identity(previews['indent_config.ini'].settings_file)
        assert 'sub_fix' in identity.groups
        assert 'sub_fix' not in identity.component_groups

    def test_layer_groups_appear_for_multilayer_sheets(self, previews):
        identity = parse_identity(previews['pes_sheet_config.ini'].settings_file)
        assert set(identity.layer_groups) == {'layer_1', 'layer_2'}

    def test_legend_can_exclude_types_with_no_atoms(self, previews):
        """The virtual driving atom is declared but contributes no atoms."""
        identity = parse_identity(previews['indent_config.ini'].settings_file)
        real_types = {t for types in identity.component_groups.values() for t in types}
        legend = identity.legend(present_types=real_types)
        assert 'Virtual atom' not in legend.values()
        assert 'Virtual atom' in identity.elements.values()

    def test_missing_file_yields_an_empty_identity(self, tmp_path):
        identity = parse_identity(tmp_path / 'absent')
        assert identity.groups == {} and identity.elements == {}


class TestTypeRangeExpansion:

    @pytest.mark.parametrize('spec,expected', [
        ('7', [7]),
        ('1*6', [1, 2, 3, 4, 5, 6]),
        ('9*10', [9, 10]),
        ('5*5', [5]),
        ('6*3', []),
        ('junk', []),
    ])
    def test_expand(self, spec, expected):
        assert expand_type_range(spec) == expected


class TestDataFileBox:

    def test_reads_all_three_axes(self, tmp_path):
        data = tmp_path / 'x.lmp'
        data.write_text(
            "LAMMPS data\n\n8 atoms\n2 atom types\n\n"
            "0.0 28.65 xlo xhi\n0.0 24.81 ylo yhi\n-5.0 30.0 zlo zhi\n\nAtoms\n",
            encoding='utf-8')
        assert read_data_file_box(data) == (0.0, 28.65, 0.0, 24.81, -5.0, 30.0)

    def test_returns_none_when_an_axis_is_missing(self, tmp_path):
        data = tmp_path / 'x.lmp'
        data.write_text("0.0 28.65 xlo xhi\n\nAtoms\n", encoding='utf-8')
        assert read_data_file_box(data) is None
