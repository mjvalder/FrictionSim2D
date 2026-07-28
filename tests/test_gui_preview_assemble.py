"""Assembling the preview: atoms must land where the generated script says.

These tests check the arithmetic against the components themselves -- the atom
count is the sum of the parts, and a shifted component's bounding box has moved
by exactly its shift -- rather than against a recomputation of the placement,
which would only prove the preview agrees with itself.

Needs ovito for the LAMMPS reader, but no display.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.core.orchestration import discover_preview_target, generate_package
from src.gui.preview.assemble import assemble, to_ovito
from src.gui.preview.init_script import parse_assembly
from src.gui.preview.settings_file import parse_identity

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'

CASES = [
    ('indent_config.ini', 'indent', {'sub', 'tip', '2D'}),
    ('afm_flake_config.ini', 'afm', {'sub', 'tip', '2D', 'flake'}),
    ('pes_tip_config.ini', 'pes_tip', {'sub', 'tip', '2D'}),
    ('pes_sheet_config.ini', 'pes_sheet', {'2D'}),
]


@pytest.fixture(scope='module', name='systems')
def _systems(tmp_path_factory):
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
            assembly = parse_assembly(target.init_script, target.simulation_root)
            identity = parse_identity(target.settings_file)
            out[name] = (assembly, assemble(assembly, identity))
        return out
    finally:
        os.chdir(cwd)


def _component_atom_count(path: Path) -> int:
    for line in path.read_text(encoding='utf-8').splitlines()[:30]:
        if line.strip().endswith('atoms'):
            return int(line.split()[0])
    raise AssertionError(f"no atom count in {path}")


@pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
def test_atom_count_is_the_sum_of_the_components(name, model, groups, systems):
    assembly, system = systems[name]
    expected = sum(_component_atom_count(path)
                   for _, path in assembly.resolved_placements())
    assert system.count == expected


@pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
def test_components_are_named_and_contiguous(name, model, groups, systems):
    _, system = systems[name]
    assert {c.name for c in system.components} == groups
    cursor = 0
    for component in system.components:
        assert component.first == cursor
        cursor = component.last
    assert cursor == system.count


@pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
def test_no_warnings_for_a_clean_build(name, model, groups, systems):
    _, system = systems[name]
    assert system.warnings == []


@pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
def test_every_atom_has_a_legend_entry(name, model, groups, systems):
    _, system = systems[name]
    for type_id in np.unique(system.type_ids):
        assert int(type_id) in system.legend, type_id


@pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
def test_atoms_lie_within_the_cell_laterally(name, model, groups, systems):
    _, system = systems[name]
    xlo, xhi, ylo, yhi, _, _ = system.cell
    extent = system.extent()
    # A small tolerance: components may sit exactly on the periodic wall.
    assert extent[0][0] >= xlo - 1.0 and extent[1][0] <= xhi + 1.0
    assert extent[0][1] >= ylo - 1.0 and extent[1][1] <= yhi + 1.0


class TestShiftIsApplied:
    """The placement arithmetic is the thing most easily got wrong."""

    def test_shifted_component_moved_by_exactly_its_shift(self, systems):
        assembly, system = systems['indent_config.ini']
        placements = {p.group: (p, path) for p, path in assembly.resolved_placements()}
        placement, path = placements['tip']

        from src.gui.preview.assemble import _read_component
        raw_positions, _ = _read_component(path)

        component = next(c for c in system.components if c.name == 'tip')
        placed = system.positions[component.first:component.last]

        delta = placed.min(axis=0) - raw_positions.min(axis=0)
        assert np.allclose(delta, np.asarray(placement.shift), atol=1e-6)

    def test_unshifted_component_is_untouched(self, systems):
        assembly, system = systems['indent_config.ini']
        placements = {p.group: (p, path) for p, path in assembly.resolved_placements()}
        _, path = placements['sub']

        from src.gui.preview.assemble import _read_component
        raw_positions, _ = _read_component(path)

        component = next(c for c in system.components if c.name == 'sub')
        placed = system.positions[component.first:component.last]
        assert np.allclose(placed.min(axis=0), raw_positions.min(axis=0), atol=1e-6)

    def test_tip_sits_above_the_sheet(self, systems):
        """A physical sanity check on the assembled geometry."""
        _, system = systems['indent_config.ini']
        tip = next(c for c in system.components if c.name == 'tip')
        sheet = next(c for c in system.components if c.name == '2D')
        tip_bottom = system.positions[tip.first:tip.last][:, 2].min()
        sheet_top = system.positions[sheet.first:sheet.last][:, 2].max()
        assert tip_bottom > sheet_top


class TestTypeOffsets:

    def test_component_type_ranges_do_not_overlap(self, systems):
        _, system = systems['afm_flake_config.ini']
        seen = set()
        for component in system.components:
            assert not seen & set(component.type_ids), component.name
            seen |= set(component.type_ids)

    def test_types_match_the_identity_groups(self, systems):
        assembly, system = systems['indent_config.ini']
        target_types = {c.name: set(c.type_ids) for c in system.components}
        identity = parse_identity(
            assembly.source.parent / 'system.in.settings')
        for name, types in target_types.items():
            assert types <= set(identity.component_groups[name]), name


class TestBoundaryDetection:
    """Feeds the PBC warning: a component touching the wall meets its own image."""

    def test_sheet_and_substrate_span_the_periodic_box(self, systems):
        _, system = systems['indent_config.ini']
        touching = system.touches_boundary()
        assert touching['sub'] and touching['2D']

    def test_tip_does_not_touch_the_boundary(self, systems):
        _, system = systems['indent_config.ini']
        assert not system.touches_boundary()['tip']


class TestUnresolvableComponents:

    def test_runtime_state_is_skipped_with_a_warning(self, systems):
        assembly, _ = systems['indent_config.ini']
        slide = next(assembly.source.parent.glob('slide_*.in'))
        runtime = parse_assembly(slide, assembly.simulation_root)
        system = assemble(runtime)
        assert system.count == 0
        assert any('run-time state' in w for w in system.warnings)


class TestOvitoConversion:

    @pytest.mark.parametrize('name,model,groups', CASES, ids=[c[0] for c in CASES])
    def test_pipeline_round_trips_the_data(self, name, model, groups, systems):
        _, system = systems[name]
        data = to_ovito(system).compute()
        assert data.particles.count == system.count
        assert 'Group' in data.particles
        assert np.allclose(np.asarray(data.particles['Position']), system.positions)

    def test_cell_is_periodic_in_x_and_y_only(self, systems):
        _, system = systems['indent_config.ini']
        data = to_ovito(system).compute()
        assert tuple(data.cell.pbc) == (True, True, False)

    def test_group_property_partitions_the_atoms(self, systems):
        _, system = systems['indent_config.ini']
        data = to_ovito(system).compute()
        groups = np.asarray(data.particles['Group'])
        for component in system.components:
            block = groups[component.first:component.last]
            assert np.all(block == component.group_id)
