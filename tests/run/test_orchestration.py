"""The generation façade and preview-target discovery, checked against real trees.

Discovery is by content, not filename: the assembly script is the one with a
``region ... block`` line. These tests generate real packages so that a change
to a template or to a builder's output layout fails here rather than silently
producing an empty preview.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from frictionsim2d.run.artifacts import (
    discover_preview_target,
    find_assembly_script,
    generate_package,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

#: (config, model, expected assembly script, expected layers)
CASES = [
    ('indent_config.ini', 'indent', 'system.in', [1, 2]),
    ('afm_flake_config.ini', 'afm', 'system.in', [1]),
    ('pes_tip_config.ini', 'pes_tip', 'slide.in', [1]),
    ('pes_sheet_config.ini', 'pes_sheet', 'slide.in', []),
]


@pytest.fixture(scope='module', name='generated')
def _generated(tmp_path_factory):
    """Generate every case once; these builds are the slow part."""
    import os
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        out = {}
        for name, model, _, _ in CASES:
            root = tmp_path_factory.mktemp(model)
            out[name] = generate_package(
                EXAMPLES / name, model, output_root=root, generate_hpc=False,
            )
        return out
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize('name,model,script,layers', CASES, ids=[c[0] for c in CASES])
def test_generation_succeeds(name, model, script, layers, generated):
    result = generated[name]
    assert result.ok, result.render()
    assert result.created
    assert result.failures == []


@pytest.mark.parametrize('name,model,script,layers', CASES, ids=[c[0] for c in CASES])
def test_discovers_the_assembly_script(name, model, script, layers, generated):
    result = generated[name]
    target = discover_preview_target(result.simulation_root, result.created[0])
    assert target is not None
    assert target.init_script.name == script
    assert target.settings_file is not None, "system.in.settings is the identity source"


@pytest.mark.parametrize('name,model,script,layers', CASES, ids=[c[0] for c in CASES])
def test_read_data_paths_resolve_from_the_simulation_root(name, model, script, layers,
                                                          generated):
    """Not from the script's own directory -- a mistake that silently yields an
    empty preview."""
    result = generated[name]
    target = discover_preview_target(result.simulation_root, result.created[0])
    text = target.init_script.read_text(encoding='utf-8')

    referenced = [
        line.split()[1] for line in text.splitlines()
        if line.strip().startswith('read_data')
    ]
    assert referenced
    for ref in referenced:
        if '$' in ref:  # an unexpanded LAMMPS variable, not a component file
            continue
        assert (target.simulation_root / ref).exists(), ref
        assert not (target.init_script.parent / ref).exists()


def _opens_a_box(target) -> bool:
    text = target.init_script.read_text(encoding='utf-8')
    return re.search(r'^\s*region\s+\S+\s+block\s', text, re.MULTILINE) is not None


def test_pes_sheet_is_preassembled(generated):
    """Its script has no region/create_box, so the cell must come from the data
    file header instead."""
    result = generated['pes_sheet_config.ini']
    target = discover_preview_target(result.simulation_root, result.created[0])
    assert not _opens_a_box(target)


@pytest.mark.parametrize('name', ['indent_config.ini', 'afm_flake_config.ini',
                                  'pes_tip_config.ini'])
def test_afm_family_is_not_preassembled(name, generated):
    result = generated[name]
    target = discover_preview_target(result.simulation_root, result.created[0])
    assert _opens_a_box(target)


class TestLayerSelection:

    def test_defaults_to_the_lowest_layer(self, generated):
        result = generated['indent_config.ini']
        target = discover_preview_target(result.simulation_root, result.created[0])
        assert target.layer == 1
        assert target.run_dir.name == 'L1'

    def test_explicit_layer_is_honoured(self, generated):
        result = generated['indent_config.ini']
        target = discover_preview_target(result.simulation_root, result.created[0], layer=2)
        assert target.layer == 2
        assert target.run_dir.name == 'L2'

    def test_missing_layer_raises_with_the_available_ones(self, generated):
        result = generated['indent_config.ini']
        with pytest.raises(ValueError, match='available: L1, L2'):
            discover_preview_target(result.simulation_root, result.created[0], layer=7)


class TestFindAssemblyScript:
    """Picking by content rather than by name."""

    def test_prefers_the_script_with_a_box(self, tmp_path):
        lammps = tmp_path / 'lammps'
        lammps.mkdir()
        (lammps / 'slide.in').write_text('read_data data/load_$(v_find)N.data\n',
                                         encoding='utf-8')
        (lammps / 'system.in').write_text(
            'region box block 0 10 0 10 0 10\ncreate_box 3 box\n', encoding='utf-8')
        assert find_assembly_script(lammps).name == 'system.in'

    def test_falls_back_to_the_plain_name_when_none_has_a_box(self, tmp_path):
        lammps = tmp_path / 'lammps'
        lammps.mkdir()
        # The refinement variant sorts FIRST, so only the explicit preference
        # for the plain name can pick slide.in.
        (lammps / 'a_refine.in').write_text('read_data build/x.lmp\n', encoding='utf-8')
        (lammps / 'slide.in').write_text('read_data build/x.lmp\n', encoding='utf-8')
        assert find_assembly_script(lammps).name == 'slide.in'

    def test_returns_none_for_an_empty_directory(self, tmp_path):
        lammps = tmp_path / 'lammps'
        lammps.mkdir()
        assert find_assembly_script(lammps) is None


class TestFailureReporting:

    def test_failures_are_reported_with_the_config_and_error(self, tmp_path, monkeypatch):
        """A build that fails must be nameable, not just absent from a count."""
        monkeypatch.chdir(REPO_ROOT)
        import frictionsim2d.builders.afm as afm_module

        def explode(self):
            raise RuntimeError("synthetic build failure")

        monkeypatch.setattr(afm_module.AFMSimulation, 'build', explode)

        result = generate_package(
            EXAMPLES / 'afm_flake_config.ini', 'afm',
            output_root=tmp_path, generate_hpc=False,
        )
        assert not result.ok
        assert result.created == []
        assert len(result.failures) == 1
        assert 'synthetic build failure' in result.failures[0].render()
        assert result.failures[0].config
