"""Mocks are for expensive boundaries only (tests/README.md).

Scans every test file for string-form patch targets under ``src.`` and fails
on any target that is neither an allowed boundary nor in the ratchet list of
pre-existing violations. Object-form ``monkeypatch.setattr(obj, "name", …)``
cannot be checked statically and is not covered.

Phase 2 of the 2026-09 review shrinks KNOWN_VIOLATIONS package by package;
it must never grow.

Note: ``conftest.py`` itself patches ``src.potentials.manager.cifread`` and
``count_atomtypes`` (via fixtures), but ``conftest.py`` is not a ``test_*.py``
file, so it is outside this scan.
"""
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

ALLOWED_TARGET_PREFIXES = (
    "src.interfaces.lammps",
    "src.interfaces.atomsk",
    "src.builders.construction.components.make_amorphous",
    "src.builders.construction.components.run_lammps_commands",
    "src.builders.components.run_lammps_commands",   # lazy alias of the line above
    "src.aiida.",
    "src.api.",
    "src.data.database",
)

# file (relative to tests/) -> targets it patches today that are NOT boundaries.
# Ratchet: entries may only be removed.
KNOWN_VIOLATIONS = {
    "builders/test_afm_builder.py": {
        "src.builders.afm.AFMSimulation._build_components",
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        "src.builders.afm.AFMSimulation._create_directories",
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        "src.builders.afm.AFMSimulation._generate_potentials",
        "src.builders.afm.AFMSimulation._get_substrate_box_dims",
        "src.builders.afm.AFMSimulation._init_provenance",
        "src.builders.afm.AFMSimulation.write_inputs",
        "src.builders.components.build_sheet",
        "src.builders.components.build_substrate",
        "src.builders.components.build_tip",
    },
    "builders/test_builders.py": {
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        "src.builders.afm.AFMSimulation._generate_potentials",
        "src.builders.afm.AFMSimulation._init_provenance",
        "src.builders.afm.AFMSimulation.write_inputs",
        "src.builders.afm.base.get_model_dimensions",
        "src.builders.components._create_base_slab",
        "src.builders.components._duplicate_lammps_data",
        "src.builders.components._orthogonalize_lammps_data",
        "src.builders.components._write_cif_as_lammps_atomic",
        "src.builders.components.apply_langevin_regions",
        "src.builders.components.apply_sheet_edge_types",
        "src.builders.components.build_monolayer",
        "src.builders.components.build_sheet",
        "src.builders.components.build_substrate",
        "src.builders.components.build_tip",
        "src.builders.components.charge2atom",
        "src.builders.components.check_potential_cif_compatibility",
        "src.builders.components.cifread",
        "src.builders.components.get_material_path",
        "src.builders.components.get_model_dimensions",
        "src.builders.components.get_num_atom_types",
        "src.builders.components.get_potential_element_order",
        "src.builders.components.shift_atoms_to_z_zero",
        "src.builders.components.shutil.copy",
        "src.builders.components.stack_multilayer_sheet",
    },
    "builders/test_pes_scan_builder.py": {
        "src.builders.sheetonsheet.SheetOnSheetSimulation.build",
    },
    "builders/test_sheetonsheet_builder.py": {
        "src.builders.components.build_sheet",
        "src.builders.sheetonsheet.SheetOnSheetSimulation._create_directories",
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_hpc_scripts",
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_potentials",
        "src.builders.sheetonsheet.SheetOnSheetSimulation._init_provenance",
        "src.builders.sheetonsheet.SheetOnSheetSimulation.write_inputs",
    },
    "builders/test_simulation_base.py": {
        "src.builders.simulation_base.get_material_path",
        "src.builders.simulation_base.get_potential_path",
    },
    "core/test_config.py": {
        "src.core.config._global_settings_path",
        "src.core.config.get_material_path",
        "src.core.config.get_potential_path",
    },
    "gui/test_gui_document.py": {
        "src.gui.controllers.document.expand_config_sweeps",
    },
    "potentials/test_potential_manager.py": {
        "src.core.config.get_material_path",
        "src.core.config.get_potential_path",
        "src.potentials.manager.cifread",
        "src.potentials.manager.count_atomtypes",
    },
    "run/test_run.py": {
        "src.run.simulate.AFMSimulation",
        "src.run.simulate.AFMSimulationConfig",
        "src.run.simulate.SheetOnSheetSimulation",
        "src.run.simulate.SheetOnSheetSimulationConfig",
        "src.run.simulate.expand_config_sweeps",
        "src.run.simulate.generate_hpc_scripts_for_root",
        "src.run.simulate.load_settings",
        "src.run.simulate.parse_config",
    },
}

_PATTERN = re.compile(r"""(?:patch|patch\.object|monkeypatch\.setattr)\(\s*['"](src\.[A-Za-z0-9_.]+)['"]""")


def _targets():
    found = {}
    for path in sorted(TESTS.rglob("test_*.py")):
        rel = str(path.relative_to(TESTS))
        hits = set(_PATTERN.findall(path.read_text(encoding="utf-8")))
        if hits:
            found[rel] = hits
    return found


def test_only_expensive_boundaries_are_mocked():
    offenders = {}
    for rel, hits in _targets().items():
        for target in hits:
            if target.startswith(ALLOWED_TARGET_PREFIXES):
                continue
            if target in KNOWN_VIOLATIONS.get(rel, ()):
                continue
            offenders.setdefault(rel, []).append(target)
    assert not offenders, f"new non-boundary mock targets: {offenders}"


def test_known_violations_only_shrink():
    live = _targets()
    stale = {rel: sorted(set(ts) - live.get(rel, set())) for rel, ts in KNOWN_VIOLATIONS.items()}
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, f"remove from KNOWN_VIOLATIONS, no longer patched: {stale}"
