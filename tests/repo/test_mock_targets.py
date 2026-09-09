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
    "src.interfaces.lammps.",
    "src.interfaces.atomsk.",
    "src.builders.construction.components.make_amorphous",
    "src.builders.construction.components.run_lammps_commands",
    "src.builders.components.run_lammps_commands",   # lazy alias of the line above
    "src.aiida.",
    "src.api.",
    "src.data.database.",
    # run_simulations builds real geometry -- a genuine expensive boundary.
    # src.cli imports the name at module level (`from .run.simulate import
    # run_simulations`), so src.cli.run_simulations is the name the CLI
    # itself resolves at call time; both spellings name the exact same
    # function object, not two different things to maintain. Contrast this
    # with _global_settings_path (Phase 2a): that is a cheap pure function
    # with an env-var alternative, and widening the allowlist for it was
    # refused -- this entry is narrow (two fully-qualified names, not a
    # prefix over a whole module) and only covers a function that is
    # actually expensive to run for real.
    "src.cli.run_simulations",
    "src.run.simulate.run_simulations",
)

# file (relative to tests/) -> targets it patches today that are NOT boundaries.
# Ratchet: entries may only be removed.
KNOWN_VIOLATIONS = {
    "builders/test_afm_builder.py": {
        # Phase 2c test-weak repair (builders-afm.md (c)): the since-removed
        # test_afm_builder_layer_specific_paths never called build() at all
        # (its mocks were dead code) and test_afm_build_orchestrates_per_
        # layer_steps no longer mocks _create_directories -- it now asserts
        # the real L{n} directory tree on disk instead. build_tip/
        # build_substrate were only patched by the removed test.
        "src.builders.afm.AFMSimulation._build_components",
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        "src.builders.afm.AFMSimulation._generate_potentials",
        "src.builders.afm.AFMSimulation._get_substrate_box_dims",
        "src.builders.afm.AFMSimulation._init_provenance",
        "src.builders.afm.AFMSimulation.write_inputs",
        "src.builders.components.build_sheet",
    },
    "builders/test_builders.py": {
        # Phase 2c test-weak repair (docs/superpowers/reviews/
        # builders-construction.md (c)): every test now builds real
        # structures via real CIF/potential fixtures and mocks only the
        # genuine boundaries (run_lammps_commands, make_amorphous -- both
        # ALLOWED_TARGET_PREFIXES) plus these four native helpers, kept
        # patched with `wraps=` (the real function still executes; only
        # call-count is observed) as a regression guard that the native
        # pipeline -- not Atomsk -- is what ran.
        "src.builders.components._duplicate_lammps_data",
        "src.builders.components._orthogonalize_lammps_data",
        "src.builders.components._write_cif_as_lammps_atomic",
        "src.builders.components.apply_sheet_edge_types",
        "src.builders.components.stack_multilayer_sheet",
    },
    "builders/test_pes_scan_builder.py": {
        "src.builders.sheetonsheet.SheetOnSheetSimulation.build",
    },
    "builders/test_simulation_base.py": {
        "src.builders.simulation_base.get_material_path",
        "src.builders.simulation_base.get_potential_path",
    },
    "gui/test_gui_document.py": {
        "src.gui.controllers.document.expand_config_sweeps",
    },
    "run/test_run.py": {
        "src.run.simulate.AFMSimulation",
        "src.run.simulate.SheetOnSheetSimulation",
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
