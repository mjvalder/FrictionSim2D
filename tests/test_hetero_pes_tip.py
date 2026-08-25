"""Tests for hetero (two-material) tip-on-sheet static PES scan.

Config fixture modelled on tests/test_hetero_afm.py's `_afm_hetero_raw()` (tip
+ sub + [2D-1]/[2D-2]), plus a `[pes]` section per tests/test_pes_scan_builder.py.
Covers the MRO/SPLIT_OUTER_LOOPS resolution, routing (including the
freestanding guard), and a real-LAMMPS build smoke that checks the emitted
scan script's period is the matched interface's *minimal in-plane repeat* --
not the assembled box, which is a 4x2 superlattice of it here.
"""
import shutil
from pathlib import Path

import pytest

from src.core.config import AFMSimulationConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _pes_tip_hetero_raw():
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 10.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "pes": {"grid_n": 4},
    }


# --- MRO / SPLIT_OUTER_LOOPS resolution (Step 1) -----------------------------

def test_hetero_pes_tip_mro_and_split_outer_loops():
    """Pin the MRO/SPLIT_OUTER_LOOPS resolution the class docstring documents:
    PESTipSimulation must resolve before HeteroAFMSimulation so the run-phase
    identity (self-contained static-scan write_inputs) is the PES tip's, not
    the plain hetero AFM slide's, and SPLIT_OUTER_LOOPS stays frozenset() (a
    static scan never reads outer_loop). A wrong SPLIT_OUTER_LOOPS would
    silently make the GUI promise a job count the build never produces (the
    exact bug class fixed elsewhere on this branch)."""
    from src.builders.hetero_afm import HeteroAFMSimulation
    from src.builders.pes_scan import HeteroPESTipSimulation, PESTipSimulation

    mro = [c.__name__ for c in HeteroPESTipSimulation.__mro__]
    print("HeteroPESTipSimulation MRO:", mro)
    print("HeteroPESTipSimulation.SPLIT_OUTER_LOOPS:", HeteroPESTipSimulation.SPLIT_OUTER_LOOPS)
    assert mro.index("PESTipSimulation") < mro.index("HeteroAFMSimulation")
    assert HeteroPESTipSimulation.SPLIT_OUTER_LOOPS == frozenset()
    assert HeteroPESTipSimulation.SPLIT_OUTER_LOOPS == PESTipSimulation.SPLIT_OUTER_LOOPS


# --- routing ------------------------------------------------------------------

def test_routing_selects_hetero_pes_tip_for_two_materials_with_substrate():
    from src.builders.pes_scan import HeteroPESTipSimulation
    from src.run.simulate import _select_afm_builder_cls
    cfg = AFMSimulationConfig(**_pes_tip_hetero_raw(), settings=load_settings())
    assert _select_afm_builder_cls("pes_tip", cfg) is HeteroPESTipSimulation


def test_routing_keeps_homogeneous_pes_tip_for_single_material():
    from src.builders.pes_scan import PESTipSimulation
    from src.run.simulate import _select_afm_builder_cls
    raw = _pes_tip_hetero_raw()
    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    assert _select_afm_builder_cls("pes_tip", cfg) is PESTipSimulation


def test_routing_hetero_pes_tip_freestanding_raises():
    """A freestanding (no [sub]) two-material config has no hetero freestanding
    builder -- must raise FreestandingConfigError naming the gap explicitly,
    not silently fall back to FreestandingPESTipSimulation (which cannot build
    a two-material stack) or route to the periodic hetero builder."""
    from src.run.simulate import FreestandingConfigError, _select_afm_builder_cls
    raw = _pes_tip_hetero_raw()
    del raw["sub"]
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    with pytest.raises(FreestandingConfigError, match="heterostructure freestanding is unsupported"):
        _select_afm_builder_cls("pes_tip", cfg)


def test_freestanding_hetero_pes_tip_aborts_run_simulations_not_skips(tmp_path, monkeypatch):
    """No [sub] + 2 materials is a config-level setting, not a per-sweep-member
    one -- every member of a sweep would fail identically. run_simulations must
    propagate FreestandingConfigError and abort (no build), mirroring the
    hetero-indent finite_sheet NoHeteroBuilderError hard-abort test."""
    from src.core.config import write_config
    from src.run.simulate import FreestandingConfigError, run_simulations

    monkeypatch.chdir(REPO_ROOT)
    raw = _pes_tip_hetero_raw()
    del raw["sub"]
    config_path = write_config(raw, tmp_path / "freestanding_hetero_pes_tip.ini")

    with pytest.raises(FreestandingConfigError, match="heterostructure freestanding is unsupported"):
        run_simulations(config_file=str(config_path), model="pes_tip",
                        settings_file=None, output_root=tmp_path / "out",
                        generate_hpc=False)
    assert not list((tmp_path / "out").rglob("lammps"))


# --- real-LAMMPS build smoke: minimal interface repeat, not the box ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_pes_tip_build_scan_period_is_interface_repeat(tmp_path):
    """builder.build() must emit the self-contained static scan script
    (lammps/slide.in, no separate system.in indentation phase, mirroring the
    homogeneous PESTipSimulation build layout) over the hetero-built system,
    and its scan period (Lx/Ly in the rendered script) must equal the matched
    interface's MINIMAL in-plane repeat -- NOT the assembled box, which is a
    4x2 superlattice of it for this pair. Also checks the job name carries the
    pair token, not sheet 1's material alone."""
    from src.builders import components
    from src.builders.pes_scan import HeteroPESTipSimulation

    cfg = AFMSimulationConfig(**_pes_tip_hetero_raw(), settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]   # HPC script gen needs a non-empty module list
    sim = HeteroPESTipSimulation(cfg, str(tmp_path))
    sim.build()

    pair = "h-MoS2+h-WS2"
    layer_dir = sim.output_dir_layer[1]
    lammps_dir = layer_dir / "lammps"
    assert (lammps_dir / "slide.in").exists()
    assert not (lammps_dir / "system.in").exists()   # self-contained scan, no indent phase
    assert (sim.output_dir / "build" / "hetero_rep.data").exists()

    assert sim._get_hpc_job_name() == f"pes_tip_{pair}"

    slide_text = (lammps_dir / "slide.in").read_text(encoding="utf-8")

    def _extract(var):
        for line in slide_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"variable        {var}"):
                return float(stripped.split("equal", 1)[1].strip())
        raise AssertionError(f"variable {var} not found in slide.in")

    period_x_emitted = _extract("Lx")
    period_y_emitted = _extract("Ly")

    # Independent expectation: the interface's MINIMAL in-plane repeat for this
    # EXACT fixture (h-MoS2 12x12, h-WS2 12x12, both `sw`, hetero_stacking
    # 'grouped'), hardcoded deliberately -- NOT read back from
    # sim.sheet_unit_cell or sim._hetero_coincidence_cell. Comparing the emitted
    # period against the attribute it was rendered from would be a tautology:
    # write_inputs faithfully emits whatever sheet_unit_cell holds, so a bug
    # that fed it the wrong value would still pass that comparison.
    #
    # NOT the coincidence cell (12.740025 x 11.033185, pinned by an earlier
    # revision of this test). build_monolayer tiles each material to the
    # configured footprint BEFORE the match runs, so this near-lattice-matched
    # pair matches with (near-)identity matrices on cells that are already a
    # 4x2 tiling of their own primitive -- leaving the assembled box a
    # superlattice of the true repeat. Scanning the box spent 4x2 of the grid
    # re-visiting registries it had already sampled.
    EXPECTED_LX, EXPECTED_LY = 3.18500625, 5.516592573245964
    assert period_x_emitted == pytest.approx(EXPECTED_LX, rel=1e-6)
    assert period_y_emitted == pytest.approx(EXPECTED_LY, rel=1e-6)

    # Derive the same number a second, independent way. find_coincidence_lattice
    # strains both lattices onto a shared MEAN cell, so the repeat unit of the
    # matched interface is the mean of the two constituents' own orthogonalized
    # unit cells -- computed here from build_sheet, touching no builder state.
    own_cells = []
    for sheet_cfg in cfg.sheets:
        own_dir = tmp_path / f"own_{sheet_cfg.mat}"
        own_dir.mkdir()
        unit_cell_out = {}
        components.build_sheet(
            sheet_cfg, own_dir, settings=cfg.settings,
            unit_cell_out=unit_cell_out,
        )
        own_cells.append((unit_cell_out["xhi"] - unit_cell_out["xlo"],
                          unit_cell_out["yhi"] - unit_cell_out["ylo"]))

    mean_x = sum(c[0] for c in own_cells) / len(own_cells)
    mean_y = sum(c[1] for c in own_cells) / len(own_cells)
    assert period_x_emitted == pytest.approx(mean_x, rel=1e-6)
    assert period_y_emitted == pytest.approx(mean_y, rel=1e-6)

    # The regression this guards is now the opposite of what it used to be: the
    # scan must NOT step over the whole assembled box. The box is an exact 4x2
    # multiple of the repeat, so stepping it would re-visit each registry 8
    # times and resolve only grid_n/4 distinct x-registries.
    box_x, box_y = sim._hetero_coincidence_cell
    assert box_x / period_x_emitted == pytest.approx(4.0, rel=1e-9)
    assert box_y / period_y_emitted == pytest.approx(2.0, rel=1e-9)
