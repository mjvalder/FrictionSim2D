"""Tests for D2 periodic hetero AFM tip-slide assembly.

Covers config folding ([2D-1]/[2D-2] -> sheets), the in-plane replication
helpers, the HeteroAFMSimulation structure build + orthogonality guard, the
sheet_1/sheet_2 potential split + 2D union groups, render-context wiring,
routing, and a short real-LAMMPS end-to-end smoke.
"""
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.config import AFMSimulationConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _afm_hetero_raw():
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
    }


def _afm_homogeneous_raw():
    raw = _afm_hetero_raw()
    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    return raw


# --- Task 1: config folding ---

def test_afm_config_folds_two_sheets():
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    assert len(cfg.sheets) == 2
    assert cfg.sheets[0].mat == "h-MoS2" and cfg.sheets[1].mat == "h-WS2"
    # single `sheet` still set (homogeneous path unchanged), = first sheet
    assert cfg.sheet.mat == "h-MoS2"


def test_afm_config_single_sheet_is_homogeneous():
    cfg = AFMSimulationConfig(**_afm_homogeneous_raw(), settings=load_settings())
    assert len(cfg.sheets) <= 1
    assert cfg.sheet.mat == "h-MoS2"


# --- Task 2: in-plane replication helpers ---

def test_replication_factor_meets_two_diameter_rule():
    from src.builders.afm.hetero import compute_replication_factors
    # footprint must be >= 4*r in each direction, min 1
    assert compute_replication_factors(lx=19.0, ly=22.0, tip_r=10.0) == (
        max(1, math.ceil(40.0 / 19.0)), max(1, math.ceil(40.0 / 22.0)))   # (3, 2)
    # already large enough -> no replication below the raw cell
    assert compute_replication_factors(lx=100.0, ly=100.0, tip_r=10.0) == (1, 1)
    # never below 1
    assert compute_replication_factors(lx=5.0, ly=5.0, tip_r=0.1) == (1, 1)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_replicated_data_has_no_pair_coeffs(tmp_path):
    from src.builders.construction.hetero_stack import build_hetero_structure
    from src.builders.afm.hetero import replicate_hetero_in_plane
    from src.core.config import SheetOnSheetSimulationConfig
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    out = tmp_path / "hetero_rep.data"
    replicate_hetero_in_plane(stack.data_path, out, 2, 2, cfg.settings)
    text = out.read_text()
    assert "Pair Coeffs" not in text          # written nocoeff
    assert "Atoms" in text


# --- Task 3: structure build + orthogonality guard ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_build_replicates_and_orthogonal(tmp_path):
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim._build_hetero_sheet(build_dir=Path(tmp_path) / "build")   # structure-only entry
    assert sim._hetero_rep_path.exists()
    lx, ly = sim._hetero_rep_box[:2]
    # footprint >= 4*tip.r each way (tip r=10 -> >= 40 A)
    assert lx >= 40.0 - 1e-6 and ly >= 40.0 - 1e-6


def test_hetero_afm_rejects_triclinic_cell(tmp_path, monkeypatch):
    from src.builders.afm import hetero as ha
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = ha.HeteroAFMSimulation(cfg, str(tmp_path))

    import numpy as np
    from src.interfaces.ase import LammpsData

    cell = np.eye(3) * 20.0
    cell[1, 0] = 5.0          # nonzero xy tilt -> triclinic
    fake = LammpsData(cell=cell, positions=np.zeros((1, 3)), numbers=np.array([14]))

    class _FakeStack:  # only .data_path is read before the guard
        data_path = tmp_path / "hetero.data"

    (tmp_path / "hetero.data").write_text("")
    monkeypatch.setattr(ha, "build_hetero_structure", lambda *a, **k: _FakeStack())
    monkeypatch.setattr(ha, "read_lammps_data", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="triclinic"):
        sim._build_hetero_sheet(build_dir=tmp_path / "build")


# --- Task 4: potential split (sheet_1/sheet_2 + 2D union groups) ---

def _prep_layer_dirs(sim, n):
    """Mirror the per-layer dir setup AFMSimulation.build() does before
    _generate_potentials, so the potential method can be exercised in isolation."""
    sim.output_dir_layer[n] = sim.output_dir / f"L{n}"
    sim.relative_run_dir_layer[n] = sim.relative_run_dir / f"L{n}"
    sim._create_directories(sim.output_dir_layer[n])


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_potentials_layout_and_2D_group(tmp_path):
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    build_dir = Path(tmp_path) / "build"
    sim._init_provenance()
    sim._build_hetero_sheet(build_dir=build_dir)
    sim.tip_path, _tip_r, sim.sub_path = sim._build_components(build_dir)
    _prep_layer_dirs(sim, 1)
    pm = sim._generate_potentials(1)
    # contiguous, disjoint blocks: sub+tip then sheet_1+sheet_2 spans all types
    n_sub = len(pm.types.ids_by_component('sub'))
    n_tip = len(pm.types.ids_by_component('tip'))
    n_s1 = len(pm.types.ids_by_component('sheet_1'))
    n_s2 = len(pm.types.ids_by_component('sheet_2'))
    assert n_s1 > 0 and n_s2 > 0
    # contiguous blocks: sub+tip, then sheet_1+sheet_2, then any drive virtual type
    n_virtual = 1 if pm.virtual_atom_type is not None else 0
    assert (n_sub + n_tip) + (n_s1 + n_s2) + n_virtual == len(pm.types)
    # the sheet block sits immediately after sub+tip
    assert min(pm.types.ids_by_component('sheet_1')) == n_sub + n_tip + 1
    # 2D_all union emitted INTO the written settings file (both scripts include it)
    settings_text = (sim.output_dir_layer[1] / "lammps" / "system.in.settings").read_text()
    assert "group 2D_all union sheet_1_all sheet_2_all" in settings_text
    # sheet_types context var = sheet_1 ∪ sheet_2 union (slide.lmp `group 2D type {{sheet_types}}`)
    assert sim.groups[1]["sheet_types"].split() == [str(i) for i in
        pm.types.ids_by_component('sheet_1') + pm.types.ids_by_component('sheet_2')]


# --- Task 5: build_render_context / build() wiring (templates unchanged) ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_system_init_references_replicated_stack(tmp_path):
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]           # HPC script gen needs a non-empty module list
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    init = sim.output_dir_layer[1] / "lammps" / "system.in"
    text = init.read_text()
    assert "hetero_rep.data" in text                 # sheet_file points at the replicated stack
    assert "reduce max z" in text                     # inherited tip-approach body intact
    # the 2D_all union lives in the included settings file (both scripts include it)
    settings_text = (sim.output_dir_layer[1] / "lammps" / "system.in.settings").read_text()
    assert "group 2D_all union sheet_1_all sheet_2_all" in settings_text


# --- Fix round 1: hetero AFM job names must not collide on material 2 ---

def test_hetero_afm_job_name_differs_by_full_pair():
    """Two periodic hetero AFM configs sharing material 1 but differing in
    material 2 must NOT produce the same HPC job name -- confirmed by
    construction (build both configs, compare _get_hpc_job_name()) per the
    review that found afm_h-MoS2 was emitted for both."""
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw_a = _afm_hetero_raw()
    raw_b = _afm_hetero_raw()
    raw_b["2D-2"]["mat"] = "h-WSe2"
    raw_b["2D-2"]["cif_path"] = f"{MAT}/h-WS2.cif"        # geometry irrelevant; job name only
    cfg_a = AFMSimulationConfig(**raw_a, settings=load_settings())
    cfg_b = AFMSimulationConfig(**raw_b, settings=load_settings())
    sim_a = HeteroAFMSimulation(cfg_a, "out_a")
    sim_b = HeteroAFMSimulation(cfg_b, "out_b")
    name_a, name_b = sim_a._get_hpc_job_name(), sim_b._get_hpc_job_name()
    assert name_a != name_b
    assert name_a == "afm_h-MoS2+h-WS2"
    assert name_b == "afm_h-MoS2+h-WSe2"


def test_finite_hetero_afm_job_name_also_uses_full_pair():
    """FiniteHeteroAFMSimulation doesn't override _get_hpc_job_name -- it must
    inherit the fixed HeteroAFMSimulation._material_id, not AFMSimulation's
    single-sheet default."""
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation
    cfg = AFMSimulationConfig(**_finite_hetero_afm_raw(), settings=load_settings())
    sim = FiniteHeteroAFMSimulation(cfg, "out")
    assert sim._get_hpc_job_name() == "afm_h-MoS2+h-WS2"


# --- Task 6: routing ---

def test_routing_selects_hetero_afm_for_two_materials():
    from src.run.simulate import _select_afm_builder_cls
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    assert _select_afm_builder_cls("afm", cfg) is HeteroAFMSimulation


def test_routing_keeps_homogeneous_afm_for_single_material():
    from src.run.simulate import _select_afm_builder_cls
    from src.builders.afm import AFMSimulation
    cfg = AFMSimulationConfig(**_afm_homogeneous_raw(), settings=load_settings())
    assert _select_afm_builder_cls("afm", cfg) is AFMSimulation


# --- Task 7: end-to-end short-LAMMPS smoke (approach + indent + short slide) ---

def _run_lammps_afm(run_dir, script_path, timeout=1800):
    """Run ``lmp_serial`` on an AFM script; return ``(proc, combined_output)``.

    ``cwd`` is ``run_dir``. Every path baked into ``system.in``/``slide.in`` is
    run-dir absolute (the builder stores ``relative_run_dir`` = the output dir),
    so ``build/hetero_rep.data``, ``L1/lammps/system.in.settings`` (with its
    staged ``provenance/potentials`` prefix), ``L1/data/load_*.data`` and
    ``L1/results/`` all resolve. ``combined_output`` is stdout + stderr + the
    LAMMPS ``log.lammps`` so callers can grep for errors / "Lost atoms".
    """
    run_dir = Path(run_dir)
    proc = subprocess.run(
        ["lmp_serial", "-in", str(script_path)],
        cwd=str(run_dir), capture_output=True, text=True, timeout=timeout,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    log = run_dir / "log.lammps"
    if log.exists():
        combined += "\n" + log.read_text()
    return proc, combined


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_runs_approach_indent_slide(tmp_path):
    """Capstone D2 smoke: assemble a REAL MoS2/WS2 replicated hetero + amorphous-Si
    substrate + Si tip and actually RUN it in ``lmp_serial`` -- ``system.in``
    (tip approach + a tiny indent load ramp -> ``load_10N.data``) then a short
    ``slide.in`` -- gating on physical output. Proves the whole D2 pipeline works
    end to end through the UNCHANGED AFM templates: run-dir path resolution, the
    ``sheet_1``/``sheet_2`` hybrid pair_coeff, the emitted ``2D_all``/``2D`` groups,
    the replicated box, langevin region groups, and the approach/indent/slide loop.

    Smallest viable config (empirically tuned -- see docs/superpowers/summaries/
    D2-hetero-afm.md). Each knob is a hard constraint, not an arbitrary shrink:
      * ``tip.r = 5`` -> 2x2 in-plane replication, the floor: a 2x2 MoS2/WS2 cell
        is ~22-25 A, just above ``pair_style ... lj/cut 11.0`` + skin; a smaller
        box aborts ``comm_style tiled``. ~705 atoms total.
      * ``tip_reduction_factor = 0.7`` so the small (r=5) tip is tall enough that
        the langevin thermo band ``[h-6, h-3]`` (h = r/factor) holds atoms; else
        ``fix langevin ... zero yes`` aborts "Cannot zero Langevin force of 0 atoms".
      * ``temp = 1`` K so the AFM indent template's displacement-convergence loop
        (loops until the tip z-COM excursion < 0.1 A over 5000 steps) actually
        settles; at 300 K a light rigid tip's thermal COM jitter never falls below
        0.1 A and the loop runs unbounded.
    Runtime ~110 s: ``system.in``'s hardcoded 10k-step equilibration + 100x100-step
    load ramp + mechanical settling dominate. Those step counts live in the AFM
    templates (reused unchanged); ``slide_run_steps`` only shortens the already
    trivial (~0.3 s) slide phase.
    """
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _afm_hetero_raw()
    raw["general"]["force"] = 10.0          # single deployable load -> one ramp
    raw["general"]["temp"] = 1.0            # so the indent convergence loop settles
    raw["tip"]["r"] = 5.0                   # 2x2 replication (LJ-cutoff floor)
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]           # HPC gen needs a non-empty module list
    cfg.settings.geometry.tip_reduction_factor = 0.7  # populate the langevin thermo band
    cfg.settings.simulation.slide_run_steps = 200     # short slide

    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()

    run_dir = Path(tmp_path)

    # 1) Approach + indent: system.in finishes cleanly, keeps every atom, and
    #    writes the per-load restart the slide phase reads.
    init = run_dir / "L1" / "lammps" / "system.in"
    assert init.exists()
    proc, out = _run_lammps_afm(run_dir, init)
    assert proc.returncode == 0, f"system.in exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost during approach/indent:\n{out[-3000:]}"
    # run-0 / pair_coeff sanity: the hetero hybrid SW+LJ pair_coeff for
    # sub/tip/sheet_1/sheet_2 was fully set (a mis-numbered sheet split would
    # leave a pair unset and abort here), and the named computes evaluated.
    assert "pair coeffs are not set" not in out.lower(), \
        f"incomplete hetero pair_coeff (sheet split mis-numbered?):\n{out[-3000:]}"
    loads = list((run_dir / "L1" / "data").glob("load_*N.data"))
    assert loads and all(p.stat().st_size > 0 for p in loads), \
        f"no non-empty load_*N.data written by the indent phase: {loads}"

    # 1b) Substrate sanity (risk: collapsed / over-thick amorphous box): assert the
    #     built amorphous Si slab's PERIODIC BOX z-height is ~thickness (config 12 A)
    #     + the build's -5..thickness padding (~17 A). NB the data file stores atoms
    #     UNWRAPPED (raw amorphous-block coords span ~90-120 A in both this and the
    #     homogeneous AFM path); LAMMPS wraps them into the box on read_data, so the
    #     BOX height -- not the raw atom span -- is the meaningful thickness check.
    from src.builders.construction.components import get_model_dimensions as _gmd
    sub_dims = _gmd(run_dir / "build" / "sub.lmp")
    box_z = float(sub_dims["zhi"]) - float(sub_dims["zlo"])
    assert 10.0 <= box_z <= 22.0, \
        f"substrate box z-height {box_z:.1f} A off ~17 A (thickness 12 + pad)"

    # 2) Slide: the load restart drives a short friction slide that writes results.
    slide = run_dir / "L1" / "lammps" / "slide.in"
    assert slide.exists()
    proc2, out2 = _run_lammps_afm(run_dir, slide)
    assert proc2.returncode == 0, f"slide.in exited {proc2.returncode}:\n{out2[-3000:]}"
    assert "Lost atoms" not in out2, f"atoms lost during slide:\n{out2[-3000:]}"
    friction = list((run_dir / "L1" / "results").glob("friction_*"))
    assert friction and all(p.stat().st_size > 0 for p in friction), \
        f"no non-empty friction output written by the slide: {friction}"


# --- D2b: crystalline lattice-matched substrate branch ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_uses_crystalline_matched_substrate(tmp_path):
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _afm_hetero_raw()
    raw["sub"]["amorph"] = "c"
    raw["sub"]["match"] = True
    # Si vs the MoS2/WS2 replicated box (~51x44 A) needs a wider tol: the best Si
    # supercell (5x4 of the 10.887 A surface cell) strains ~6.4% onto the box, above
    # the 0.02 production default. Widen in the TEST config only (production default
    # in SubstrateConfig stays 0.02); documents the D2b E2E commensurability risk.
    raw["sub"]["match_strain_tol"] = 0.05
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    from ase.io import read
    sub = read(str(next(Path(tmp_path).rglob("sub*.lmp"))), format="lammps-data")
    lx, ly, _ = sim._hetero_rep_box
    cell = sub.get_cell()
    assert abs(cell[0, 0] - lx) < 1e-3 and abs(cell[1, 1] - ly) < 1e-3   # substrate box == hetero box (no seam)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_crystalline_runs_no_seam(tmp_path):
    """D2b capstone: a crystalline lattice-matched Si substrate under a MoS2/WS2
    hetero AFM actually RUNS in ``lmp_serial`` with NO periodic-seam blowup -- the
    substrate tiles the fixed hetero box exactly (no Lost atoms), and the indent
    load ramp writes ``load_*N.data``.

    Config mirrors the D2 capstone's runtime-bounding knobs (temp=1, force=10,
    tip_reduction_factor=0.7) but with ``[sub] amorph=c, match=true``.

    ``tip.r = 7`` (not the default 10) is deliberate: it sets the hetero box to a
    3x3 replication (~38x33 A), onto which Si's *primitive* 5.44 A surface cell
    commensurates as a 7x6 supercell at only ~1.3% residual strain -- WITHIN the
    0.02 production ``match_strain_tol`` default (no widening needed). The default
    r=10 box (~51x44 A) needs a 9x8 Si supercell at ~4% strain; the coarse 2x2
    cell would need ~6.4%, whose post-release ringing stalls the AFM template's
    hardcoded tip-COM convergence loop for tens of thousands of extra steps (a
    30-min+ timeout was observed). Keeping the match strain low is what makes this
    end-to-end smoke both stable and reasonably fast.
    """
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _afm_hetero_raw()
    raw["sub"]["amorph"] = "c"
    raw["sub"]["match"] = True
    raw["tip"]["r"] = 7.0                # 3x3 box; Si primitive cell matches at ~1.3% strain
    raw["general"]["force"] = 10.0
    raw["general"]["temp"] = 1.0
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7
    cfg.settings.simulation.slide_run_steps = 200

    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()

    run_dir = Path(tmp_path)
    init = run_dir / "L1" / "lammps" / "system.in"
    assert init.exists()
    proc, out = _run_lammps_afm(run_dir, init)
    assert proc.returncode == 0, f"system.in exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost (crystalline periodic seam?):\n{out[-3000:]}"
    loads = list((run_dir / "L1" / "data").glob("load_*N.data"))
    assert loads and all(p.stat().st_size > 0 for p in loads), \
        f"no non-empty load_*N.data written by the crystalline-substrate indent: {loads}"


# ============================================================================
# D5 — Finite hetero stack + AFM tip (edge effects)
# ============================================================================

def _finite_hetero_afm_raw():
    """Finite (edge-typed) MoS2/WS2 patch + amorphous-Si substrate + Si tip.

    The substrate is sized to ``[sub] x/y`` (80x80) -- larger than the replicated
    patch, so the patch is centred with margin. ``general.finite_sheet`` routes
    ``build_hetero_structure`` to its edge-typed finite path.
    """
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped",
                    "finite_sheet": True},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 10.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0, "x": 80.0, "y": 80.0},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 20.0, "y": 20.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 20.0, "y": 20.0, "layers": [1]},
    }


def _fake_pm_with_sheet_block(block):
    """Stub PM exposing an ordered ``(pot_label, element, mass, region)`` sheet block.

    Registers each entry as ``sheet_1`` / ``sheet_2`` (grouped by material label,
    in block order) on a real :class:`TypeRegistry`, so the layout guard's
    ``pm.types`` iteration + ``ids_by_component('sheet_1'/'sheet_2')`` queries
    resolve exactly as on a real build -- without a full LAMMPS assembly.
    """
    from types import SimpleNamespace
    from src.potentials.manager import TypeRegistry
    reg = TypeRegistry()
    labels = []
    for pot_label, element, _mass, region in block:
        if pot_label not in labels:
            labels.append(pot_label)
        comp = f"sheet_{labels.index(pot_label) + 1}"
        reg.register(comp, element, pot_label, region=region)
    return SimpleNamespace(types=reg)


def _write_data_with_masses(path, masses):
    """Write a minimal LAMMPS data file whose ``Masses`` section = ``masses`` order."""
    lines = [
        "Permuted layout test data", "",
        f"{len(masses)} atoms", f"{len(masses)} atom types", "",
        "0.0 10.0 xlo xhi", "0.0 10.0 ylo yhi", "0.0 10.0 zlo zhi", "",
        "Masses", "",
    ]
    for i, (_pot, _el, mass, _region) in enumerate(masses, start=1):
        lines.append(f"{i} {mass}")
    lines.append("")
    Path(path).write_text("\n".join(lines))
    return Path(path)


# --- D5 Task 1: finite hetero sheet build + centering + layout guard ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_afm_builds_edge_typed_centered_patch(tmp_path):
    from ase.io import read
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation
    cfg = AFMSimulationConfig(**_finite_hetero_afm_raw(), settings=load_settings())
    cfg.settings.geometry.finite_sheet_afm_edge_mode = "spring"
    cfg.settings.hpc.modules = ["lammps"]
    sim = FiniteHeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    settings_file = next(Path(tmp_path).rglob("system.in.settings"))
    text = settings_file.read_text()
    assert "2D_edge" in text                          # B3 edge group present
    # substrate larger than patch (patch centred with margin), sized to [sub] x=80
    sub = read(str(next(Path(tmp_path).rglob("sub*.lmp"))), format="lammps-data").get_cell()
    assert sub[0, 0] >= 40.0


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_layout_guard_covers_the_region_split(tmp_path):
    # future-works §2: with regions=[None,'edge'] the type count doubles but a
    # COUNT-only assert cannot catch an ordering divergence. Prove the
    # single-sourced numbering + the element-wise guard cover the region-split path.
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation, assert_sheet_block_matches_data
    cfg = AFMSimulationConfig(**_finite_hetero_afm_raw(), settings=load_settings())
    cfg.settings.geometry.finite_sheet_afm_edge_mode = "spring"
    cfg.settings.hpc.modules = ["lammps"]
    sim = FiniteHeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()                                        # build passes the guard -> no divergence
    offset = (len(sim._pm.types.ids_by_component('sub'))
              + len(sim._pm.types.ids_by_component('tip')))
    # POSITIVE: unified PM's sheet block == hetero_rep.data's block, element-wise.
    assert_sheet_block_matches_data(sim._pm, sim._hetero_rep_path, offset=offset)


def test_layout_guard_trips_on_equal_count_region_permutation(tmp_path):
    # NEGATIVE: a data file whose sheet block is a PERMUTATION of the PM's block at
    # equal count (the silent-corruption mode). The guard MUST raise; a count check
    # would not.
    from src.builders.afm.hetero import assert_sheet_block_matches_data
    pm = _fake_pm_with_sheet_block([("MoS2", "Mo", 95.9, None), ("MoS2", "S", 32.1, "edge"),
                                    ("WS2", "W", 183.8, None), ("WS2", "S", 32.1, "edge")])
    data = _write_data_with_masses(
        tmp_path / "perm.data",
        masses=[("MoS2", "S", 32.1, "edge"), ("MoS2", "Mo", 95.9, None),   # SWAPPED
                ("WS2", "W", 183.8, None), ("WS2", "S", 32.1, "edge")])
    with pytest.raises((AssertionError, ValueError), match=r"[Ll]ayout|order|mismatch"):
        assert_sheet_block_matches_data(pm, data, offset=0)


# --- D5 Task 2: finite-edge context flags + 2D_edge group wiring ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_afm_renders_edge_fix(tmp_path):
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation
    cfg = AFMSimulationConfig(**_finite_hetero_afm_raw(), settings=load_settings())
    cfg.settings.geometry.finite_sheet_afm_edge_mode = "spring"
    cfg.settings.hpc.modules = ["lammps"]
    sim = FiniteHeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    init = sim.output_dir_layer[1] / "lammps" / "system.in"
    text = init.read_text()
    assert "group           sheet_edge intersect 2D 2D_edge" in text  # finite-edge block fired
    assert "spring/self" in text                                      # edge tether present
    assert "reduce max z" in text                                     # inherited tip approach intact
    # The 2D_edge union the tether intersects with is defined in the RUN settings
    # (the AFM path assembles its own settings; D2 only emits 2D_all there).
    settings_text = (sim.output_dir_layer[1] / "lammps" / "system.in.settings").read_text()
    assert "group 2D_edge union sheet_1_edge sheet_2_edge" in settings_text
    assert "group sheet_1_edge type" in settings_text
    assert "group sheet_2_edge type" in settings_text
    assert "group 2D_all union sheet_1_all sheet_2_all" in settings_text  # D2 union intact


# --- D5 Task 3: routing (finite hetero before periodic hetero) ---

def test_routing_finite_hetero_selects_finite_builder():
    from src.run.simulate import _select_afm_builder_cls
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation
    cfg = AFMSimulationConfig(**_finite_hetero_afm_raw(), settings=load_settings())
    assert _select_afm_builder_cls("afm", cfg) is FiniteHeteroAFMSimulation


def test_routing_periodic_hetero_still_selects_d2():
    from src.run.simulate import _select_afm_builder_cls
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _finite_hetero_afm_raw()
    raw["general"]["finite_sheet"] = False
    del raw["sub"]["x"]
    del raw["sub"]["y"]
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    assert _select_afm_builder_cls("afm", cfg) is HeteroAFMSimulation


# --- D5 Task 4: end-to-end short-LAMMPS finite-hetero-AFM smoke ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_afm_runs_with_edge_tether(tmp_path):
    """Capstone D5 smoke: build a REAL finite MoS2/WS2 edge-typed patch + amorphous-Si
    substrate + Si tip and RUN it in ``lmp_serial`` end to end -- ``system.in``
    (tip approach + indent load ramp -> ``load_*N.data``) then a short ``slide.in`` --
    with ``finite_sheet_afm_edge_mode='spring'`` so the UNCHANGED template edge-tether
    block (``group sheet_edge intersect 2D 2D_edge`` + ``spring/self xy``) fires on the
    hetero ``2D_edge`` group. Gates on: no shed atoms (edges tethered), the load
    restart + friction output are written, and the patch's in-plane COM drift is
    bounded (the tether holds the footprint through the slide).

    Config knobs (empirically tuned for a cheap, reliably-CONVERGING run -- a finite
    free patch introduces an undamped vertical drum/piston mode absent from the D2
    periodic sheet, and the indent template loops until the tip z-COM settles):
      * ``force = 30`` nN presses the patch FIRMLY onto the amorphous substrate,
        pinning that free z-mode against the substrate's langevin bath so the tip
        settles (a light load lets the patch float on the vdW gap and the tip never
        converges). The edge tether is xy-only, so it holds the footprint but does
        not damp the z-mode -- the load does.
      * ``tip.r = 5`` -> 2x2 in-plane replication of the ~12.7x11 A coincidence cell
        (~25x22 A patch, the >=2x-tip-diameter floor), on a 40x40 A substrate
        (patch + margin > LJ cutoff). ~1300 atoms.
      * ``tip_reduction_factor = 0.7`` so the r=5 tip is tall enough to populate the
        langevin thermo band (else ``fix langevin ... zero yes`` aborts on 0 atoms).
      * ``temp = 1`` K: low enough that the settling loop's tip z-COM range clears the
        0.1 A convergence gate once the patch is pinned.
    Runtime ~140 s (dominated by ``system.in``'s hardcoded equilibration + load ramp +
    settling); ``slide_run_steps`` only shortens the trivial slide phase.
    """
    from src.builders.afm.hetero import FiniteHeteroAFMSimulation
    raw = _finite_hetero_afm_raw()
    raw["general"]["force"] = 30.0        # firm press -> pins the free z drum-mode
    raw["general"]["temp"] = 1.0
    raw["tip"]["r"] = 5.0                  # 2x2 replication (LJ-cutoff floor)
    raw["sub"]["x"] = raw["sub"]["y"] = 40.0
    raw["2D-1"]["x"] = raw["2D-1"]["y"] = 12.0
    raw["2D-2"]["x"] = raw["2D-2"]["y"] = 12.0
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.geometry.finite_sheet_afm_edge_mode = "spring"
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7   # populate langevin thermo band
    cfg.settings.simulation.slide_run_steps = 200      # short slide
    cfg.settings.output.dump_frequency["slide"] = 100  # >=2 frames for the drift check

    sim = FiniteHeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    run_dir = Path(tmp_path)

    # 1) Approach + indent: system.in finishes cleanly, keeps every atom (edges
    #    tethered -> no shedding), and writes the per-load restart the slide reads.
    init = run_dir / "L1" / "lammps" / "system.in"
    assert init.exists()
    proc, out = _run_lammps_afm(run_dir, init)
    assert proc.returncode == 0, f"system.in exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost during approach/indent:\n{out[-3000:]}"
    loads = list((run_dir / "L1" / "data").glob("load_*N.data"))
    assert loads and all(p.stat().st_size > 0 for p in loads), \
        f"no non-empty load_*N.data written by the indent phase: {loads}"

    # 2) Slide: the load restart drives a short friction slide -> results written,
    #    still no shed atoms with the edge tether re-applied in slide.lmp.
    slide = run_dir / "L1" / "lammps" / "slide.in"
    assert slide.exists()
    proc2, out2 = _run_lammps_afm(run_dir, slide)
    assert proc2.returncode == 0, f"slide.in exited {proc2.returncode}:\n{out2[-3000:]}"
    assert "Lost atoms" not in out2, f"atoms lost during slide:\n{out2[-3000:]}"
    friction = list((run_dir / "L1" / "results").glob("friction_*"))
    assert friction and all(p.stat().st_size > 0 for p in friction), \
        f"no non-empty friction output written by the slide: {friction}"

    # 3) Edge tether holds the footprint: the patch's in-plane COM barely drifts over
    #    the slide (an untethered finite patch would translate/spin many A as the tip
    #    drags it). Compare the sheet-atom xy-COM between the first and last dump frame.
    import numpy as np
    from ase.io import read as _read
    dump = next((run_dir / "L1" / "visuals").glob("slide_*.lammpstrj"))
    frames = _read(str(dump), format="lammps-dump-text", index=":")
    assert len(frames) >= 2, f"need >=2 slide frames for the drift check, got {len(frames)}"
    sheet_ids = (sim._pm.types.ids_by_component("sheet_1")
                 + sim._pm.types.ids_by_component("sheet_2"))

    def _sheet_xy_com(atoms):
        mask = np.isin(atoms.numbers, sheet_ids)   # ase maps lammps type -> numbers
        assert mask.sum() > 0, "no sheet atoms found in the slide dump by type"
        return atoms.get_positions()[mask][:, :2].mean(axis=0)

    drift = float(np.linalg.norm(_sheet_xy_com(frames[-1]) - _sheet_xy_com(frames[0])))
    assert drift < 1.5, f"patch in-plane COM drift {drift:.3f} A too large (tether failed?)"


# --- D3b: incommensurate flake twist on the AFM (tip-on-flake) drive ---

def _afm_flake_raw():
    """A homogeneous AFM config WITH a [Flake] on a [2D] MoS2 sheet + [sub] + [tip].

    Mirrors ``examples/afm_flake_config.ini`` (a finite MoS2 hexagon between the
    Si tip and the supported MoS2 sheet on an amorphous-Si substrate), sized with
    the D2 capstone's runtime-bounding knobs (tip.r=5, temp=1, force=10) so the
    real-LAMMPS approach+indent settles quickly.
    """
    return {
        "general": {"temp": 1.0, "force": 10.0, "scan_speed": 2.0},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 5.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
               "pot_type": "sw", "x": 40.0, "y": 40.0, "layers": [1]},
        "flake": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                  "pot_type": "sw", "shape": "hexagon", "edge_length": 10.0,
                  "corner_radius": 5.0, "corner_spring_k": 9.0},
    }


def _afm_flake_cfg(twist=None):
    raw = _afm_flake_raw()
    if twist is not None:
        raw["general"]["twist_angle"] = twist
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7   # populate langevin thermo band
    cfg.settings.simulation.slide_run_steps = 200
    return cfg


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_afm_flake_twist_rotates_and_runs(tmp_path):
    from ase.io import read
    from src.builders.afm import AFMSimulation
    from tests.test_lattice_apply import _flake_lattice_orientation

    sim = AFMSimulation(_afm_flake_cfg(twist=18.0), str(tmp_path / "tw"))
    sim.build()
    sim0 = AFMSimulation(_afm_flake_cfg(twist=None), str(tmp_path / "nt"))
    sim0.build()

    # The built AFM flake component is twisted vs the untwisted build.
    fl = next((tmp_path / "tw").rglob("*hexagon.data"))
    fl0 = next((tmp_path / "nt").rglob("*hexagon.data"))
    rel = (_flake_lattice_orientation(read(str(fl), format="lammps-data"))
           - _flake_lattice_orientation(read(str(fl0), format="lammps-data"))) % 60.0
    assert min(rel, 60.0 - rel) > 5.0

    # And the twisted flake system actually runs: approach+indent, no lost atoms.
    init = tmp_path / "tw" / "L1" / "lammps" / "system.in"
    assert init.exists()
    proc, out = _run_lammps_afm(tmp_path / "tw", init)
    assert proc.returncode == 0, f"system.in exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost with twisted AFM flake:\n{out[-3000:]}"
    loads = list((tmp_path / "tw" / "L1" / "data").glob("load_*N.data"))
    assert loads and all(p.stat().st_size > 0 for p in loads), \
        f"no non-empty load_*N.data written by the twisted-flake indent: {loads}"


# --- Fix round 2: provenance must record every sheet's files, not just sheet 1 ---

def test_hetero_afm_init_provenance_records_both_sheets(tmp_path):
    """AFMSimulation._init_provenance previously iterated only self.config.sheet
    (sheet 1) -- never self.config.sheets -- so for every heterostructure AFM
    build, sheet 2's cif/potential files were silently never copied into
    provenance/. HeteroAFMSimulation inherits _init_provenance unmodified, so
    this proves the fix from the hetero AFM builder (the affected path)."""
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim._init_provenance()

    prov_dir = tmp_path / "provenance"
    assert (prov_dir / "cif" / "h-MoS2.cif").exists()
    assert (prov_dir / "cif" / "h-WS2.cif").exists()
    assert (prov_dir / "potentials" / "MoS2_wen.sw").exists()
    assert (prov_dir / "potentials" / "t-WS2.sw").exists()


def test_afm_init_provenance_single_material_unchanged(tmp_path):
    """The single-material path must still record exactly what it recorded
    before this fix: the sheet under the plain 'sheet' component name (not
    'sheet_1'), plus tip/sub."""
    from src.builders.afm import AFMSimulation
    cfg = AFMSimulationConfig(**_afm_homogeneous_raw(), settings=load_settings())
    sim = AFMSimulation(cfg, str(tmp_path))

    seen = []
    real_add = sim._add_component_files_to_provenance

    def _spy(name, config):
        seen.append(name)
        return real_add(name, config)

    sim._add_component_files_to_provenance = _spy
    sim._init_provenance()

    assert seen == ["sheet", "tip", "sub"]
    prov_dir = tmp_path / "provenance"
    assert (prov_dir / "cif" / "h-MoS2.cif").exists()
    assert (prov_dir / "potentials" / "MoS2_wen.sw").exists()


# --- Fix round 1 (cont'd): AFM's own [flake] must key output dir + job name ---
#
# AFMSimulationConfig.flake uses alias='flake' (lowercase) -- distinct from
# SheetOnSheetSimulationConfig.flake's alias='Flake' -- but shares the
# identical FlakeConfig model. flake_material_token only recognised the
# capital-F key, so it silently never saw an AFM [flake] section at all: two
# `run afm` configs sharing [2D] but naming different [flake] materials built
# into the SAME output directory and the SAME HPC job name (verified by
# construction below, mirroring examples/afm_flake_config.ini).

def test_output_dir_differs_when_only_afm_flake_material_differs():
    from src.run.naming import material_token, output_dir_for

    raw_a = _afm_flake_raw()
    raw_b = _afm_flake_raw()
    raw_b["flake"]["mat"] = "h-WS2"
    raw_b["flake"]["cif_path"] = f"{MAT}/h-WS2.cif"
    raw_b["flake"]["pot_path"] = f"{POT}/sw_lammps/t-WS2.sw"

    dir_a = output_dir_for(raw_a, "afm")
    dir_b = output_dir_for(raw_b, "afm")
    assert dir_a != dir_b
    assert material_token(raw_a, "afm") != material_token(raw_b, "afm")
    # Pin the exact collision the reviewer constructed against 72109b7.
    assert str(dir_a) == "afm/h-MoS2+flake-h-MoS2/40x_40y/sub_aSi_tip_Si_r5/K1/pot_sw_MoS2_wen"
    assert str(dir_b) == "afm/h-MoS2+flake-h-WS2/40x_40y/sub_aSi_tip_Si_r5/K1/pot_sw_MoS2_wen"


def test_hpc_job_name_differs_when_only_afm_flake_material_differs():
    from src.builders.afm import AFMSimulation

    raw_a = _afm_flake_raw()
    raw_b = _afm_flake_raw()
    raw_b["flake"]["mat"] = "h-WS2"
    raw_b["flake"]["cif_path"] = f"{MAT}/h-WS2.cif"
    raw_b["flake"]["pot_path"] = f"{POT}/sw_lammps/t-WS2.sw"

    cfg_a = AFMSimulationConfig(**raw_a, settings=load_settings())
    cfg_b = AFMSimulationConfig(**raw_b, settings=load_settings())
    sim_a = AFMSimulation(cfg_a, "out_a")
    sim_b = AFMSimulation(cfg_b, "out_b")

    name_a, name_b = sim_a._get_hpc_job_name(), sim_b._get_hpc_job_name()
    assert name_a != name_b
    assert name_a == "afm_h-MoS2+flake-h-MoS2"
    assert name_b == "afm_h-MoS2+flake-h-WS2"


def test_output_dir_and_job_name_unaffected_by_absent_afm_flake_section():
    """An AFM config with no [flake] section at all is unaffected."""
    from src.run.naming import output_dir_for
    from src.builders.afm import AFMSimulation

    raw = _afm_flake_raw()
    del raw["flake"]
    assert str(output_dir_for(raw, "afm")) == "afm/h-MoS2/40x_40y/sub_aSi_tip_Si_r5/K1/pot_sw_MoS2_wen"

    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    sim = AFMSimulation(cfg, "out")
    assert sim._get_hpc_job_name() == "afm_h-MoS2"


def test_sheetonsheet_flake_omit_mat_directory_still_byte_identical():
    """The sheet-on-sheet flake convention this fix must not disturb: a
    [Flake] section that omits mat (reuses the substrate's) still builds into
    today's directory -- unchanged by teaching flake_material_token about the
    AFM family's lowercase 'flake' key too."""
    from src.run.naming import output_dir_for

    no_flake = {"general": {"temp": 300}, "2D": {"mat": "h-MoS2", "x": 100, "y": 100}}
    reused_flake = {**no_flake, "Flake": {"shape": "hexagon", "edge_length": 5.0}}

    assert output_dir_for(no_flake, "sheetonsheet") == output_dir_for(reused_flake, "sheetonsheet")
