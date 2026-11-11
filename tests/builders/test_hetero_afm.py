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
from tests.helpers.hetero import MAT, POT, _afm_flake_raw, _afm_hetero_raw

LMP = shutil.which("lmp_serial")


def _afm_homogeneous_raw():
    raw = _afm_hetero_raw()
    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    # Dropping to one sheet leaves validate_tip_sheet_footprint in play (no
    # longer exempted by the hetero >=2-sheets path): _afm_hetero_raw's 12x12
    # cell relies on hetero replication to clear the tip.r=10.0 floor, which
    # a single homogeneous sheet no longer gets, so widen it here.
    raw["2D"]["x"] = raw["2D"]["y"] = 50.0
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
    replicate_hetero_in_plane(stack.data_path, out, 2, 2)
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


def test_layout_guard_trips_on_type_count_mismatch(tmp_path):
    # builders-afm.md finding 7: only the third (element-wise permutation)
    # branch had a dedicated negative test. The simpler count-mismatch guard
    # that precedes it (data file missing a whole sheet type) was untested.
    from src.builders.afm.hetero import assert_sheet_block_matches_data
    pm = _fake_pm_with_sheet_block([("MoS2", "Mo", 95.9, None), ("MoS2", "S", 32.1, "edge"),
                                    ("WS2", "W", 183.8, None), ("WS2", "S", 32.1, "edge")])
    data = _write_data_with_masses(
        tmp_path / "short.data",
        masses=[("MoS2", "Mo", 95.9, None), ("MoS2", "S", 32.1, "edge"),
                ("WS2", "W", 183.8, None)])   # one type short of the PM's 4
    with pytest.raises(ValueError, match="layout mismatch"):
        assert_sheet_block_matches_data(pm, data, offset=0)


def test_layout_guard_trips_on_block_offset_mismatch(tmp_path):
    # finding 7: the block-offset guard (sheet block must start at
    # offset + 1, i.e. right after sub+tip) was also untested. Equal counts,
    # matching masses in order -- only the wrong starting offset trips it.
    from src.builders.afm.hetero import assert_sheet_block_matches_data
    pm = _fake_pm_with_sheet_block([("MoS2", "Mo", 95.9, None), ("MoS2", "S", 32.1, "edge"),
                                    ("WS2", "W", 183.8, None), ("WS2", "S", 32.1, "edge")])
    data = _write_data_with_masses(
        tmp_path / "offset.data",
        masses=[("MoS2", "Mo", 95.9, None), ("MoS2", "S", 32.1, "edge"),
                ("WS2", "W", 183.8, None), ("WS2", "S", 32.1, "edge")])
    # pm's sheet block actually starts at type 1 (fresh TypeRegistry); demand
    # offset=3 (expected start type 4) so the check fails on the offset alone.
    with pytest.raises(ValueError, match="block starts at type"):
        assert_sheet_block_matches_data(pm, data, offset=3)


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
    from tests.helpers.lattice import _flake_lattice_orientation

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
# --- The hetero stack must land the intended gap above the substrate ---------
#
# `read_data ... shift` translates a file's ATOMS, and hetero_rep.data's atoms
# do not start at z=0: assemble_hetero_stack pads ~7.5 A below them. Shifting by
# `sub_thickness + gap_sub_sheet` alone therefore put the stack that far too
# high -- a measured 12.23 A above the substrate's top atom against an intended
# 4.665 A, outside `lj/cut 11.0`, so every hetero run started with the stack
# seeing no substrate and dropped ~11 A during relaxation. This pins the
# contract against a real data file, not against the builder's own bookkeeping.


def _padded_stack_data(path, *, bottom, top, box_lo, box_hi):
    """A minimal LAMMPS data file whose atoms sit `bottom`..`top` inside a box
    padded below and above them -- the shape hetero_rep.data really has."""
    path.write_text(
        "Synthetic padded stack\n\n2 atoms\n1 atom types\n\n"
        f"0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n{box_lo} {box_hi} zlo zhi\n\n"
        "Masses\n\n1 95.95\n\n"
        f"Atoms # atomic\n\n1 1 0.0 0.0 {bottom}\n2 1 1.0 1.0 {top}\n",
        encoding="utf-8")
    return path


def test_hetero_stack_lands_the_intended_gap_above_the_substrate(tmp_path):
    """z_positions['sheet'] is a SHIFT, so it must carry the stack's own bottom
    offset out: shift + (file's lowest atom z) == sub.thickness + gap_sub_sheet.

    Driven through the real ``_measure_stack_atom_extent`` on a real file and
    the real ``_calculate_z_positions`` against a real PotentialManager (the one
    that supplies the gap), so nothing here restates the arithmetic under test.
    """
    from src.builders.afm.hetero import HeteroAFMSimulation
    from src.builders.construction.hetero_stack import register_sheet_block
    from src.potentials.manager import PotentialManager

    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))

    # The gap the placement is supposed to realise, from the same PM the
    # builder uses -- not a hardcoded 4.665.
    pm = PotentialManager(cfg.settings)
    pm.register_component('sub', cfg.sub)
    pm.register_component('tip', cfg.tip)
    sheet_names = register_sheet_block(pm, cfg.sheets)
    gap_sub_sheet = max(pm.calculate_gap('sub', n, buffer=0.5) for n in sheet_names)
    sim.pm[1] = pm

    rep = _padded_stack_data(tmp_path / "hetero_rep.data",
                             bottom=7.5, top=18.6, box_lo=-2.5, box_hi=28.6)
    sim._hetero_rep_path = rep
    sim._hetero_rep_box = (20.0, 20.0, 31.1)
    bottom, top = sim._measure_stack_atom_extent(rep)
    assert (bottom, top) == pytest.approx((7.5, 18.6))     # the padding is real
    sim._hetero_stack_atom_bottom, sim._hetero_stack_atom_top = bottom, top

    sim._calculate_z_positions(1, tip_radius=5.0)

    shift = sim.z_positions[1]["sheet"]
    assert shift + bottom == pytest.approx(cfg.sub.thickness + gap_sub_sheet)
    # i.e. the shift is NOT the intended height itself -- that was the bug.
    assert shift == pytest.approx(cfg.sub.thickness + gap_sub_sheet - bottom)
    # and the tip still starts above the stack's real top.
    assert sim.z_positions[1]["tip"] > shift + top


def test_hetero_z_placement_refuses_to_guess_the_stack_offset(tmp_path):
    """Without the measurement the old silent-7.5-A-error is the only thing
    that could be produced, so the placement refuses rather than assuming 0."""
    from src.builders.afm.hetero import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    with pytest.raises(ValueError, match="_build_hetero_sheet must run"):
        sim._calculate_z_positions(1, tip_radius=5.0)


# --- Finding 8: a [flake] on the HETERO AFM drive (hetero.py's own copy of the
#     flake build + twist block, previously unreached by any test) ---
#
# HeteroAFMSimulation._build_components is a full override of
# AFMSimulation._build_components (see its docstring: the base builds the
# substrate inline, so the tip / flake / langevin blocks are re-expressed rather
# than hooked). That makes the flake build+twist a SECOND copy of the same code
# -- decision row AFM-2, pair 1 -- and until these tests no test built any hetero
# AFM variant with a [flake] section at all, so only the base.py copy was ever
# exercised (test_afm_flake_twist_rotates_and_runs, above, uses the plain
# single-material AFMSimulation).


def _hetero_flake_raw(twist=None, with_sub=True):
    """The previously-untested combination: a two-material [2D-1]/[2D-2] hetero
    AFM config that ALSO carries a [flake] section.

    Composed from the two halves that already exist -- _afm_hetero_raw's sheet
    pair and _afm_flake_raw's hexagonal MoS2 flake -- so the flake here is the
    identical section the homogeneous D3b flake tests use.
    """
    raw = _afm_hetero_raw()
    raw["flake"] = dict(_afm_flake_raw()["flake"])
    if twist is not None:
        raw["general"]["twist_angle"] = twist
    if not with_sub:
        del raw["sub"]
    return raw


def _atom_type_histogram(data_path):
    """``{atom type: count}`` read straight out of a LAMMPS data file."""
    lines = Path(data_path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    hist = {}
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) >= 5:
            hist[int(fields[1])] = hist.get(int(fields[1]), 0) + 1
    return hist


def _build_hetero_flake_components(out_dir, monkeypatch, twist=None):
    """Run ``HeteroAFMSimulation._build_components`` with a flake configured.

    Only the flake branch is under test, so the two components that are NOT --
    the LAMMPS-backed tip build and the amorphous substrate -- are taken out of
    the way: ``components.build_tip`` is stubbed (patched on the module object
    hetero.py itself calls through) and ``[sub]`` is dropped, which also takes
    the langevin block with it. ``build_flake_component`` and
    ``_maybe_twist_flake`` then run for real, needing no LAMMPS, so this test
    runs in the fast suite.
    """
    from src.builders.construction import components
    from src.builders.afm.hetero import HeteroAFMSimulation

    def _stub_tip(tip_cfg, build_dir, settings):        # noqa: ARG001
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / "tip.lmp"
        path.write_text("LAMMPS data\n\n0 atoms\n0 atom types\n")
        return path, float(tip_cfg.r)

    monkeypatch.setattr(components, "build_tip", _stub_tip)

    cfg = AFMSimulationConfig(**_hetero_flake_raw(twist=twist, with_sub=False),
                              settings=load_settings())
    # nose-hoover so the langevin region surgery (which would run against the
    # stub tip file) is skipped; it is not what this test covers.
    cfg.settings.thermostat.type = 'nose-hoover'
    sim = HeteroAFMSimulation(cfg, str(out_dir))
    # The real build sets this in _build_sheet_for_layers, which this test
    # skips; the flake's PBC clearance check reads it. r=10 replicates the
    # h-MoS2/h-MoS2 coincidence cell 4x4 -> 50.96 x 44.13 A (the measured
    # figure in future-works, "No PBC self-interaction guard for [flake]").
    sim._hetero_rep_box = (50.96, 44.13, 20.0)
    built = sim._build_components(Path(out_dir) / "build")
    return sim, built


def test_hetero_afm_build_components_builds_the_flake(tmp_path, monkeypatch):
    """The hetero builder's flake branch produces a real, corner-typed flake file.

    Asserts on the artifact the branch writes -- the carved MoS2 hexagon and its
    interleaved core/corner type split -- not on a call-happened spy, so removing
    the ``if self.config.flake is not None`` branch from hetero.py fails it.
    """
    from ase.io import read
    from src.builders.utils.lammps_data import get_model_dimensions

    sim, (_tip_path, tip_radius, sub_path) = _build_hetero_flake_components(
        tmp_path, monkeypatch)

    # This really is the hetero builder's path (two folded sheets), not base's.
    assert len(sim.config.sheets) == 2
    assert sub_path is None and tip_radius == 10.0

    assert sim.flake_path is not None, \
        "hetero _build_components did not build the configured [flake]"
    assert sim.flake_path.exists()
    assert sim.flake_path.name == "flake_h-MoS2_hexagon.data"

    # flake_dims is the built file's own box -- the numbers _calculate_z_positions
    # and build_render_context later read back off it -- not a config echo.
    dims = get_model_dimensions(sim.flake_path)
    assert set(sim.flake_dims) >= {'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'}
    for key in ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'):
        assert sim.flake_dims[key] == pytest.approx(float(dims[key]))

    # A hexagon really was carved out of the MoS2 monolayer, and
    # apply_flake_corner_types split every element into an interleaved
    # core (2t-1) / corner (2t) pair -- the layout PotentialManager's
    # regions=[None, 'corner'] registration is numbered against.
    atoms = read(str(sim.flake_path), format="lammps-data")
    assert len(atoms) > 0
    hist = _atom_type_histogram(sim.flake_path)
    assert sorted(hist) == [1, 2, 3, 4], f"expected 2 elements x core/corner: {hist}"
    assert hist[2] > 0 and hist[4] > 0, f"no atoms typed as flake corners: {hist}"
    assert sum(hist.values()) == len(atoms)


def test_hetero_afm_build_components_twists_the_flake(tmp_path, monkeypatch):
    """``_maybe_twist_flake`` fires on the hetero copy of the branch too.

    Twisted vs untwisted builds of the same config: the flake's LATTICE is
    rotated by the requested angle while its box is untouched (an atoms-only
    twist -- the flake is finite, so this is a genuine relative twist against
    the hetero stack below it).
    """
    import numpy as np
    from ase.io import read
    from tests.helpers.lattice import _flake_lattice_orientation

    sim_tw, _ = _build_hetero_flake_components(tmp_path / "tw", monkeypatch, twist=18.0)
    sim_nt, _ = _build_hetero_flake_components(tmp_path / "nt", monkeypatch)

    a_tw = read(str(sim_tw.flake_path), format="lammps-data")
    a_nt = read(str(sim_nt.flake_path), format="lammps-data")

    assert len(a_tw) == len(a_nt)                                    # atoms conserved
    assert np.allclose(np.array(a_tw.cell), np.array(a_nt.cell), atol=1e-6)  # box fixed

    rel = (_flake_lattice_orientation(a_tw) - _flake_lattice_orientation(a_nt)) % 60.0
    assert abs(min(rel, 60.0 - rel) - 18.0) < 1.0


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_with_a_flake_builds_and_runs(tmp_path):
    """End-to-end counterpart of the two build-only tests above: a REAL hetero
    stack with a finite MoS2 flake pressed onto it by the tip, built and run.

    A new smoke rather than a [flake] bolted onto
    ``test_hetero_afm_runs_approach_indent_slide``: that one is the D2 capstone
    (replication, sheet_1/sheet_2 pair_coeff, 2D_all groups, substrate box) and
    adding a flake would both change what a failure there means and -- while the
    combination stays unwired -- take the whole D2 smoke down with it.

    Sized exactly like the D2 capstone (tip.r = 5 -> 2x2 replication above the
    LJ-cutoff floor, temp = 1 K so the indent settle loop converges,
    tip_reduction_factor = 0.7 so the langevin thermo band is populated), with
    one addition: the flake is shrunk to edge_length 6.

    That is a hard constraint, not a tidy-up. The 2x2 replicated cell is only
    ~25.5 x 22.1 A, and the [flake] section the two tests above share carves a
    hexagon spanning ~19 x 16 A -- which would leave the flake ~6 A from its own
    periodic image, well inside `pair_style ... lj/cut 11.0`. edge_length 6
    spans ~9.6 x 8.3 A (30 atoms, all four core/corner types populated), so the
    image separation clears the cutoff in both directions. Nothing validates
    this: the AFM tip/sheet footprint rule has no flake counterpart, so an
    oversized flake on a small hetero cell builds silently.

    Measured caveat, recorded so it is not rediscovered as a surprise: this
    flake SHEDS ATOMS under the indent. Two runs lost 1 and 2 of its 30 atoms
    respectively (they end up 40-95 A up, inside the box, so LAMMPS reports no
    lost atoms and the job succeeds). That is the config, not the builder --
    10 nN, inherited from the D2 capstone where it presses on a full periodic
    sheet, is ~100 GPa over a 30-atom flake's contact area. The assertions
    below deliberately gate on the WIRING being healthy (the flake is read in,
    typed, corner-restrained, and the run completes writing its load state);
    a flake that survives its load is a config question -- a gentler force or
    a bigger flake on a bigger cell -- and needs the missing footprint
    validator to be posed properly.
    """
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _hetero_flake_raw()
    raw["general"]["force"] = 10.0
    raw["general"]["temp"] = 1.0
    raw["tip"]["r"] = 5.0
    raw["flake"]["edge_length"] = 6.0
    raw["flake"]["corner_radius"] = 2.5
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7
    cfg.settings.simulation.slide_run_steps = 200

    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()                     # currently raises KeyError('flake')

    run_dir = Path(tmp_path)
    # The flake is read into the system and corner-restrained by the templates.
    init = run_dir / "L1" / "lammps" / "system.in"
    init_text = init.read_text()
    assert sim.flake_path.name in init_text
    assert "group           flake_corner type" in init_text

    # The flake's types are registered AFTER both sheets, so read_data's
    # `offset` lands the flake file's 4 types exactly on the PM's flake block.
    # Checked against the emitted settings file, not against the builder's own
    # bookkeeping -- a wrong offset here is silent wrong results, not a crash.
    settings_text = (run_dir / "L1" / "lammps" / "system.in.settings").read_text()
    flake_ids = sim.groups[1]["flake_types"].split()
    assert f"group flake_all type {' '.join(flake_ids)}" in settings_text
    offset_flake = len(sim.groups[1]["sub_types"].split()) \
        + len(sim.groups[1]["tip_types"].split()) \
        + len(sim.groups[1]["sheet_types"].split())
    assert f"group flake offset {offset_flake} 0 0 0 0" in init_text
    assert [int(i) for i in flake_ids] == list(
        range(offset_flake + 1, offset_flake + 1 + len(flake_ids)))

    # And it is placed ON the stack, not above the padded box: the flake's
    # lowest atom must sit within a bond-length-ish gap of the stack's highest.
    from src.interfaces.ase import read_lammps_data
    sheet_z = sim.z_positions[1]["sheet"]
    stack_top = read_lammps_data(sim._hetero_rep_path).positions[:, 2].max() + sheet_z
    flake_bottom = (read_lammps_data(sim.flake_path).positions[:, 2].min()
                    + sim.z_positions[1]["flake"])
    gap = flake_bottom - stack_top
    assert 2.0 < gap < 7.0, (
        f"flake sits {gap:.2f} A above the stack -- the padded box height "
        f"(~12.5 A too high) leaked back into the placement" )

    proc, out = _run_lammps_afm(run_dir, init)
    assert proc.returncode == 0, f"system.in exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost with a flake on the hetero stack:\n{out[-3000:]}"
    assert "pair coeffs are not set" not in out.lower(), \
        f"incomplete pair_coeff with a flake registered:\n{out[-3000:]}"
    loads = list((run_dir / "L1" / "data").glob("load_*N.data"))
    assert loads and all(p.stat().st_size > 0 for p in loads), \
        f"no non-empty load_*N.data written by the hetero-flake indent: {loads}"


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
    assert str(dir_a) == "afm/h-MoS2+flake-h-MoS2/40x_40y/sub_amorph-Si_tip_Si_r5/K1/pot_sw_MoS2_wen"
    assert str(dir_b) == "afm/h-MoS2+flake-h-WS2/40x_40y/sub_amorph-Si_tip_Si_r5/K1/pot_sw_MoS2_wen"


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
    assert str(output_dir_for(raw, "afm")) == "afm/h-MoS2/40x_40y/sub_amorph-Si_tip_Si_r5/K1/pot_sw_MoS2_wen"

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


# --- Interlayer styles between the hetero stack's layers (spec 2026-11-11) ---

def _hetero_afm_sim(tmp_path, monkeypatch, style, flake=False):
    """A HeteroAFMSimulation ready for _generate_potentials, without LAMMPS.

    The sheet-block guard reads hetero_rep.data, which only a real stack
    build writes; it is stubbed (the ilp full build below exercises it).
    """
    import src.builders.afm.hetero as ha
    from src.core.config import GlobalSettings
    monkeypatch.setattr(ha, "_assert_sheet_block_matches_data", lambda *a, **k: None)
    raw = _afm_hetero_raw()
    raw["general"]["interlayer_style"] = style
    if flake:
        raw["flake"] = _afm_flake_raw()["flake"]
    cfg = AFMSimulationConfig(**raw, settings=GlobalSettings())
    sim = ha.HeteroAFMSimulation(cfg, str(tmp_path))
    _prep_layer_dirs(sim, 1)
    return sim


def test_hetero_afm_ilp_acts_between_the_stack_layers_only(tmp_path, monkeypatch):
    sim = _hetero_afm_sim(tmp_path, monkeypatch, "ilp")
    pm = sim._generate_potentials(1)
    text = (sim.output_dir_layer[1] / "lammps" / "system.in.settings").read_text()
    ilp = [ln for ln in text.splitlines() if ln.startswith("pair_coeff * * ilp/tmd")]
    assert len(ilp) == 1
    # sub (3) + tip (3) NULL; sheet_1 Mo S S (wen's S split); sheet_2 W S (t-WS2
    # labels S once too, so it splits as well); virtual NULL
    assert ilp[0].split("TMD.ILP ")[1].split(" #")[0].split() == (
        ["NULL"] * 6 + ["Mo", "S", "S", "W", "S", "S", "NULL"])
    s1 = pm.types.ids_by_component("sheet_1")
    s2 = pm.types.ids_by_component("sheet_2")
    lj = [ln.split("# ")[1] for ln in text.splitlines()
          if ln.startswith("pair_coeff") and " lj/cut " in ln and "# " in ln]
    # tip/substrate-sheet LJ stays; the sheet_1-sheet_2 LJ is gone
    for tag in ("Si(tip)-Mo(sheet_1)", "Si(sub)-W(sheet_2)", "Si(tip)-S(sheet_2)"):
        assert tag in lj
    assert not [t for t in lj if "(sheet_1)-" in t and t.endswith("(sheet_2)")]
    assert pm.needs_molecule_ids
    assert len(s1) == 3 and len(s2) == 3


def test_hetero_afm_molecule_ids_by_z_band(tmp_path, monkeypatch):
    sim = _hetero_afm_sim(tmp_path, monkeypatch, "ilp")
    sim.pm[1] = sim._generate_potentials(1)
    sim.z_positions[1] = {"sheet": 2.5}
    sim._hetero_layer_bands = [{"idx": 1, "zlo": 6.0, "zhi": 11.0},
                               {"idx": 2, "zlo": 13.0, "zhi": 18.0}]
    ctx = sim._molecule_id_context(1)
    types = sim.groups[1]["sheet_types"]
    assert ctx["needs_molecule_ids"] and ctx["molecule_ids_in_data"]
    assert ctx["molecule_groups"] == [
        {"name": "mol_layer_1", "types": types, "zlo": 8.5, "zhi": 13.5},
        {"name": "mol_layer_2", "types": types, "zlo": 15.5, "zhi": 20.5}]


def test_hetero_afm_lj_sets_no_molecule_ids(tmp_path, monkeypatch):
    sim = _hetero_afm_sim(tmp_path, monkeypatch, "lj")
    sim.pm[1] = sim._generate_potentials(1)
    assert sim._molecule_id_context(1) == {
        "needs_molecule_ids": False, "molecule_groups": [],
        "molecule_ids_in_data": False}


def test_hetero_afm_flake_with_ilp_is_refused(tmp_path, monkeypatch):
    sim = _hetero_afm_sim(tmp_path, monkeypatch, "ilp", flake=True)
    with pytest.raises(ValueError, match="flake"):
        sim._generate_potentials(1)


def test_molecule_id_band_entries_render():
    """common/molecule_ids.lmp's z-band form: types inside the band only."""
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader("src.templates"),
                      trim_blocks=True, lstrip_blocks=True)
    out = env.from_string(
        "{% import 'common/molecule_ids.lmp' as m %}{{ m.emit_molecule_ids(g) }}"
    ).render(g=[{"name": "mol_layer_1", "types": "7 8 9", "zlo": 1.0, "zhi": 2.5}])
    assert out.splitlines()[1:] == [
        "fix             mol_ids all property/atom mol ghost yes",
        "region          mol_layer_1 block INF INF INF INF 1.0000 2.5000 units box",
        "group           il_band region mol_layer_1",
        "group           il_types type 7 8 9",
        "group           mol_layer_1 intersect il_band il_types",
        "group           il_band delete",
        "group           il_types delete",
        "set group mol_layer_1 mol 1",
    ]


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_ilp_build_carries_molecule_ids_in_the_load_file(tmp_path):
    """system.in sets the IDs by z-band and keeps them in load_*.data;
    slide.in reads them back rather than re-banding relaxed positions."""
    from src.builders.afm.hetero import HeteroAFMSimulation
    raw = _afm_hetero_raw()
    raw["general"]["interlayer_style"] = "ilp"
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim.build()
    lmp_dir = sim.output_dir_layer[1] / "lammps"
    system = (lmp_dir / "system.in").read_text()
    slide = (lmp_dir / "slide.in").read_text()
    assert "set group mol_layer_1 mol 1" in system
    assert "set group mol_layer_2 mol 2" in system
    assert " nofix" not in system
    assert "read_data       ${load_file} fix mol_ids NULL Molecules" in slide
    assert slide.index("fix             mol_ids") < slide.index("read_data       ${load_file}")
    assert "set group" not in slide
