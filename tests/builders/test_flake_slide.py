"""Tests for D4 flake-driven slide (flake on a substrate sheet, no tip).

The flake (single-material or a hetero bilayer) is driven across a substrate
sheet by a virtual-atom / COM spring. The substrate sheet's bottom band is
fixed; its interior + the flake are thermostatted; the configured pressure
loads the flake over its projected area; lateral friction is read from the
bottom-sheet reaction force (the many-body ``sw`` observable) and the normal
load from ``v_fz``.
"""
import re
import shutil
from pathlib import Path

import pytest
from frictionsim2d.core.config import SheetOnSheetSimulationConfig, load_settings
# Reuse the D1 slide runner + friction-record parser for the e2e smoke, and its
# path extractors for the sim-root resolution check.
from tests.helpers.hetero import (
    _read_friction_records,
    _referenced_input_paths,
    _run_lammps_slide,
    _staged_potential_paths,
)

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _settings():
    """Settings a full standalone ``build()`` can complete on.

    ``hpc.modules`` gates job-script generation, which ``build()`` now runs
    without swallowing the failure of -- mirroring ``test_hetero_slide``. The
    CLI is unaffected either way: ``run_simulations`` calls
    ``set_base_output_dir``, so per-simulation generation returns early there.
    """
    settings = load_settings()
    settings.hpc.modules = ["lammps"]
    return settings


def _minimal_flake_slide_context():
    """Fill every key ``flake/slide.lmp`` references.

    Mirrors ````tests/builders/test_hetero_slide.py::_minimal_slide_context```` (the D1 slide
    keys), but renames the substrate sheet's per-layer z-bands to ``sheet_layers``
    and adds the flake-drive keys (``flake_group_types``, ``corner_hold``,
    ``flake_corner_types``, ``flake_corner_k``, ``virtual_offset``).
    """
    sheet_layers = [{"idx": i, "zlo": 5.0 * i, "zhi": 5.0 * i + 3.0} for i in range(1, 3)]
    return {
        "temp": 300,
        "xlo": 0.0,
        "xhi": 40.0,
        "ylo": 0.0,
        "yhi": 40.0,
        "zhi": 40.0,
        "data_file": "flake_slide.data",
        "potential_file": "lammps/system.in.settings",
        # Real types 1..6 (sheet 1..4, flake 5..6); the reserved virtual driver
        # is a DEDICATED extra type (7), distinct from every real/flake type.
        "num_atom_types": 7,
        "ngroups": 6,
        "n_layers": 2,
        "constraint_mode": "none",
        "drive_method": "virtual_atom",
        "thermostat_type": "langevin",
        "atom_style": "atomic",
        "pot_type": "sw",
        "has_internal_lj": True,
        "pressures": 0.0,
        "scan_speed_config": 100,
        "scan_angle_config": 0,
        "scan_angle_force": None,
        "timestep": 0.001,
        "thermo": 1000,
        "run_steps": 4000,
        "min_style": "cg",
        "minimization_command": "minimize 1.0e-8 1.0e-10 10000 100000",
        "neighbor_list": "2.0",
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "results_freq": 1000,
        "results_file_pattern": "results/friction_p${pressure}_a${a}_s${speed}",
        "dump_enabled": False,
        "dump_freq": 1000,
        "dump_file_pattern": "visuals/slide.lammpstrj",
        "driving_spring_ev": 3.121,
        "bond_spring_ev": 4.994,
        "lat_c": 3.16,
        "ev_a_to_nn": 1.602176565,
        "ev_a3_to_gpa": 160.2176565,
        "sheet_layers": sheet_layers,
        "flake_group_types": "5 6",
        "flake_corner_types": "6",
        "flake_corner_k": 10.0,
        "corner_hold": True,
        "virtual_offset": 10.0,
        "flake_area": 64.951905,
        "flake_shape": "hexagon",
        "flake_edge_length": 5.0,
    }


def _render_flake_slide_for_test(context):
    """Render flake/slide.lmp from a partial context, filling obs_ctx.

    Replaces the production `render_flake_slide` + `_with_obs_ctx` pair, which
    had no caller outside these tests and was removed (decision row B4-1).
    Mirrors `FlakeSlideSimulation._build_obs_ctx`: sim_type 'flake', the
    fallback contact cutoff, and NO extra_filter -- flake declares no
    interlayer channel at all, so unlike hetero there is nothing to drop.

    Same shape as `tests/builders/test_template_contract.py`'s own
    `_render_flake_slide`, which has always built its obs_ctx this way.
    """
    from jinja2 import Environment

    from frictionsim2d.builders.afm.base import _CONTACT_CUTOFF_FALLBACK
    from frictionsim2d.builders.utils.obs_ctx import build_obs_ctx
    from frictionsim2d.interfaces.jinja import PackageLoader

    if "obs_ctx" not in context:
        n_layers = context["n_layers"]
        context = {**context, "obs_ctx": build_obs_ctx(
            sim_type="flake",
            n_layers=n_layers,
            results_file_pattern=context.get(
                "results_file_pattern", "results/friction.txt"),
            use_langevin=(context.get("thermostat_type") == "langevin"),
            enabled_tiers={"window": True, "raw": True, "peratom": True},
            contact_cutoff=_CONTACT_CUTOFF_FALLBACK,
            results_freq=context.get("results_freq", 1000),
            optional_channels=(),
            interface_layers=(n_layers // 2, n_layers // 2 + 1),
        )}
    env = Environment(loader=PackageLoader("frictionsim2d.templates"),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template("flake/slide.lmp").render(context)


def test_flake_slide_drives_flake_group_not_layer():
    ctx = _minimal_flake_slide_context()
    ctx["drive_method"] = "virtual_atom"
    ctx["corner_hold"] = True
    script = _render_flake_slide_for_test(ctx)

    # Drive remapped to the flake group (not a sheet layer_N). The template
    # keeps LAMMPS' aligned columns (``fix`` padded), so match whitespace-tolerantly.
    assert re.search(r"fix\s+spr flake spring couple virtual", script)
    # Corner-hold rendered (default ON).
    assert re.search(r"fix\s+corner_hold flake_corner spring/self", script)
    # Single read of the assembled sheet+flake data file.
    assert script.count("read_data") == 1
    # No tip approach.
    assert "reduce max z" not in script
    # Top-sheet-layer drive/load blocks are DROPPED (plan item 4/5).
    assert "fstage_top" not in script      # top sheet layer not rigidified/driven
    # no aveforce anywhere: the sheet top is not loaded, and the flake's load
    # is a per-atom addforce (aveforce would average its forces away)
    assert not re.search(r"^fix\s+\S+\s+\S+\s+aveforce", script, re.M)
    # No undefined layer_0 (a 2-layer sheet must never reference n_layers-2).
    assert "layer_0" not in script


def test_flake_slide_defines_sheet_layer_groups_and_flake_group():
    ctx = _minimal_flake_slide_context()
    script = _render_flake_slide_for_test(ctx)

    # Substrate-sheet layer groups defined by region.
    assert "group           layer_1 region reg_layer_1" in script
    assert "group           layer_2 region reg_layer_2" in script
    # Sheet bottom band fixed.
    assert "fix             fsbot layer_1 setforce 0.0 0.0 0.0" in script
    # Flake group defined by type.
    assert "group           flake type 5 6" in script
    # Friction proxy = bottom-sheet reaction force (v_sx/v_sy).
    assert "variable        xfrict equal v_sx" in script
    # Flake COM logged to the results file (Task-4 assertion needs these columns).
    assert "compute         COM_flake flake com" in script
    assert "v_comx_flake" in script and "v_comy_flake" in script


def test_flake_slide_corner_hold_can_be_disabled():
    ctx = _minimal_flake_slide_context()
    ctx["corner_hold"] = False
    script = _render_flake_slide_for_test(ctx)
    assert "corner_hold" not in script


# --------------------------------------------------------------------------- #
# Builder: substrate sheet + flake placement + single combined PM/data file
# --------------------------------------------------------------------------- #

def _flake_on_sheet_cfg(hetero_flake: bool = False):
    """A flake-on-sheet config: single ``[2D]`` MoS2 substrate (2 layers) + a
    ``[Flake]`` hexagon. ``hetero_flake=True`` adds ``[Flake-1]``/``[Flake-2]``
    (a MoS2/WS2 bilayer flake); otherwise the single-material flake reuses the
    substrate ``[2D]`` material (the builder fills its material fields).
    """
    # 24 A, not the 16 A this fixture used to carve: the flake is finite and
    # the sheet is periodic, so the cell has to clear the flake's ~9.6 x 8.3 A
    # span by more than the 11 A pair cutoff or the flake slides against its
    # own image (check_flake_pbc_clearance). At 16 A the gap was 6.4 x 7.7 A.
    raw = {
        "general": {"temp": 300, "scan_speed": 100},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 24.0, "y": 24.0, "layers": [2]},
        "Flake": {"shape": "hexagon", "edge_length": 5.0, "corner_radius": 3.5,
                  "corner_spring_k": 10.0},
    }
    if hetero_flake:
        raw["Flake-1"] = {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
                          "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
                          "x": 12.0, "y": 12.0, "layers": [1]}
        raw["Flake-2"] = {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif",
                          "pot_path": f"{POT}/sw_lammps/t-WS2.sw", "pot_type": "sw",
                          "x": 12.0, "y": 12.0, "layers": [1]}
    return SheetOnSheetSimulationConfig(**raw, settings=_settings())


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_builder_places_flake_and_reserves_virtual(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.settings.simulation.drive_method = "virtual_atom"
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()

    slide = next(Path(tmp_path).rglob("*slide*.in"))
    text = slide.read_text()
    assert "spring couple virtual" in text
    assert re.search(r"fix\s+spr flake spring couple virtual", text)   # driven group is the flake
    assert "fsbot" in text and "setforce 0.0 0.0 0.0" in text          # sheet bottom fixed
    assert "corner_hold" not in text                                   # default OFF (follow-up E)
    assert re.search(r"^write_data\s+\S+/data/final_p\$\(v_pressure\)GPa", text, re.M)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_assembly_keeps_sheet_types_and_periodic_box(tmp_path):
    """The stacked substrate carries PER-LAYER types and its own periodic cell.

    Regressions: (1) the sheet was registered as one layer, so the flake's type
    offset landed on the sheet's upper types and ``group flake`` held the
    sheet's S atoms too; (2) ``read_data add`` grew the box to the flake file's
    carving box, leaving vacuum seams that cut the periodic substrate.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from frictionsim2d.interfaces.ase import read_lammps_data

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()
    run_dir = Path(sim.output_dir)

    sheet_file = next(run_dir.glob("build/h-MoS2_*.lmp"))
    sheet = read_lammps_data(sheet_file)
    n_sheet = int(re.search(r"^\s*(\d+)\s+atom types", sheet_file.read_text(), re.M).group(1))
    sheet_types = set(range(1, n_sheet + 1))      # 2 layers x (Mo, S) = 4
    assert n_sheet == 4
    slide = next(run_dir.rglob("*slide*.in")).read_text()
    flake_types = {int(t) for t in re.search(r"group\s+flake type ([\d ]+)", slide)
                   .group(1).split()}
    assert not sheet_types & flake_types, (sheet_types, flake_types)

    settings = (run_dir / "lammps" / "system.in.settings").read_text()
    assert re.search(r"^group layer_1 type 1 3$", settings, re.M)   # Mo_L1 S_L1
    assert re.search(r"^group layer_2 type 2 4$", settings, re.M)   # Mo_L2 S_L2

    combined = read_lammps_data(run_dir / "flake_slide.data")
    assert combined.cell[0, 0] == pytest.approx(sheet.cell[0, 0], abs=1e-4)
    assert combined.cell[1, 1] == pytest.approx(sheet.cell[1, 1], abs=1e-4)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_driven_group_is_flake_and_virtual_type_reserved(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.settings.simulation.drive_method = "virtual_atom"
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()

    data = next(Path(tmp_path).rglob("*flake_slide*.data"))
    text = data.read_text()
    m = re.search(r"(\d+)\s+atom types", text)
    assert m, "no 'atom types' header in the assembled data file"
    header_types = int(m.group(1))

    # Real types actually present in the Atoms section.
    real = set()
    in_atoms = False
    for raw in text.splitlines():
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        if s.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms:
            tok = s.split()
            if not tok[0].lstrip("-").isdigit():
                break
            if len(tok) >= 2:
                real.add(int(tok[1]))
    total_real = max(real)
    # The reserved virtual type is a DEDICATED extra type: header = real + 1,
    # and no real atom carries the virtual type id.
    assert header_types == total_real + 1
    slide_text = next(Path(tmp_path).rglob("*slide*.in")).read_text()
    assert re.search(rf"create_atoms\s+{total_real + 1}\s+single", slide_text)
    assert re.search(rf"group\s+virtual\s+type\s+{total_real + 1}\b", slide_text)


def _single_material_cfg():
    """A homogeneous single-[2D] sheet-on-sheet config with NO [Flake] section."""
    raw = {
        "general": {"temp": 300, "scan_speed": 1},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 8, "y": 8, "layers": [4]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=_settings())


def test_routing_flake_section_selects_flake_slide():
    from frictionsim2d.run.simulate import _select_sheet_builder_cls
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake is not None
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is FlakeSlideSimulation


def test_routing_no_flake_unchanged():
    from frictionsim2d.run.simulate import _select_sheet_builder_cls
    from frictionsim2d.builders.sheetonsheet import SheetOnSheetSimulation

    cfg = _single_material_cfg()
    assert cfg.flake is None
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is SheetOnSheetSimulation


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_builder_requires_multilayer_substrate(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.sheet.layers = [1]          # single-layer substrate -> no fixed bottom + interior
    cfg.sheets[0].layers = [1]
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    with pytest.raises(ValueError, match=r"[Aa]t least 2"):
        sim.build()


# --------------------------------------------------------------------------- #
# Output-dir / HPC-job-name collision: two flake configs differing ONLY in
# their flake material must not build into the same output directory (the
# defect: output_dir_for/material_token never looked at flake_sheets/[Flake],
# so they silently overwrote each other -- data loss, not just a name clash).
# --------------------------------------------------------------------------- #

def _flake_slide_raw(flake_mat=None):
    """A [2D]+[Flake] raw config dict (pre-pydantic), optionally overriding the
    flake's material away from the substrate's."""
    raw = {
        "general": {"temp": 300, "scan_speed": 100},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 16.0, "y": 16.0, "layers": [2]},
        "Flake": {"shape": "hexagon", "edge_length": 5.0, "corner_radius": 3.5,
                  "corner_spring_k": 10.0},
    }
    if flake_mat is not None:
        raw["Flake"]["mat"] = flake_mat
    return raw


def test_output_dir_differs_when_only_flake_material_differs():
    from frictionsim2d.run.naming import material_token, output_dir_for

    run_a = _flake_slide_raw(flake_mat="h-MoS2")
    run_b = _flake_slide_raw(flake_mat="h-WS2")

    assert material_token(run_a) != material_token(run_b)
    assert output_dir_for(run_a, "sheetonsheet") != output_dir_for(run_b, "sheetonsheet")


def test_hpc_job_name_differs_when_only_flake_material_differs(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg_a = SheetOnSheetSimulationConfig(**_flake_slide_raw(flake_mat="h-MoS2"), settings=_settings())
    cfg_b = SheetOnSheetSimulationConfig(**_flake_slide_raw(flake_mat="h-WS2"), settings=_settings())
    sim_a = FlakeSlideSimulation(cfg_a, str(tmp_path / "a"))
    sim_b = FlakeSlideSimulation(cfg_b, str(tmp_path / "b"))

    assert sim_a._get_hpc_job_name() != sim_b._get_hpc_job_name()


def test_output_dir_and_job_name_unaffected_by_absent_flake_section():
    """A config with no [Flake] section at all is unaffected by the fix."""
    from frictionsim2d.run.naming import output_dir_for

    no_flake = {
        "general": {"temp": 300},
        "2D": {"mat": "h-MoS2", "x": 100, "y": 100},
    }
    assert str(output_dir_for(no_flake, "sheetonsheet")) == "sheetonsheet/h-MoS2/100x_100y/K300"


def test_output_dir_byte_identical_when_flake_reuses_substrate_material(tmp_path):
    """A single-[2D]-plus-[Flake] config that omits the flake's own material
    (the common case -- the flake reuses the substrate's) must build into the
    SAME directory / job name as before this fix (no [Flake] section at all)."""
    from frictionsim2d.run.naming import output_dir_for

    no_flake = {"general": {"temp": 300}, "2D": {"mat": "h-MoS2", "x": 100, "y": 100}}
    reused_flake = {**no_flake, "Flake": {"shape": "hexagon", "edge_length": 5.0}}

    assert output_dir_for(no_flake, "sheetonsheet") == output_dir_for(reused_flake, "sheetonsheet")

    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    cfg = _flake_on_sheet_cfg(hetero_flake=False)  # omits flake.mat -> reuses [2D]'s
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    assert sim._get_hpc_job_name() == f"flake_{cfg.sheet.mat}"


def test_output_dir_distinguishes_hetero_flake_pair():
    """A [Flake-1]/[Flake-2] hetero flake must also be reflected in the token
    (it always names two materials explicitly, unlike a single [Flake])."""
    from frictionsim2d.run.naming import output_dir_for

    raw = _flake_slide_raw()
    raw["Flake-1"] = {"mat": "h-MoS2", "x": 12.0, "y": 12.0}
    raw["Flake-2"] = {"mat": "h-WS2", "x": 12.0, "y": 12.0}
    single = _flake_slide_raw()

    assert output_dir_for(raw, "sheetonsheet") != output_dir_for(single, "sheetonsheet")


# --------------------------------------------------------------------------- #
# End-to-end smoke: build a real MoS2 flake on a MoS2 sheet and RUN the drive.
# Reuses the D1 slide runner + friction-record parser (imported at module top).
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_flake_slide_runs_and_translates_flake_com(tmp_path):
    """Capstone D4 smoke: assemble a real MoS2 flake on a 2-layer MoS2 substrate
    sheet and actually RUN the virtual-atom drive in LAMMPS. Proves: run-dir-
    relative path resolution (data, settings, staged potentials, results), the
    baked reserved virtual type, a live flake drive (the flake COM translates),
    no lost atoms and complete pair coeffs. Relies on `corner_hold` defaulting
    to False (follow-up E) -- with it on the flake cannot translate at all,
    see `test_flake_slide_twisted_with_hold_on_keeps_corners_near_start`.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake.corner_hold is False
    cfg.settings.simulation.drive_method = "virtual_atom"
    # The template equilibrates a fixed 10000 steps, then defines the friction
    # fix (ave/time 1 1000 results_freq) and drives run_steps more. Nfreq=1000
    # -> first record at 11000; 4000 driven steps give records at
    # 11000/12000/13000/14000 (>=2, needed for the displacement check).
    cfg.settings.simulation.slide_run_steps = 4000
    # Fast drive so the translation dominates thermal COM jitter (~4 A over 4 ps).
    cfg.general.scan_speed = 100

    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()

    run_dir = sim.output_dir
    assert (run_dir / "flake_slide.data").exists()
    assert (run_dir / "lammps" / "system.in.settings").exists()

    slide = next(Path(run_dir).rglob("*slide*.in"))
    proc, out = _run_lammps_slide(run_dir, slide.name)

    # 1) Clean finish, no lost atoms, all pair coeffs set.
    assert proc.returncode == 0, f"lmp_serial exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost during flake slide:\n{out[-3000:]}"
    assert "All pair coeffs are not set" not in out, f"incomplete pair coeffs:\n{out[-3000:]}"

    # 1b) The virtual driver attaches to EXACTLY ONE reserved atom.
    m = re.search(r"(\d+) atoms in group virtual", out)
    assert m is not None, f"no 'atoms in group virtual' line:\n{out[-3000:]}"
    assert m.group(1) == "1", (
        f"group virtual has {m.group(1)} atoms, expected exactly 1 (the reserved "
        f"non-interacting virtual atom)"
    )

    # 2) A non-empty friction results file was written.
    friction = next(run_dir.rglob("friction_*"))
    assert friction.stat().st_size > 0, f"empty friction file: {friction}"

    # 3) The drive is LIVE: the flake's in-plane COM translated a nonzero net
    #    amount between the first and last record (the template logs
    #    v_comx_flake / v_comy_flake = xcm(flake,x/y)).
    cols, rows = _read_friction_records(friction)
    assert cols is not None, f"no column header parsed from {friction}"
    assert len(rows) >= 2, f"need >=2 records to measure displacement, got {len(rows)}"
    ix, iy = cols.index("v_comx_flake"), cols.index("v_comy_flake")
    dx = rows[-1][ix] - rows[0][ix]
    dy = rows[-1][iy] - rows[0][iy]
    disp = abs(dx) + abs(dy)
    assert disp > 0.5, (
        f"flake COM barely moved (dx={dx:.3f}, dy={dy:.3f}, |dx|+|dy|={disp:.3f} A); "
        f"virtual_atom drive appears dead"
    )


# --------------------------------------------------------------------------- #
# D3b: incommensurate flake twist on the sheet-on-sheet (D4) drive
# --------------------------------------------------------------------------- #

def test_flake_slide_routing_flake_plus_twist_selects_flake_slide():
    # [Flake] + twist must go to the FLAKE builder (D3b), not the D3 periodic hetero builder.
    from frictionsim2d.run.simulate import _select_sheet_builder_cls
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.general.twist_angle = 12.0
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is FlakeSlideSimulation


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_twist_rotates_the_flake(tmp_path):
    from ase.io import read
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from tests.helpers.lattice import _flake_lattice_orientation
    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.general.twist_angle = 20.0
    sim = FlakeSlideSimulation(cfg, str(tmp_path / "tw")); sim.build()
    cfg0 = _flake_on_sheet_cfg(hetero_flake=False)
    sim0 = FlakeSlideSimulation(cfg0, str(tmp_path / "nt")); sim0.build()
    # Compare the FLAKE COMPONENT (build/flake_*_<shape>.data), not the combined
    # sheet+flake file (flake_slide.data, sheet-dominated).
    fl = next((tmp_path / "tw").rglob("*hexagon.data"))
    fl0 = next((tmp_path / "nt").rglob("*hexagon.data"))
    rel = (_flake_lattice_orientation(read(str(fl), format="lammps-data"))
           - _flake_lattice_orientation(read(str(fl0), format="lammps-data"))) % 60.0
    assert min(rel, 60.0 - rel) > 5.0     # visibly twisted vs the untwisted flake


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_flake_slide_twisted_with_hold_off_translates(tmp_path):
    """`corner_hold` defaults to False (follow-up E): the twisted flake is
    deformable and translates under the COM drive, same as the untwisted case.

    Renamed from `test_flake_slide_twisted_runs_and_holds` -- that name
    described the corner-hold config, not this assertion (translation). The
    "holds" behaviour now has its own test below.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake.corner_hold is False
    cfg.general.twist_angle = 20.0
    cfg.settings.simulation.slide_run_steps = 4000
    cfg.general.scan_speed = 100
    sim = FlakeSlideSimulation(cfg, str(tmp_path)); sim.build()
    run_dir = sim.output_dir
    slide = next(Path(run_dir).rglob("*slide*.in"))
    proc, out = _run_lammps_slide(run_dir, slide.name)
    assert proc.returncode == 0 and "Lost atoms" not in out
    friction = next(run_dir.rglob("friction_*")); assert friction.stat().st_size > 0
    cols, rows = _read_friction_records(friction)
    ix, iy = cols.index("v_comx_flake"), cols.index("v_comy_flake")
    disp = abs(rows[-1][ix] - rows[0][ix]) + abs(rows[-1][iy] - rows[0][iy])
    assert disp > 0.5           # twisted flake still slides (drive live; superlubric contact is fine)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_flake_slide_twisted_with_hold_on_keeps_corners_near_start(tmp_path):
    """`corner_hold=True` tethers each corner atom to its START xy position
    (`fix spring/self`), so it is not "rigid while driven": it stops the
    flake translating at all. See task-C-debug-report.md §4 -- with
    `corner_radius=3.5` on this R=5 hexagon every flake atom is a corner atom,
    so the tether (10 eV/Å² per atom) heavily outweighs the ~3 eV/Å² COM drive
    spring, and the measured COM displacement was ~0.04 Å over the run.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.flake.corner_hold = True
    cfg.general.twist_angle = 20.0
    cfg.settings.simulation.slide_run_steps = 4000
    cfg.general.scan_speed = 100
    sim = FlakeSlideSimulation(cfg, str(tmp_path)); sim.build()
    run_dir = sim.output_dir
    slide = next(Path(run_dir).rglob("*slide*.in"))
    text = slide.read_text()
    assert "corner_hold" in text
    proc, out = _run_lammps_slide(run_dir, slide.name)
    assert proc.returncode == 0 and "Lost atoms" not in out
    friction = next(run_dir.rglob("friction_*")); assert friction.stat().st_size > 0
    cols, rows = _read_friction_records(friction)
    ix, iy = cols.index("v_comx_flake"), cols.index("v_comy_flake")
    disp = abs(rows[-1][ix] - rows[0][ix]) + abs(rows[-1][iy] - rows[0][iy])
    assert disp < 0.5           # held: the flake barely moves, unlike the hold-off case above


# --------------------------------------------------------------------------- #
# D3b Task 4: hetero-flake twist coverage + achieved-angle manifest
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_hetero_flake_slide_twist_rotates(tmp_path):
    from ase.io import read
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from tests.helpers.lattice import _flake_lattice_orientation
    cfg = _flake_on_sheet_cfg(hetero_flake=True)      # MoS2/WS2 hetero flake
    cfg.sheet.x = cfg.sheet.y = 30.0                  # room for the ~16 A hetero flake footprint
    cfg.general.twist_angle = 22.0
    sim = FlakeSlideSimulation(cfg, str(tmp_path / "tw")); sim.build()
    cfg0 = _flake_on_sheet_cfg(hetero_flake=True)
    cfg0.sheet.x = cfg0.sheet.y = 30.0
    sim0 = FlakeSlideSimulation(cfg0, str(tmp_path / "nt")); sim0.build()
    fl = next((tmp_path / "tw").rglob("*hexagon.data"))
    fl0 = next((tmp_path / "nt").rglob("*hexagon.data"))
    rel = (_flake_lattice_orientation(read(str(fl), format="lammps-data"))
           - _flake_lattice_orientation(read(str(fl0), format="lammps-data"))) % 60.0
    assert min(rel, 60.0 - rel) > 5.0


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_twist_angle_applied_recorded_in_manifest(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    cfg = _flake_on_sheet_cfg(hetero_flake=False); cfg.general.twist_angle = 17.5
    sim = FlakeSlideSimulation(cfg, str(tmp_path)); sim.build()
    manifest = next(Path(tmp_path).rglob("*manifest*"))
    text = manifest.read_text()
    assert "17.5" in text or "twist" in text.lower()


# --- HPC job scripts for a flake run --------------------------------------- #

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_paths_resolve_from_the_simulation_root(tmp_path):
    """Every path the flake slide references must resolve from the SIM ROOT.

    The generated job scripts run LAMMPS from the simulation root, not from the
    run dir, so run-dir-relative paths produce a script that aborts on
    submission with "cannot open sw potential file" -- after the build reported
    success. The potential spine needs a second registration for this: assembly
    runs LAMMPS with cwd = output_dir and can only open run-dir-relative paths,
    and pair_coeff lines are baked at registration time, so the sim-root prefix
    has to be re-registered once assembly is done (build step 4b). Mirrors
    test_hetero_slide's equivalent.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from frictionsim2d.run.hpc_dispatch import generate_hpc_scripts_for_root

    root = tmp_path / "simulation_root"
    run_dir = root / "sheetonsheet" / "h-MoS2" / "16x_16y" / "K300"
    cfg = _flake_on_sheet_cfg(hetero_flake=False)

    sim = FlakeSlideSimulation(cfg, str(run_dir))
    sim.set_base_output_dir(root)
    sim.build()

    assert generate_hpc_scripts_for_root(root, cfg.settings) is None
    assert "run.pbs" in sorted(p.name for p in (root / "hpc").iterdir())

    slide = (run_dir / "lammps" / "slide.in").read_text()
    referenced = _referenced_input_paths(slide)
    assert referenced, "slide.in references no input files at all"
    for path in referenced:
        assert (root / path).is_file(), (
            f"{path} does not resolve from the simulation root {root}")

    staged = _staged_potential_paths(
        (run_dir / "lammps" / "system.in.settings").read_text())
    assert staged, "no staged potential files referenced"
    for path in staged:
        assert (root / path).is_file(), (
            f"staged potential {path} does not resolve from {root}")


# --- the flake-footprint guard (builders-sheetonsheet.md finding 12) ---------
#
# `_assemble_flake_on_sheet` refuses a flake wider than the periodic substrate
# box. It is the ONLY footprint guard the flake path has and it was untested.
# It is also the near-miss neighbour of the flake PBC self-interaction gap filed
# in specs/TO-DO/future-works.md (recorded, not fixed, 2026-09-09): this guard
# catches a flake wider than the box, and says nothing about one that fits with
# less than lj/cut clearance to its own periodic image. A reader who finds this
# raise could reasonably assume PBC is handled. It is not.
#
# No LAMMPS: the raise precedes _run_lammps, and `pm` is only read after it.

def _write_slab(path, xs, cell_xy):
    """A minimal LAMMPS data file: one atom per x, at z=0, in a cell_xy box."""
    import numpy as np
    from ase import Atoms
    from ase.io import write

    cell = np.array([[cell_xy[0], 0.0, 0.0], [0.0, cell_xy[1], 0.0], [0.0, 0.0, 50.0]])
    atoms = Atoms(symbols=["H"] * len(xs),
                  positions=[[x, 1.0, 0.0] for x in xs], cell=cell, pbc=True)
    write(str(path), atoms, format="lammps-data", atom_style="atomic")
    return path


def _assemble(tmp_path, flake_xs, sheet_cell=(10.0, 10.0)):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    sim = FlakeSlideSimulation(_flake_on_sheet_cfg(), str(tmp_path))
    return sim._assemble_flake_on_sheet(
        sheet_path=_write_slab(tmp_path / "sheet.lmp", [1.0, 2.0], sheet_cell),
        flake_path=_write_slab(tmp_path / "flake.lmp", flake_xs, sheet_cell),
        n_sheet_types=1, pm=None,
        settings_path=tmp_path / "system.in.settings",
        bottom_flake_component="flake",
    )


def test_assemble_refuses_a_flake_wider_than_the_substrate_box(tmp_path):
    """A 12 A flake span in a 10 A periodic box must be refused, not wrapped.

    The message has to carry both footprints -- the user's next move is to pick
    a bigger [2D] x/y or a smaller edge_length, and neither number is guessable
    from "does not fit".
    """
    with pytest.raises(ValueError, match="does not fit") as exc:
        _assemble(tmp_path, flake_xs=[0.0, 12.0])

    message = str(exc.value)
    assert "12.0" in message          # the flake span
    assert "10.0" in message          # the box it has to fit inside
    assert "edge_length" in message


def test_assemble_refuses_a_flake_that_exactly_spans_the_box(tmp_path):
    """The comparison is `>=`, not `>`: a flake exactly as wide as the box has
    zero clearance to its own periodic image, so it is refused too."""
    with pytest.raises(ValueError, match="does not fit"):
        _assemble(tmp_path, flake_xs=[0.0, 10.0])


def test_assemble_accepts_a_flake_that_fits_and_reaches_the_lammps_call(tmp_path):
    """The guard must not over-fire. A narrow flake passes it and the method
    proceeds -- to `pm.calculate_gap`, which is `None` here, so an AttributeError
    is exactly the evidence that the footprint check let it through."""
    with pytest.raises(AttributeError):
        _assemble(tmp_path, flake_xs=[1.0, 3.0])


# --------------------------------------------------------------------------- #
# A [Flake-N] count other than 2 is refused, not silently dropped
# (builders-sheetonsheet.md B4-7).
#
# `_fold_flake_sheet_sections` caps nothing, and the hetero-flake branch is
# selected by `len(flake_sheets) == 2` -- so 1 or 3+ sections fell through to
# the single-material path and every [Flake-N] was ignored, while
# `material_token` named all of them in the output directory. The run built one
# thing and was filed under the name of another. `_get_hpc_job_name`'s docstring
# claim to "mirror that helper's rule exactly, so the two stay in lockstep" was
# false for every count but 2.
#
# `build_hetero_flake_component` DOES raise "exactly two materials ... got N" --
# the `== 2` test meant it was never reached.
# --------------------------------------------------------------------------- #

def _flake_n_raw(n: int):
    raw = _flake_slide_raw()
    for i in range(1, n + 1):
        raw[f"Flake-{i}"] = {
            "mat": "h-WS2" if i == 2 else "h-MoS2",
            "cif_path": f"{MAT}/h-WS2.cif" if i == 2 else f"{MAT}/h-MoS2.cif",
            "pot_path": f"{POT}/sw_lammps/t-WS2.sw" if i == 2 else f"{POT}/MoS2_wen.sw",
            "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1],
        }
    return raw


@pytest.mark.parametrize("n_sections", [1, 3])
def test_flake_slide_refuses_a_flake_section_count_other_than_two(n_sections, tmp_path):
    """1 or 3+ [Flake-N] sections must raise rather than build the wrong flake."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = SheetOnSheetSimulationConfig(**_flake_n_raw(n_sections), settings=_settings())
    assert len(cfg.flake_sheets) == n_sections      # the config layer caps nothing
    sim = FlakeSlideSimulation(cfg, str(tmp_path))

    with pytest.raises(ValueError, match="exactly two materials") as exc:
        sim.build()
    assert f"{n_sections} [Flake-N]" in str(exc.value)


@pytest.mark.parametrize("n_sections", [1, 3])
def test_the_refused_counts_are_the_ones_the_directory_name_disagreed_with(n_sections):
    """The defect stated as one assertion, independent of the builder.

    `material_token` names every [Flake-N] material; the build ignored them all
    for any count but 2. This pins WHY those counts are refused -- if a future
    change made the builder honour them, this test says the directory name was
    already right and only the builder was wrong.
    """
    from frictionsim2d.run.naming import material_token

    named = material_token(_flake_n_raw(n_sections), "sheetonsheet")
    plain = material_token(_flake_slide_raw(), "sheetonsheet")

    assert named != plain                       # the directory reflects [Flake-N]
    assert named.count("+") >= n_sections       # ...every one of them


def test_flake_slide_still_accepts_the_two_section_hetero_flake(tmp_path):
    """The guard must not over-fire on the shape it exists to protect.

    Stops at the guard rather than running the whole build (which needs LAMMPS):
    reaching the substrate-layer check below it proves the flake-count check let
    a 2-section config through.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = SheetOnSheetSimulationConfig(**_flake_n_raw(2), settings=_settings())
    cfg.sheet.layers = [1]
    cfg.sheets[0].layers = [1]
    sim = FlakeSlideSimulation(cfg, str(tmp_path))

    with pytest.raises(ValueError, match=r"[Aa]t least 2"):   # NOT the flake-count one
        sim.build()


def test_flake_slide_still_accepts_a_single_material_flake(tmp_path):
    """A plain [Flake] with no [Flake-N] at all -- the common case -- is
    unaffected: flake_sheets is empty, which the guard allows."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake_sheets == []
    cfg.sheet.layers = [1]
    cfg.sheets[0].layers = [1]
    sim = FlakeSlideSimulation(cfg, str(tmp_path))

    with pytest.raises(ValueError, match=r"[Aa]t least 2"):
        sim.build()


# --------------------------------------------------------------------------- #
# Normal load on the flake (follow-up C, 2026-09-22): the flake slide applies
# the configured pressure over the flake's PROJECTED in-plane area and emits
# the felt normal force v_fz, so a flake run gets a COF defined like
# sheet-on-sheet (lateral force / normal load).
# --------------------------------------------------------------------------- #

def _carved_area(shape: str, edge_length: float, rotation_deg: float,
                 step: float = 0.02) -> float:
    """Numerically integrate the builder's OWN carving predicate on a grid.

    Independent of the closed-form formula under test: it counts grid cells
    inside `components._in_*`, the exact test `build_flake` uses to keep atoms.
    """
    import numpy as np
    from frictionsim2d.builders.construction import components as comp

    inside = {
        "triangle": lambda x, y: comp._in_equilateral_triangle(
            x, y, edge_length, (0.0, 0.0), rotation_deg),
        "square": lambda x, y: comp._in_square(x, y, edge_length, (0.0, 0.0), rotation_deg),
        "hexagon": lambda x, y: comp._in_hexagon(x, y, edge_length, (0.0, 0.0), rotation_deg),
    }[shape]
    half = 1.1 * edge_length
    grid = np.arange(-half, half, step) + step / 2.0
    count = sum(1 for x in grid for y in grid if inside(float(x), float(y)))
    return count * step * step


@pytest.mark.parametrize("shape, edge, expected", [
    ("triangle", 10.0, 3 ** 0.5 / 4.0 * 100.0),        # side a: sqrt(3)/4 a^2
    ("square", 10.0, 100.0),                            # side a: a^2
    ("hexagon", 10.0, 1.5 * 3 ** 0.5 * 100.0),         # circumradius R: 3 sqrt(3)/2 R^2
])
def test_flake_projected_area_closed_form_per_shape(shape, edge, expected):
    from frictionsim2d.builders.sheetonsheet.flake_slide import flake_projected_area

    assert flake_projected_area(shape, edge) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
@pytest.mark.parametrize("rotation_deg", [0.0, 17.0, 45.0])
def test_flake_projected_area_matches_the_carving_predicate(shape, rotation_deg):
    """The formula is the area of the outline `build_flake` actually cuts, for
    every supported shape and independent of in-plane rotation (rotation_deg
    and the D3b twist are rigid rotations: area-preserving)."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import flake_projected_area

    edge = 8.0
    assert _carved_area(shape, edge, rotation_deg) == pytest.approx(
        flake_projected_area(shape, edge), rel=5e-3)


def test_flake_projected_area_refuses_an_unknown_shape():
    from frictionsim2d.builders.sheetonsheet.flake_slide import flake_projected_area

    with pytest.raises(ValueError, match="circle"):
        flake_projected_area("circle", 5.0)


def test_flake_load_context_uses_the_flake_outline_not_the_box():
    """The render context carries the FLAKE's area (hexagon, R=5 A), not the
    substrate box's -- the load is a pressure on the flake contact."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    sim = FlakeSlideSimulation(cfg, "unused")
    ctx = sim._normal_load_context()
    assert ctx["flake_shape"] == "hexagon"
    assert ctx["flake_edge_length"] == pytest.approx(5.0)
    assert ctx["flake_area"] == pytest.approx(1.5 * 3 ** 0.5 * 25.0)


def _loaded_ctx(**overrides):
    ctx = _minimal_flake_slide_context()
    ctx.update({"pressures": [0.5, 1.0], "flake_area": 64.951905,
                "flake_shape": "hexagon", "flake_edge_length": 5.0})
    ctx.update(overrides)
    return ctx


def _first_line(script: str, pattern: str) -> int:
    for i, line in enumerate(script.splitlines()):
        if re.search(pattern, line):
            return i
    raise AssertionError(f"no line matches {pattern!r}")


def test_flake_slide_applies_the_pressure_over_the_flake_area():
    script = _render_flake_slide_for_test(_loaded_ctx())

    assert re.search(r"variable\s+A equal 64\.951905\b", script)
    assert re.search(r"variable\s+p equal v_pressure/160\.2176565", script)
    assert re.search(r"variable\s+f equal -v_p\*v_A/\(count\(flake\)\)", script)
    # the flake (the driven body) receives the load, as a per-atom addforce --
    # not aveforce, which would average away the deformable flake's forces
    assert re.search(r"fix\s+force flake addforce 0\.0 0\.0 \$f", script)
    assert not re.search(r"^fix\s+\S+\s+\S+\s+aveforce", script, re.M)
    # nN, not GPa: 1 eV/A = 1.602176565 nN (the GPa factor is 100x that)
    assert re.search(r"variable\s+fz equal f_force\[3\]\*1\.602176565\b", script)


def test_flake_slide_emits_the_normal_force_in_the_results_file():
    script = _render_flake_slide_for_test(_loaded_ctx())
    fc_win = [l for l in script.splitlines() if "fc_win" in l and "ave/time" in l]
    assert fc_win and " v_fz" in fc_win[0]


def test_flake_load_precedes_the_thermostat_and_the_equilibration_run():
    """Load then equilibrate then slide (the sheet ordering), and the load fix
    precedes every other force-modifying fix so f_force[3] -- the group force
    BEFORE the load is added -- is the sheet-flake interaction alone."""
    script = _render_flake_slide_for_test(_loaded_ctx(thermostat_type="langevin"))
    load = _first_line(script, r"^fix\s+force flake addforce")
    assert load < _first_line(script, r"^fix\s+lang center langevin")
    assert load < _first_line(script, r"^fix\s+corner_hold")
    assert load < _first_line(script, r"^run\s+10000")
    assert load < _first_line(script, r"^fix\s+spr ")


_DEF_PATTERNS = {
    "v": re.compile(r"^variable\s+(\w+)\s"),
    "c": re.compile(r"^compute\s+(\w+)\s"),
    "f": re.compile(r"^fix\s+(\w+)\s"),
}
_REF = re.compile(r"\b([vcf])_(\w+)")


def _undefined_references(script: str):
    """Every v_/c_/f_ reference that precedes its definition, plus every
    `unfix` naming no live fix. `clear` resets the fix/compute namespace (a
    loop iteration); index variables survive it, as in LAMMPS."""
    defined = {"v": set(), "c": set(), "f": set()}
    problems = []
    for n, raw in enumerate(script.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "clear":
            defined["c"].clear()
            defined["f"].clear()
            continue
        if line.startswith("unfix"):
            name = line.split()[1]
            if name not in defined["f"]:
                problems.append((n, f"unfix {name}"))
            defined["f"].discard(name)
            continue
        new = None
        for kind, pat in _DEF_PATTERNS.items():
            m = pat.match(line)
            if m:
                new = (kind, m.group(1))
        body = line
        if new is not None:
            body = line.split(None, 2)[2] if len(line.split(None, 2)) > 2 else ""
        for kind, name in _REF.findall(body):
            if name not in defined[kind]:
                problems.append((n, f"{kind}_{name}"))
        if new is not None:
            defined[new[0]].add(new[1])
    return problems


@pytest.mark.parametrize("drive_method", ["smd", "fix_move", "virtual_atom"])
@pytest.mark.parametrize("constraint_mode", ["none", "com_spring"])
@pytest.mark.parametrize("corner_hold", [True, False])
@pytest.mark.parametrize("thermostat", ["langevin", "nose-hoover"])
def test_loaded_flake_script_defines_everything_before_use(
        drive_method, constraint_mode, corner_hold, thermostat):
    script = _render_flake_slide_for_test(_loaded_ctx(
        drive_method=drive_method, constraint_mode=constraint_mode,
        corner_hold=corner_hold, thermostat_type=thermostat))
    assert re.search(r"fix\s+force flake addforce", script)
    assert _undefined_references(script) == []


def test_the_def_before_use_checker_catches_a_missing_definition():
    """The checker must be able to fail: drop the load fix and v_fz dangles."""
    script = _render_flake_slide_for_test(_loaded_ctx())
    broken = re.sub(r"^fix\s+force flake addforce.*$", "", script, flags=re.M)
    assert any("f_force" in p for _n, p in _undefined_references(broken))


@pytest.mark.parametrize("constraint_mode", ["none", "com_spring"])
def test_no_flake_constraint_pins_z_against_the_load(constraint_mode):
    """Every flake constraint leaves z free, so the applied load is carried by
    the contact: com_spring couples x/y only (z NULL), corner-hold is
    spring/self xy, and every drive tethers x/y only."""
    for drive in ("smd", "fix_move", "virtual_atom"):
        script = _render_flake_slide_for_test(_loaded_ctx(
            constraint_mode=constraint_mode, drive_method=drive, corner_hold=True))
        for line in script.splitlines():
            if re.match(r"fix\s+com_spring ", line):
                assert line.split()[-2] == "NULL", line
            if re.match(r"fix\s+corner_hold ", line):
                assert line.split()[-1] == "xy", line
            if re.match(r"fix\s+spr ", line):
                assert "NULL" in line.split(), line


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_flake_slide_under_load_reports_the_applied_normal_force(tmp_path):
    """LAMMPS confirmation of the load block (lammps-marked; not in the fast
    suite). With the flake's COM z-acceleration averaging to zero, the
    pre-load group force the fix records (f_force[3], the sheet pushing up on
    the flake) must average to +P*A_flake: in nN, P[GPa]*A[A^2]/100.

    **The premise is checked, not assumed.** `MoS2_wen.sw` does not hold the
    flake at its starting height (future-works, "MoS2_wen.sw does not hold its
    structure at temperature"): the flake sinks by ~4.5 A at a seed-dependent
    moment, and while it is accelerating downward the reaction force is
    legitimately below the load. Measured 2026-09-22 over four builds of this
    same config: a quiescent window gives 2.8-3.3 nN against the 3.25 nN
    expectation, a window straddling the sink gives 1.6-1.7 -- so averaging
    every record makes the assertion a coin flip on the seed. The run is
    therefore long enough to contain a settled stretch, and only the records
    where the COM is not moving in z are averaged.
    """
    from frictionsim2d.builders.sheetonsheet.flake_slide import (FlakeSlideSimulation,
                                                       flake_projected_area)

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.settings.simulation.drive_method = "virtual_atom"
    cfg.settings.simulation.slide_run_steps = 20000
    cfg.general.scan_speed = 100
    cfg.general.pressure = 5.0

    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()
    run_dir = sim.output_dir
    slide = next(Path(run_dir).rglob("*slide*.in"))
    proc, out = _run_lammps_slide(run_dir, slide.name)
    assert proc.returncode == 0, f"lmp_serial exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out

    friction = next(run_dir.rglob("friction_*"))
    cols, rows = _read_friction_records(friction)
    assert "v_fz" in cols and "v_comz_flake" in cols
    ifz, icz = cols.index("v_fz"), cols.index("v_comz_flake")

    #: A record counts as settled when the flake COM moved less than this in z
    #: over the preceding window. The sink moves it 0.25-1.14 A per window;
    #: quiescent windows sit at 0.01-0.12 A, so the two are well separated.
    settled_dz = 0.15
    settled = [rows[k][ifz] for k in range(1, len(rows))
               if abs(rows[k][icz] - rows[k - 1][icz]) < settled_dz]
    assert len(settled) >= 3, (
        f"the flake never settled in z: only {len(settled)} of {len(rows) - 1} "
        f"windows moved less than {settled_dz} A. Load balance is not defined "
        f"while it sinks -- see future-works on MoS2_wen.sw. "
        f"comz={[r[icz] for r in rows]}")

    expected = 5.0 * flake_projected_area("hexagon", 5.0) / 100.0   # nN
    mean_fz = sum(settled) / len(settled)
    assert mean_fz == pytest.approx(expected, rel=0.3), (settled, expected)


# --------------------------------------------------------------------------- #
# Interlayer styles (interlayer-potential-choice Task 4). No LAMMPS: the
# potential registration and the template are checked directly.
# --------------------------------------------------------------------------- #

def _flake_pm(tmp_path, monkeypatch, **general):
    """`_register_potentials` for the single-material flake config."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    cfg = _flake_on_sheet_cfg()
    for key, value in general.items():
        setattr(cfg.general, key, value)
    builder = FlakeSlideSimulation(cfg, output_dir=str(tmp_path / "out"))
    settings_path = tmp_path / "out" / "lammps" / "system.in.settings"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    pm = builder._register_potentials(
        "provenance/potentials", sheet_cfg=cfg.sheet, n_sheet_layers=2,
        flake_names=["flake"], flake_components=[builder._filled_flake_config()],
        reserve_virtual_type=True, settings_path=settings_path)
    return builder, pm, settings_path.read_text(encoding="utf-8")


def test_flake_lj_potentials_match_the_pre_interlayer_sequence(tmp_path, monkeypatch):
    """lj emits exactly the calls the inline `_register` made before."""
    from frictionsim2d.potentials.manager import PotentialManager
    builder, _, text = _flake_pm(tmp_path, monkeypatch)
    cfg = builder.config
    pm = PotentialManager(cfg.settings,
                          potentials_dir=tmp_path / "out" / "provenance" / "potentials",
                          potentials_prefix="provenance/potentials")
    pm.set_lj_overrides(cfg.lj_override)
    pm.register_component("sheet", cfg.sheet, n_layers=2)
    pm.register_component("flake", builder._filled_flake_config(),
                          regions=[None, "corner"])
    pm.register_virtual_atom()
    pm.add_self_interaction("sheet")
    pm.add_interlayer_interaction("sheet")
    pm.add_self_interaction("flake")
    pm.add_cross_interaction("sheet", "flake")
    expected = tmp_path / "expected.settings"
    pm.write_file(expected)
    assert text == expected.read_text(encoding="utf-8")


def test_flake_ilp_covers_sheet_and_flake_without_their_lj(tmp_path, monkeypatch):
    """ilp: one ilp/tmd pair_coeff over the sheet layers AND the flake (the
    sliding interface); the sheet-flake lj/cut is gone."""
    _, pm, text = _flake_pm(tmp_path, monkeypatch, interlayer_style="ilp")
    ilp = [ln for ln in text.splitlines() if ln.startswith("pair_coeff * * ilp/tmd")]
    assert len(ilp) == 1
    emap = ilp[0].split("#")[0].split()[5:]
    for t in pm.types.ids_by_component("sheet") + pm.types.ids_by_component("flake"):
        assert emap[t - 1] in ("Mo", "S")
    assert not [ln for ln in text.splitlines()
                if "lj/cut" in ln and "(L" in ln and "flake" in ln.lower()]
    assert pm.needs_molecule_ids


def test_flake_template_sets_molecule_ids_before_minimize():
    ctx = _minimal_flake_slide_context()
    plain = _render_flake_slide_for_test(dict(ctx))
    assert "property/atom" not in plain
    text = _render_flake_slide_for_test({
        **ctx, "needs_molecule_ids": True, "use_group_group": False,
        "interlayer_energy_substyle": None,
        "molecule_groups": ["layer_1", "layer_2", "flake"]})
    lines = text.splitlines()
    first_min = next(i for i, ln in enumerate(lines) if ln.startswith("minimize"))
    head = lines[:first_min]
    flake_group = head.index("group           flake type 5 6")
    fix = head.index("fix             mol_ids all property/atom mol ghost yes")
    assert flake_group < fix
    assert head[fix + 1:fix + 4] == [
        "set group layer_1 mol 1", "set group layer_2 mol 2", "set group flake mol 3"]
    assert "variable        xfrict equal v_sx" in text


def test_molecule_groups_for_assembly_and_hetero_flake_carry_types(tmp_path, monkeypatch):
    """The assembly run has no layer/flake groups, so its entries carry type ids."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    _, pm, _ = _flake_pm(tmp_path, monkeypatch, interlayer_style="ilp")
    groups = FlakeSlideSimulation._molecule_groups(pm, 2, ["flake"], in_slide=False)
    assert [g["name"] for g in groups] == ["mol_layer_1", "mol_layer_2", "mol_flake"]
    assert groups[0]["types"] == pm.types.get_layer_group_string("sheet", 0)
    assert groups[2]["types"] == " ".join(map(str, pm.types.ids_by_component("flake")))


_NO_LAMMPS_BINARY = shutil.which("lmp_serial") is None


@pytest.mark.skipif(_NO_LAMMPS_BINARY, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
@pytest.mark.parametrize("hetero_flake", [False, True], ids=["homogeneous", "hetero"])
def test_flake_slide_ilp_builds_with_lj_assembly_settings(tmp_path, hetero_flake):
    """ilp flake slides build end to end: the assembly run 0 includes a
    settings file with the ilp type layout but no ilp/tmd (lmp_serial need
    not have INTERLAYER); the slide's own settings carry the ILP term."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=hetero_flake)
    cfg.general.interlayer_style = "ilp"
    out = tmp_path / "run"
    FlakeSlideSimulation(cfg, str(out)).build()

    assembly_script = (out / "assemble_flake_slide.lmp").read_text()
    assembly_settings = out / "build" / "assembly.in.settings"
    assert f"include         {assembly_settings}" in assembly_script
    run_settings = (out / "lammps" / "system.in.settings").read_text()
    assembly_text = assembly_settings.read_text()
    assert "ilp/tmd" not in assembly_text and "overlay" not in assembly_text
    assert "pair_coeff * * ilp/tmd" in run_settings
    # identical type layout (split chalcogens) on both sides
    masses = lambda t: [ln for ln in t.splitlines() if ln.startswith("mass ")]
    assert masses(assembly_text) == masses(run_settings)
    slide = (out / "lammps" / "slide.in").read_text()
    assert "fix             mol_ids all property/atom mol ghost yes" in slide
