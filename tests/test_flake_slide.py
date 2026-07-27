"""Tests for D4 flake-driven slide (flake on a substrate sheet, no tip).

The flake (single-material or a hetero bilayer) is driven across a substrate
sheet by a virtual-atom / COM spring. The substrate sheet's bottom band is
fixed; its interior + the flake are thermostatted; lateral friction is read
from the bottom-sheet reaction force (the many-body ``sw`` observable).
"""
import re
import shutil
from pathlib import Path

import pytest
from src.core.config import SheetOnSheetSimulationConfig, load_settings
# Reuse the D1 slide runner + friction-record parser for the e2e smoke.
from tests.test_hetero_slide import _read_friction_records, _run_lammps_slide

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _minimal_flake_slide_context():
    """Fill every key ``flake/slide.lmp`` references.

    Mirrors ``tests/test_hetero_slide.py::_minimal_slide_context`` (the D1 slide
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
        "ev_a_to_nn": 160.2176565,
        "ev_a3_to_gpa": 160.2176565,
        "sheet_layers": sheet_layers,
        "flake_group_types": "5 6",
        "flake_corner_types": "6",
        "flake_corner_k": 10.0,
        "corner_hold": True,
        "virtual_offset": 10.0,
    }


def test_flake_slide_drives_flake_group_not_layer():
    from src.builders.flake_slide import render_flake_slide

    ctx = _minimal_flake_slide_context()
    ctx["drive_method"] = "virtual_atom"
    ctx["corner_hold"] = True
    script = render_flake_slide(ctx)

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
    assert "aveforce" not in script        # no normal load on the sheet top
    # No undefined layer_0 (a 2-layer sheet must never reference n_layers-2).
    assert "layer_0" not in script


def test_flake_slide_defines_sheet_layer_groups_and_flake_group():
    from src.builders.flake_slide import render_flake_slide

    ctx = _minimal_flake_slide_context()
    script = render_flake_slide(ctx)

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
    from src.builders.flake_slide import render_flake_slide

    ctx = _minimal_flake_slide_context()
    ctx["corner_hold"] = False
    script = render_flake_slide(ctx)
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
    raw = {
        "general": {"temp": 300, "scan_speed": 100},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 16.0, "y": 16.0, "layers": [2]},
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
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_builder_places_flake_and_reserves_virtual(tmp_path):
    from src.builders.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.settings.simulation.drive_method = "virtual_atom"
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    sim.build()

    slide = next(Path(tmp_path).rglob("*slide*.in"))
    text = slide.read_text()
    assert "spring couple virtual" in text
    assert re.search(r"fix\s+spr flake spring couple virtual", text)   # driven group is the flake
    assert "fsbot" in text and "setforce 0.0 0.0 0.0" in text          # sheet bottom fixed
    assert "corner_hold" in text                                       # default ON


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_driven_group_is_flake_and_virtual_type_reserved(tmp_path):
    from src.builders.flake_slide import FlakeSlideSimulation

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
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def test_routing_flake_section_selects_flake_slide():
    from src.core.run import _select_sheet_builder_cls
    from src.builders.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake is not None
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is FlakeSlideSimulation


def test_routing_no_flake_unchanged():
    from src.core.run import _select_sheet_builder_cls
    from src.builders.sheetonsheet import SheetOnSheetSimulation

    cfg = _single_material_cfg()
    assert cfg.flake is None
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is SheetOnSheetSimulation


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_flake_slide_builder_requires_multilayer_substrate(tmp_path):
    from src.builders.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    cfg.sheet.layers = [1]          # single-layer substrate -> no fixed bottom + interior
    cfg.sheets[0].layers = [1]
    sim = FlakeSlideSimulation(cfg, str(tmp_path))
    with pytest.raises(ValueError, match=r"[Aa]t least 2"):
        sim.build()


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
    no lost atoms and complete pair coeffs.
    """
    from src.builders.flake_slide import FlakeSlideSimulation

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
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

    run_dir = sim.run_dir()
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
