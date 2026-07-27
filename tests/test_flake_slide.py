"""Tests for D4 flake-driven slide (flake on a substrate sheet, no tip).

The flake (single-material or a hetero bilayer) is driven across a substrate
sheet by a virtual-atom / COM spring. The substrate sheet's bottom band is
fixed; its interior + the flake are thermostatted; lateral friction is read
from the bottom-sheet reaction force (the many-body ``sw`` observable).
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from src.core.config import SheetOnSheetSimulationConfig, load_settings

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
