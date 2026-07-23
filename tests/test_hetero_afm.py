"""Tests for D2 periodic hetero AFM tip-slide assembly.

Covers config folding ([2D-1]/[2D-2] -> sheets), the in-plane replication
helpers, the HeteroAFMSimulation structure build + orthogonality guard, the
sheet_1/sheet_2 potential split + 2D union groups, render-context wiring,
routing, and a short real-LAMMPS end-to-end smoke.
"""
import math
import shutil
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
