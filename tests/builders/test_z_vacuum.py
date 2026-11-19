"""Periodic-z vacuum: cutoff + skin + margin between the stack and its image.

Every run deck is ``boundary p p p``. Before this rule the sheet box was
``-15 .. zhi + 15`` with ``zhi`` the *monolayer* data-file top, so a 5-layer
h-MoS2 stack (atoms 0..31.6 A) had 9 A of vacuum under a 12 A LJ cutoff.
"""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.builders.sheetonsheet import SheetOnSheetSimulation
from src.builders.utils.z_vacuum import ZBox, data_atom_z_range
from src.core.config import GlobalSettings, SheetOnSheetSimulationConfig, read_config

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOX = re.compile(r'^region\s+box\s+block(?:\s+\S+){4}\s+(\S+)\s+(\S+)', re.M)


def _zbox(**kw):
    base = dict(zmin=0.0, zmax=30.0, cutoff=12.0, skin=2.0, margin=8.0)
    base.update(kw)
    return ZBox(**base)


# --- the formula and the deck check ---------------------------------------------

def test_box_is_span_plus_cutoff_skin_margin():
    box = _zbox()
    assert box.vacuum == pytest.approx(22.0)
    assert (box.zlo, box.zhi) == (pytest.approx(-11.0), pytest.approx(41.0))


def test_top_travel_and_clearance_only_add_room():
    assert _zbox(top_travel=20.0).zhi == pytest.approx(61.0)
    assert _zbox(top_clearance=15.0).zhi == pytest.approx(45.0)
    assert _zbox(top_clearance=5.0).zhi == pytest.approx(41.0)   # below V/2: no effect


def test_check_deck_refuses_the_old_sheet_box():
    """The pre-fix 5-layer box: -15 .. 25.63 around atoms 0 .. 31.63."""
    box = _zbox(zmax=31.63, cutoff=12.0, skin=0.3)
    deck = "region          box block 0 10 0 10 -15.0 25.63 units box\n"
    with pytest.raises(ValueError, match="does not contain"):
        box.check_deck(deck)
    deck = "region          box block 0 10 0 10 -0.5 40.13 units box\n"   # same period
    with pytest.raises(ValueError, match=r"vacuum 9\.00 A is below"):
        box.check_deck(deck)


def test_box_from_zhi_passes_when_the_clearance_wins():
    """virtual_atom slides: clearance 11 > V/2 = 10.15 puts zhi exactly on
    zmax + clearance. The box the ZBox itself makes must pass its own check."""
    box = _zbox(zmax=31.303, cutoff=12.0, skin=0.3, top_clearance=11.0)
    assert box.zhi == pytest.approx(31.303 + 11.0)
    box.check(box.zlo, box.zhi)
    with pytest.raises(ValueError, match="does not contain"):
        box.check(box.zlo, box.zhi - 0.01)


def test_check_deck_reads_the_last_change_box():
    box = _zbox()
    deck = ("region          box block 0 1 0 1 -1 31 units box\n"
            "change_box      all z final -11.000000 41.000000 units box\n")
    box.check_deck(deck)
    box.check_deck("read_data       ${load_file}\n")   # no box: inherits it


def test_margin_is_validated():
    with pytest.raises(ValidationError):
        GlobalSettings(geometry={'z_vacuum_margin': -1.0})
    assert GlobalSettings().geometry.z_vacuum_margin == 8.0


# --- real sheet builds ------------------------------------------------------------

def _build(tmp_path, monkeypatch, *, mat, cif, pot, layers, lat_c, style='lj',
           xy=12.0, lj_cutoff=12.0):
    """Build a sheet-on-sheet run (given lat_c: no minimisation) and return
    ``(slide.in text, atom z-range of its data file, settings)``."""
    monkeypatch.chdir(REPO_ROOT)
    raw = read_config(REPO_ROOT / "examples" / "sheet_config.ini")
    raw["general"].update({"interlayer_style": style, "pressure": 1.0})
    raw["2D"].update({"mat": mat, "cif_path": cif, "pot_path": pot,
                      "layers": [layers], "x": xy, "y": xy, "lat_c": lat_c})
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    settings.potential.lj_cutoff = lj_cutoff
    settings.simulation.neighbor_list = 2.0
    config = SheetOnSheetSimulationConfig(**raw, settings=settings)
    out = tmp_path / "out"
    builder = SheetOnSheetSimulation(config, output_dir=str(out))
    builder.build()
    slide = (out / "lammps" / "slide.in").read_text(encoding="utf-8")
    return slide, data_atom_z_range(builder.structure_paths['sheet']), settings


def _vacuum(slide, zrange):
    zlo, zhi = map(float, _BOX.search(slide).groups())
    assert zlo < zrange[0] and zhi > zrange[1]
    return (zhi - zlo) - (zrange[1] - zrange[0])


def test_five_layer_mos2_sevik_lj12(tmp_path, monkeypatch):
    slide, zr, settings = _build(
        tmp_path, monkeypatch, mat="h-MoS2", cif="examples/materials/h-MoS2.cif",
        pot="examples/potentials/sw/MoS2_sevik.sw", layers=5, lat_c=6.15)
    assert zr[1] - zr[0] > 27.0                     # five layers, not one
    assert _vacuum(slide, zr) >= 12.0 + 2.0 + settings.geometry.z_vacuum_margin - 1e-6


def test_ilp_gets_the_larger_cutoff(tmp_path, monkeypatch):
    slide, zr, settings = _build(
        tmp_path, monkeypatch, mat="h-MoS2", cif="examples/materials/h-MoS2.cif",
        pot="examples/potentials/sw/MoS2_wen.sw", layers=3, lat_c=6.15,
        style="ilp", xy=20.0)
    assert _vacuum(slide, zr) >= 16.0 + 2.0 + settings.geometry.z_vacuum_margin - 1e-6


def test_thick_monolayer_inte_passes(tmp_path, monkeypatch):
    slide, zr, settings = _build(
        tmp_path, monkeypatch, mat="InTe", cif="examples/materials/InTe.cif",
        pot="examples/potentials/sw/sw_lammps/InTe.sw", layers=4, lat_c=8.5,
        xy=16.0)
    assert _vacuum(slide, zr) >= 12.0 + 2.0 + settings.geometry.z_vacuum_margin - 1e-6


def test_internal_lj_potentials_report_their_own_range():
    from src.potentials.utils import internal_pair_cutoff
    pots = REPO_ROOT / "examples" / "potentials"
    assert internal_pair_cutoff("rebomos", pots / "rebo" / "MoS.rebomos", 11.0) == \
        pytest.approx(2.5 * 4.2)                          # 2.5 sigma_MM
    assert internal_pair_cutoff(
        "reaxff", pots / "reaxff" / "ffield_ponomarev.reax.MoS", 11.0) == 10.0
    assert internal_pair_cutoff("sw", pots / "sw" / "MoS2_wen.sw", 11.0) == 0.0

