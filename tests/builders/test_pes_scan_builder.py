"""Tests for the static PES-scan builders (src/builders/pes_scan.py)."""

import logging
import math
from pathlib import Path

import pytest

from src.builders.afm.pes_tip import PESTipSimulation
from src.builders.utils.pes_common import _cell_period
from src.builders.sheetonsheet.pes_sheet import PESSheetSimulation, _tile_count
from src.core.config import (
    AFMSimulationConfig,
    FlakeConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
)


def _sheet_cfg(tmp_path: Path, layers, grid_n=12, z_relax=True):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return SheetOnSheetSimulationConfig(**{
        "general": {"temp": 0.0},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 30.0, "y": 30.0, "layers": layers},
        "pes": {"grid_n": grid_n, "z_relax": z_relax},
        "settings": load_settings().model_dump(),
    })


def _afm_cfg(tmp_path: Path, force):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return AFMSimulationConfig(**{
        "general": {"temp": 300.0, "force": force, "scan_speed": 2.0, "finite_sheet": False},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 60.0, "y": 60.0, "layers": [1]},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "sub": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "thickness": 12.0, "amorph": "a"},
        "pes": {"grid_n": 16},
        "settings": load_settings().model_dump(),
    })


# --- helpers ---------------------------------------------------------------

def test_cell_period_and_tile_count():
    uc = {"xlo": 0.0, "xhi": 3.0, "ylo": 0.0, "yhi": 5.0}
    assert _cell_period(uc, "x") == pytest.approx(3.0)
    assert _cell_period(uc, "y") == pytest.approx(5.0)
    full = {"xlo": 0.0, "xhi": 30.0, "ylo": 0.0, "yhi": 25.0}
    assert _tile_count(full, uc) == 10 * 5


def test_cell_period_missing_raises():
    with pytest.raises(ValueError, match="unit-cell"):
        _cell_period({}, "x")


def test_cell_period_subtracts_lo_not_just_hi():
    # A zero-origin cell (xlo/ylo == 0.0) can't distinguish `hi - lo` from a
    # buggy `hi`-only implementation, since they'd return the same number --
    # the flaw in an earlier version of this test. Both bounds are non-zero
    # here so the subtraction is actually exercised. The period values
    # themselves are h-MoS2's own lattice constant (examples/materials/
    # h-MoS2.cif, a=3.1840664646845926) orthogonalized the way this
    # codebase's own hex-to-orthogonal conversions scale y by sqrt(3)
    # (src/builders/construction/components.py:156,220,225-226) -- real
    # numbers, but the origin offset, not the numbers' realism, is what
    # makes the assertion discriminate a correct implementation from a
    # wrong one.
    a = 3.1840664646845926
    unit_cell = {"xlo": -a / 2, "xhi": a / 2,
                 "ylo": 2.5, "yhi": 2.5 + a * math.sqrt(3)}
    assert _cell_period(unit_cell, "x") == pytest.approx(a)
    assert _cell_period(unit_cell, "y") == pytest.approx(a * math.sqrt(3))


def test_cell_period_rejects_a_non_positive_period():
    unit_cell = {"xlo": 5.0, "xhi": 5.0}
    with pytest.raises(ValueError, match="Non-positive"):
        _cell_period(unit_cell, "x")


# --- PESSheetSimulation ----------------------------------------------------

def test_pes_sheet_min_layers_is_two():
    assert PESSheetSimulation.MIN_LAYERS == 2


def test_pes_sheet_rejects_non_bilayer(tmp_path: Path):
    builder = PESSheetSimulation(_sheet_cfg(tmp_path, layers=[3]), output_dir=str(tmp_path / "o"))
    with pytest.raises(ValueError, match="exactly 2 layers"):
        builder.build()


def test_pes_sheet_no_virtual_drive_atom(tmp_path: Path):
    """The static scan must not register the virtual driving atom."""
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.settings.simulation.drive_method = "virtual_atom"
    builder = PESSheetSimulation(cfg, output_dir=str(tmp_path / "o"))

    calls = {"virtual": 0}

    class _PM:
        def register_virtual_atom(self):
            calls["virtual"] += 1

    builder._register_drive_atom(_PM())
    assert calls["virtual"] == 0


def test_pes_sheet_build_forces_none_constraint(tmp_path: Path, monkeypatch):
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.settings.simulation.constraint_mode = "atom_bonds"
    builder = PESSheetSimulation(cfg, output_dir=str(tmp_path / "o"))

    seen = {}

    def fake_super_build(self):
        seen["constraint_mode"] = self.config.settings.simulation.constraint_mode

    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation.build", fake_super_build)
    builder.build()
    assert seen["constraint_mode"] == "none"


def test_pes_sheet_write_inputs_renders_scan(tmp_path: Path):
    """write_inputs should emit a slide.in PES scan (+ refine) with grid context."""
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.pes.grid_n_refine = 20
    out = tmp_path / "o"
    builder = PESSheetSimulation(cfg, output_dir=str(out))
    builder.relative_run_dir = Path("pes_sheet/h-MoS2/30x_30y/K0")

    # Stand in for the geometry/potential state build() would populate.
    class _PM:  # minimal PotentialManager stand-in (unused by write_inputs)
        pass
    builder.pm = _PM()
    builder.structure_paths = {"sheet": tmp_path / "h-MoS2_2.lmp"}
    builder.sheet_dims = {"xlo": 0.0, "xhi": 30.0, "ylo": 0.0, "yhi": 30.0, "zhi": 20.0}
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.0, "ylo": 0.0, "yhi": 5.0}
    builder.lat_c = 6.5

    builder.write_inputs()

    slide = (out / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "interlayer PES scan" in slide
    assert "displace_atoms  layer_2 move" in slide
    assert "setforce 0.0 0.0 0.0" in slide            # frozen bottom
    assert "Nx     equal 12" in slide
    assert "$(v_dx-v_prev_dx)" in slide                # snapshotted delta, not lazy
    assert (out / "lammps" / "slide_refine.in").exists()   # convergence grid


# --- PESTipSimulation ------------------------------------------------------

def test_pes_tip_uses_fixed_contact_gap():
    """Tip scan is a fixed firm-contact scan (no repulsive-load setpoint)."""
    from src.builders.afm.pes_tip import PES_TIP_CONTACT_GAP
    assert PES_TIP_CONTACT_GAP > 0


def test_pes_tip_z_relax_from_config(tmp_path: Path):
    cfg = _afm_cfg(tmp_path, force=[2.0])
    cfg.pes.z_relax = False
    builder = PESTipSimulation(cfg, output_dir=str(tmp_path / "o"))
    assert builder._z_relax() is False


def test_pes_tip_hpc_job_name(tmp_path: Path):
    builder = PESTipSimulation(_afm_cfg(tmp_path, force=[2.0]), output_dir=str(tmp_path / "o"))
    assert builder._get_hpc_job_name() == "pes_tip_h-MoS2"


# --- A [flake] on a PES-tip config is refused, not silently dropped ---------
#
# afm/pes_scan.lmp (and afm_freestanding/pes_scan.lmp) have only sub/tip/sheet
# read_data lines -- no flake block. The AFM build in front of them is the full
# one, so a [flake] config used to size create_box for the flake's atom types
# and write their masses/pair_coeff into system.in.settings while the flake's
# atoms were never read in: the scan succeeded and reported the corrugation of
# the BARE surface. PESTipSimulation.__init__ now refuses instead.


def _afm_cfg_with_flake(tmp_path: Path):
    cfg = _afm_cfg(tmp_path, force=[2.0])
    pot = tmp_path / "d.sw"
    cif = tmp_path / "d.cif"
    cfg.flake = FlakeConfig(shape="hexagon", edge_length=10.0, corner_radius=5.0,
                            mat="h-MoS2", cif_path=str(cif), pot_path=str(pot),
                            pot_type="sw")
    return cfg


def test_pes_tip_refuses_a_flake_config(tmp_path: Path):
    """The refusal fires at construction -- before any directory, structure or
    rendered script exists -- and names the section, the builder and why."""
    with pytest.raises(ValueError) as exc:
        PESTipSimulation(_afm_cfg_with_flake(tmp_path), output_dir=str(tmp_path / "o"))

    message = str(exc.value)
    assert "[flake]" in message
    assert "PESTipSimulation" in message
    assert "afm/pes_scan.lmp" in message
    assert "bare surface" in message


def test_pes_tip_without_a_flake_still_renders_the_whole_scan(tmp_path: Path):
    """The refusal must not over-fire: a flakeless PES-tip config still builds
    its scan, with all three read_data lines and no flake anywhere in it.

    Rendered through the REAL afm/pes_scan.lmp off seeded builder state (the
    tests/helpers/afm.py pattern), so this fails if the guard were to reject a
    flakeless config or perturb the render path -- not merely if a constructor
    call raised.
    """
    from tests.helpers.afm import _seed_builder_state

    builder = PESTipSimulation(_afm_cfg(tmp_path, force=[2.0]),
                               output_dir=str(tmp_path / "o"))
    layer_dir = _seed_builder_state(builder, tmp_path)
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.16, "ylo": 0.0, "yhi": 5.47}
    builder.box_dims = {"xlo": 0.0, "xhi": 100.0, "ylo": 0.0, "yhi": 100.0}

    builder.write_inputs(1)

    scan = (layer_dir / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert scan.count("read_data") == 3          # sub + tip + sheet, as always
    assert "group sub" in scan and "group tip" in scan and "group 2D" in scan
    assert "flake" not in scan.lower()
    assert "PES" in scan


# --- _warn_if_footprint_too_small (multi-cell scan periodic-image guard) ---
# Review builders-afm.md finding 1: no test ever set [pes] scan_cells > 1, so
# neither branch of this method's m > 1 body (the tip-image overlap check
# `_warn_if_footprint_too_small` guards) was ever exercised. No real build is
# needed -- box_dims and config.tip.r are plain attributes -- so a bare
# PESTipSimulation with them set directly is enough to call the method
# directly, mirroring tests/helpers/afm.py's seed-state pattern.

def _pes_tip_builder(tmp_path: Path, tip_r: float = 2.0) -> PESTipSimulation:
    cfg = _afm_cfg(tmp_path, force=[2.0])
    cfg.pes.scan_cells = 2
    cfg.tip.r = tip_r
    return PESTipSimulation(cfg, output_dir=str(tmp_path / "o"))


def test_warn_if_footprint_too_small_noop_when_m_is_one(tmp_path, caplog):
    """The m <= 1 (default scan_cells) guard must short-circuit before any
    box math, even for a box far too small to ever pass the check."""
    builder = _pes_tip_builder(tmp_path)
    builder.box_dims = {"xlo": 0.0, "xhi": 1.0, "ylo": 0.0, "yhi": 1.0}
    with caplog.at_level(logging.WARNING):
        builder._warn_if_footprint_too_small(1, 5.0, 5.0)
    assert not caplog.records


def test_warn_if_footprint_too_small_warns_on_undersized_box(tmp_path, caplog):
    # r=2.0, m=2, period=5.0: need_x = 2*2 + 2*5 + 12 = 26; a 20x20 box is
    # short of that on both axes -> the tip could reach its periodic image.
    builder = _pes_tip_builder(tmp_path, tip_r=2.0)
    builder.box_dims = {"xlo": 0.0, "xhi": 20.0, "ylo": 0.0, "yhi": 20.0}
    with caplog.at_level(logging.WARNING):
        builder._warn_if_footprint_too_small(2, 5.0, 5.0)
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "may overlap the tip's periodic image" in msg
    assert "scan_cells=2" in msg


def test_warn_if_footprint_too_small_silent_on_large_enough_box(tmp_path, caplog):
    # Same geometry as above but a box comfortably past the 26x26 need.
    builder = _pes_tip_builder(tmp_path, tip_r=2.0)
    builder.box_dims = {"xlo": 0.0, "xhi": 200.0, "ylo": 0.0, "yhi": 200.0}
    with caplog.at_level(logging.WARNING):
        builder._warn_if_footprint_too_small(2, 5.0, 5.0)
    assert not caplog.records


def test_warn_if_footprint_too_small_swallows_missing_box_dims(tmp_path, caplog):
    # box_dims never populated (e.g. called before _get_substrate_box_dims) --
    # the try/except (AttributeError, KeyError, TypeError) guard must return
    # quietly rather than raising or warning off a KeyError.
    builder = _pes_tip_builder(tmp_path)
    builder.box_dims = {}
    with caplog.at_level(logging.WARNING):
        builder._warn_if_footprint_too_small(2, 5.0, 5.0)
    assert not caplog.records


def test_pes_tip_ilp_bilayer_sets_molecule_ids(tmp_path, monkeypatch):
    """The tip PES on a substrate: ILP between the sheet layers, molecule IDs in the scan."""
    import re
    from tests.helpers.afm import build_example_afm
    out = build_example_afm(
        tmp_path, monkeypatch, "pes_tip_config.ini", PESTipSimulation,
        general={"interlayer_style": "ilp"}, **{"2D": {"layers": [2], "lat_c": 6.15}})
    settings = (out / "L2" / "lammps" / "system.in.settings").read_text(encoding="utf-8")
    assert "pair_coeff * * ilp/tmd" in settings
    slide = (out / "L2" / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert re.findall(r"^set group (\S+) mol (\d+)$", slide, re.M) == [
        ("layer_1", "1"), ("layer_2", "2")]


# --- interlayer styles (spec 2026-11-11 §7, Task 7) -------------------------

def _build_example_pes_sheet(tmp_path, monkeypatch, **general) -> Path:
    """Build ``examples/pes_sheet_config.ini`` small (lat_c given).

    12 x 12, or 20 x 20 under ilp/kc: the build refuses a cell narrower than
    their 16 A cutoff + skin (tests/builders/test_interlayer_box_guard.py).
    """
    from src.core.config import GlobalSettings, read_config
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    raw = read_config(repo / "examples" / "pes_sheet_config.ini")
    raw["general"].update(general)
    size = 20 if general.get("interlayer_style") in ("ilp", "kc") else 12
    raw["2D"].update({"x": size, "y": size, "lat_c": 6.15})
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    config = SheetOnSheetSimulationConfig(**raw, settings=settings)
    out = tmp_path / "out"
    PESSheetSimulation(config, output_dir=str(out)).build()
    return out


def _first_run(lines):
    return next(i for i, ln in enumerate(lines)
                if ln.split()[:1] in (["run"], ["minimize"]))


def test_pes_sheet_ilp_energy_channel(tmp_path, monkeypatch):
    """ilp: molecule IDs before the first minimize, ILP energy as an extra column."""
    out = _build_example_pes_sheet(tmp_path, monkeypatch, interlayer_style="ilp")
    assert "pair_coeff * * ilp/tmd" in (
        out / "lammps" / "system.in.settings").read_text(encoding="utf-8")
    for name in ("slide.in", "slide_refine.in"):
        lines = (out / "lammps" / name).read_text(encoding="utf-8").splitlines()
        head = lines[:_first_run(lines)]
        include = next(i for i, ln in enumerate(lines) if ln.startswith("include"))
        fix = lines.index("fix             mol_ids all property/atom mol ghost yes")
        assert include < fix < _first_run(lines), name
        assert lines[fix + 1:fix + 3] == ["set group layer_1 mol 1",
                                          "set group layer_2 mol 2"], name
        assert "compute         e_il all pair ilp/tmd" in head, name
        assert "variable        e_int_il_all equal c_e_il" in head, name
        assert "variable        e_il_cell equal v_e_int_il_all/v_ncell" in head, name
        # c_e_il must be current when the print evaluates it between runs.
        assert "thermo_style    custom step pe c_e_il" in head, name
        text = "\n".join(lines)
        assert '"dx,dy,energy_eV,e_interlayer_eV"' in text, name
        assert '"$(v_dx),$(v_dy),$(v_pe_cell),$(v_e_il_cell)"' in text, name


def test_pes_sheet_lj_scan_has_no_interlayer_keys(tmp_path, monkeypatch):
    """lj: no molecule IDs, no energy compute, the three-column CSV of before."""
    out = _build_example_pes_sheet(tmp_path, monkeypatch)
    slide = (out / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "mol_ids" not in slide
    assert "e_il" not in slide
    assert "thermo_style" not in slide
    assert '"dx,dy,energy_eV"' in slide
