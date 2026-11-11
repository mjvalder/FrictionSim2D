"""The interlayer styles against the installed LAMMPS (spec 2026-11-11 §4, §5, §7).

What only a real LAMMPS can tell:

* LJ-G: the generated ``kolmogorov/crespi/z`` files, with the +g/2 all-layer
  instance and one -g/2 instance per layer (``pair_coeff * * kolmogorov/
  crespi/z N file <NULL map>``), give exactly g*exp(-rho^2) between layers,
  independent of z, and nothing between two atoms of one layer.
* LJ-G: whether ``compute group/group`` survives the KC/z overlay. It does
  not (KC/z has no ``single()``), which is why ``lj_g`` takes the
  reaction-force branch (``GROUP_GROUP_STYLES`` in
  src/builders/utils/interlayer.py).
* ILP: an SW + ilp/tmd MoS2 bilayer relaxes to the 2H interlayer spacing.
* ILP: a built PES-sheet scan with ``z_relax`` (minimize under ``compute
  pair ilp/tmd`` in the thermo) runs end to end.

The two-atom and spacing checks run in-process through the ``lammps`` Python
module; the PES smoke runs the ``lmp`` binary (the SERIAL build in this env --
``lmp_serial`` is an MPI build without INTERLAYER). Every LAMMPS test carries
an ``lmp_serial``-worded skipif, from which ``tests/conftest.py`` derives the
``lammps`` marker. ``test_example_interlayer_builds`` is fast and unmarked.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.config import GlobalSettings, SheetConfig
from src.potentials.manager import PotentialManager
from tests.helpers.examples import REPO_ROOT

pytestmark = pytest.mark.integration

MAT = REPO_ROOT / "examples" / "materials"
SW = REPO_ROOT / "examples" / "potentials" / "sw"
G_MOS2_MEV = 175.68


def _module_has_interlayer() -> bool:
    try:
        from lammps import lammps  # pylint: disable=import-outside-toplevel
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        try:
            return bool(lmp.has_style("pair", "ilp/tmd")
                        and lmp.has_style("pair", "kolmogorov/crespi/z"))
        finally:
            lmp.close()
    except Exception:  # pylint: disable=broad-except
        return False


def _binary_has_interlayer() -> bool:
    if not shutil.which("lmp"):
        return False
    proc = subprocess.run(["lmp", "-h"], capture_output=True, text=True,
                          timeout=60, check=False)
    return "ilp/tmd" in proc.stdout


NO_MODULE = not _module_has_interlayer()
NO_BINARY = not _binary_has_interlayer()
needs_module = pytest.mark.skipif(
    NO_MODULE, reason="lammps Python module without INTERLAYER "
                      "(lmp_serial-guard convention: marks the test lammps)")
needs_binary = pytest.mark.skipif(
    NO_BINARY, reason="serial lmp with INTERLAYER not on PATH "
                      "(lmp_serial-guard convention: marks the test lammps)")


def _sheet(pot: str = "MoS2_sevik.sw", size: float = 20.0) -> SheetConfig:
    return SheetConfig(mat="h-MoS2", pot_type="sw", pot_path=str(SW / pot),
                       cif_path=str(MAT / "h-MoS2.cif"), x=size, y=size,
                       layers=[2], lat_c=6.15)


def _bilayer_settings(tmp_path: Path, style: str, **interlayer_kw):
    """Sheet-only 2-layer settings file, written the way a build writes it."""
    pm = PotentialManager(GlobalSettings(), use_langevin=False,
                          potentials_dir=tmp_path / "potentials",
                          potentials_prefix=str(tmp_path / "potentials"))
    pm.set_interlayer(style, **interlayer_kw)
    pm.register_component("sheet", _sheet(), n_layers=2)
    pm.add_self_interaction("sheet")
    pm.add_interlayer(["sheet"])
    path = tmp_path / "system.in.settings"
    pm.write_file(path)
    layers = [[int(t) for t in pm.types.get_layer_group_string("sheet", i).split()]
              for i in range(2)]
    return pm, path, layers


def _lammps():
    from lammps import lammps  # pylint: disable=import-outside-toplevel
    return lammps(cmdargs=["-log", "none", "-screen", "none"])


# --------------------------------------------------------------------------
# LJ-G (spec §4)
# --------------------------------------------------------------------------

@pytest.fixture(name="ljg_pair")
def _ljg_pair(tmp_path):
    """Two atoms, ``type_a`` at (20, 20, 15) and ``type_b`` placed by ``put``,
    under the LJ-G settings; ``gaussian()`` is the summed KC/z energy (eV)."""
    pm, settings, layers = _bilayer_settings(tmp_path, "lj_g",
                                             g_by_component={"sheet": G_MOS2_MEV})
    n_instances = sum(1 for ln in settings.read_text().splitlines()
                      if ln.startswith("pair_coeff * * kolmogorov/crespi/z"))
    assert n_instances == 3            # +g/2 over both layers, -g/2 per layer

    lmp = _lammps()

    def setup(type_a: int, type_b: int):
        for cmd in ("units metal", "atom_style atomic", "boundary p p p",
                    "region box block 0 40 0 40 0 40",
                    f"create_box {len(pm.types)} box",
                    f"create_atoms {type_a} single 20 20 15 units box",
                    f"create_atoms {type_b} single 20 20 18 units box",
                    f"include {settings}"):
            lmp.command(cmd)
        for i in range(1, n_instances + 1):
            lmp.command(f"compute g{i} all pair kolmogorov/crespi/z {i}")
        lmp.command("thermo_style custom step pe "
                    + " ".join(f"c_g{i}" for i in range(1, n_instances + 1)))

    def put(dx: float, dz: float) -> float:
        lmp.command(f"set atom 2 x {20 + dx} y 20 z {15 + dz}")
        lmp.command("run 0")
        return sum(lmp.extract_compute(f"g{i}", 0, 0)
                   for i in range(1, n_instances + 1))

    yield layers, setup, put
    lmp.close()


@needs_module
def test_ljg_two_atom_gaussian(ljg_pair):
    """Mo (layer 1) - Mo (layer 2): exactly g e^(-rho^2), z-independent."""
    layers, setup, put = ljg_pair
    setup(layers[0][0], layers[1][0])
    g_ev = G_MOS2_MEV * 1e-3
    worst = 0.0
    for dx in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        got = put(dx, 3.0)
        worst = max(worst, abs(got - g_ev * math.exp(-dx * dx)))
    assert worst < 1e-6, worst
    fixed = [put(0.7, dz) for dz in (2.5, 3.0, 3.5, 4.0, 4.5)]
    assert max(fixed) - min(fixed) < 1e-6, fixed
    assert fixed[0] == pytest.approx(g_ev * math.exp(-0.49), abs=1e-6)


@needs_module
def test_ljg_same_layer_pair_has_no_gaussian(ljg_pair):
    """Mo and S of ONE layer at rho = 0 (the eclipsed pair the -g/2 instance
    exists for): the +g and -g instances cancel exactly."""
    layers, setup, put = ljg_pair
    setup(layers[0][0], layers[0][1])
    assert abs(put(0.0, 1.6)) < 1e-9
    assert abs(put(1.84, 1.56)) < 1e-9


@needs_module
def test_ljg_group_group_runs(tmp_path):
    """``compute group/group`` under the KC/z overlay: LAMMPS refuses it.

    Pinned, so the day KC/z grows a ``single()`` this fails and ``lj_g`` can
    go back to group/group (``GROUP_GROUP_STYLES``). The plain ``lj`` control
    shows the abort is KC/z's, not the SW sub-styles'.
    """
    error = _group_group_attempt(tmp_path / "ljg", "lj_g",
                                 g_by_component={"sheet": G_MOS2_MEV})
    assert error is not None and "does not support single call" in error, error
    assert _group_group_attempt(tmp_path / "lj", "lj") is None


def _group_group_attempt(tmp_path: Path, style: str, **kw):
    """Run ``compute group/group layer_1 layer_2``; the LAMMPS error, or None."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pm, settings, layers = _bilayer_settings(tmp_path, style, **kw)
    lmp = _lammps()
    try:
        for cmd in ("units metal", "atom_style atomic", "boundary p p p",
                    "region box block 0 40 0 40 0 40",
                    f"create_box {len(pm.types)} box",
                    f"create_atoms {layers[0][0]} single 20 20 15 units box",
                    f"create_atoms {layers[1][0]} single 20.5 20 18 units box",
                    f"include {settings}",
                    "compute gg layer_1 group/group layer_2",
                    "thermo_style custom step pe c_gg",
                    "run 0"):
            lmp.command(cmd)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return str(exc)
    finally:
        lmp.close()


# --------------------------------------------------------------------------
# ILP (spec §5)
# --------------------------------------------------------------------------

def _mean_z_by_type(lmp) -> dict:
    natoms = lmp.get_natoms()
    types = lmp.numpy.extract_atom("type")[:natoms]
    z = lmp.numpy.extract_atom("x")[:natoms, 2]
    return {int(t): float(z[types == t].mean()) for t in set(types.tolist())}


@needs_module
def test_ilp_mos2_bilayer_spacing(tmp_path):
    """SW (Sevik) + ilp/tmd (TMD.ILP) 2H MoS2 bilayer, relaxed in-plane and
    out: the Mo-Mo interlayer spacing lands at the 2H value.

    Measured 6.21 A from 5.6, 6.15 and 6.8 A starts. That is 0.06 A above the
    bulk c/2 = 6.15 A: Sevik's SW makes the layer 0.13 A thicker (Mo-S height
    1.65 vs 1.59 A), which the ILP S-S gap (2.92 A) only partly takes back.
    The band is 6.10-6.25 A, not the 6.1-6.2 A first planned.
    """
    from types import SimpleNamespace
    from src.builders.construction import components
    _, settings, layers = _bilayer_settings(tmp_path, "ilp")
    general = SimpleNamespace(interlayer_style="ilp", interlayer_file=None,
                              interlayer_comparison=False)
    (tmp_path / "build").mkdir()
    data, _dims, _lat_c = components.build_sheet(
        _sheet(), tmp_path / "build", stack_if_multi=True,
        settings=GlobalSettings(), n_layers_override=2,
        interlayer_general=general)
    lmp = _lammps()
    try:
        for cmd in ("units metal", "atom_style atomic", "boundary p p p",
                    f"read_data {data}",
                    "fix mol_ids all property/atom mol ghost yes",
                    f"include {settings}",
                    "set group layer_1 mol 1", "set group layer_2 mol 2",
                    # Start away from 6.15 so the test sees the minimiser move.
                    "displace_atoms layer_2 move 0 0 0.5 units box",
                    "fix relax all box/relax x 0.0 y 0.0 couple xy vmax 0.001",
                    "min_style cg",
                    "minimize 0 1e-8 20000 200000"):
            lmp.command(cmd)
        z = _mean_z_by_type(lmp)
    finally:
        lmp.close()
    mo_1, mo_2 = layers[0][0], layers[1][0]
    spacing = z[mo_2] - z[mo_1]
    assert 6.10 <= spacing <= 6.25, spacing


# --------------------------------------------------------------------------
# PES sheet smoke (spec §7): minimize with compute pair ilp/tmd in the thermo
# --------------------------------------------------------------------------

@needs_binary
def test_pes_sheet_ilp_z_relax_runs(tmp_path, monkeypatch):
    """A built ilp PES-sheet scan with z_relax runs under the serial ``lmp``
    and writes one finite interlayer energy per grid point."""
    from src.builders.sheetonsheet.pes_sheet import PESSheetSimulation
    from src.core.config import SheetOnSheetSimulationConfig, read_config
    monkeypatch.chdir(REPO_ROOT)
    raw = read_config(REPO_ROOT / "examples" / "pes_sheet_config.ini")
    raw["general"]["interlayer_style"] = "ilp"
    raw["2D"].update({"x": 20, "y": 20, "lat_c": 6.15,
                      "pot_path": str(SW / "MoS2_sevik.sw")})
    raw["pes"].update({"grid_n": 2, "grid_n_refine": None, "z_relax": True})
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    config = SheetOnSheetSimulationConfig(**raw, settings=settings)
    out = tmp_path / "out"
    sim = PESSheetSimulation(config, output_dir=str(out))
    sim.build()
    script = out / "lammps" / "slide.in"
    assert "compute         e_il all pair ilp/tmd" in script.read_text()
    # Baked paths are relative to the simulation root, one level above the run dir.
    root = out
    for _ in Path(str(sim.relative_run_dir)).parts:
        root = root.parent
    proc = subprocess.run(["lmp", "-in", str(script), "-log", str(tmp_path / "log")],
                          cwd=str(root), capture_output=True, text=True,
                          timeout=900, check=False)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    csvs = [p for p in out.rglob("results/*") if p.is_file()]
    assert csvs, list(out.rglob("*"))
    rows = [ln.split(",") for ln in csvs[0].read_text().splitlines()
            if ln and not ln.startswith("#")]
    assert rows[0][-1] == "e_interlayer_eV", rows[0]
    values = [float(r[-1]) for r in rows[1:]]
    assert len(values) == 4                       # 2 x 2 grid
    assert all(math.isfinite(v) and v < 0 for v in values), values


# --------------------------------------------------------------------------
# The shipped example (fast, no LAMMPS run)
# --------------------------------------------------------------------------

def test_example_interlayer_builds(build_example):
    """examples/sheet_config_interlayer.ini: three runs, one per style, all on
    the reaction-force branch (comparison mode), each with its own term."""
    result = build_example("sheet_config_interlayer.ini", "sheetonsheet")
    assert result.ok, result.render()
    members = sorted(Path(p) for p in result.created)
    assert len(members) == 3
    # Directory tokens: il-<style>, ``lj_g`` spelt ljg, ``-cmp`` for comparison.
    tokens = {"il-lj-cmp": "lj", "il-ljg-cmp": "lj_g", "il-ilp-cmp": "ilp"}
    by_style = {tokens[m.name]: m for m in members}
    assert sorted(by_style) == ["ilp", "lj", "lj_g"]
    for style, member in by_style.items():
        slide = next(member.rglob("lammps/slide.in")).read_text()
        settings = next(member.rglob("lammps/system.in.settings")).read_text()
        assert "variable        xfrict equal v_sx" in slide, style
        assert "compute         frict" not in slide, style
        assert ("kolmogorov/crespi/z" in settings) == (style == "lj_g"), style
        assert ("ilp/tmd" in settings) == (style == "ilp"), style
