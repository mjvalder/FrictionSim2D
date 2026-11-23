"""Tests for PES-scan descriptor extraction (frictionsim2d/postprocessing/pes_scan.py)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from frictionsim2d.postprocessing import pes_scan
from tests.helpers.results_tree import make_family_run


def _cosine_grid(n: int = 12, amp: float = 1.0, period: float = 3.0,
                 kind: str = "sheet") -> pd.DataFrame:
    """Separable cosine corrugation E = amp*(2 - cos(2pi ix/n) - cos(2pi iy/n)).

    Analytic references: barrier_U = 4*amp, corr_rms = amp, channel barrier = 2*amp,
    anisotropy = 1, minimum at (0, 0).
    """
    rows = []
    for ix in range(n):
        for iy in range(n):
            e = amp * (2 - math.cos(2 * math.pi * ix / n) - math.cos(2 * math.pi * iy / n))
            row = {"dx": ix * period / n, "dy": iy * period / n, "energy_eV": e}
            if kind == "tip":
                # radial lateral force pointing toward the minimum, peak 0.5 nN
                row.update(x=row.pop("dx"), y=row.pop("dy"),
                           fx=0.5 * math.sin(2 * math.pi * ix / n),
                           fy=0.5 * math.sin(2 * math.pi * iy / n))
            rows.append(row)
    return pd.DataFrame(rows)


def test_material_id_underscores():
    assert pes_scan.material_id("h-MoS2") == "h_MoS2"
    assert pes_scan.material_id("black_phosphorus") == "black_phosphorus"


def test_grid_descriptors_match_analytic_cosine():
    df = _cosine_grid(n=12, amp=1.0, period=3.0)
    d = pes_scan.grid_descriptors(df, kind="sheet")

    assert d["pes_barrier_U"] == pytest.approx(4.0, abs=1e-9)
    assert d["pes_corr_rms"] == pytest.approx(1.0, abs=1e-9)
    assert d["pes_barrier_mep"] == pytest.approx(2.0, abs=1e-9)
    assert d["pes_barrier_aniso"] == pytest.approx(1.0, abs=1e-9)   # symmetric
    assert d["pes_curv_min"] > 0.0                                   # convex at min
    assert d["pes_eta"] > 0.0
    assert d["pes_grid_n"] == 12
    assert "pes_fmax" not in d                                       # sheet: no forces


def test_grid_descriptors_scale_with_amplitude():
    small = pes_scan.grid_descriptors(_cosine_grid(amp=1.0), "sheet")
    big = pes_scan.grid_descriptors(_cosine_grid(amp=3.0), "sheet")
    assert big["pes_barrier_U"] == pytest.approx(3.0 * small["pes_barrier_U"], rel=1e-9)
    assert big["pes_corr_rms"] == pytest.approx(3.0 * small["pes_corr_rms"], rel=1e-9)


def test_grid_descriptors_anisotropy_detected():
    """A grid corrugated only along x has a large a/b barrier ratio."""
    n = 12
    rows = []
    for ix in range(n):
        for iy in range(n):
            e = 1.0 - math.cos(2 * math.pi * ix / n)   # flat in y
            rows.append({"dx": ix * 0.25, "dy": iy * 0.25, "energy_eV": e})
    d = pes_scan.grid_descriptors(pd.DataFrame(rows), "sheet")
    # barrier along a (x) is finite; along b (y) is ~0 -> large ratio
    assert d["pes_barrier_aniso"] > 100.0


def test_tip_descriptors_include_fmax():
    df = _cosine_grid(kind="tip")
    d = pes_scan.grid_descriptors(df, kind="tip")
    assert d["pes_fmax"] == pytest.approx(0.5 * math.sqrt(2), rel=1e-6)
    assert "pes_flat_rms" in d


def _write_grid(root: Path, family: str, material: str,
                size=(30.0, 30.0), temp=0) -> Path:
    """A real PES run directory (provenance.json included) with a cosine grid."""
    run_dir = make_family_run(root, family, material=material, size=size,
                              temp=temp)
    csv = run_dir / 'results' / 'pes_scan.csv'
    _cosine_grid().to_csv(csv, index=False)
    return csv


def test_discover_and_run(tmp_path: Path):
    _write_grid(tmp_path, 'pes_sheet', 'h-MoS2')
    _write_grid(tmp_path, 'pes_sheet', 'h-WS2')
    _write_grid(tmp_path, 'pes_tip', 'h-MoS2', size=(60.0, 60.0), temp=300)

    sheet = pes_scan.discover_grids(tmp_path, "sheet")
    assert {m for m, _ in sheet} == {"h_MoS2", "h_WS2"}
    tip = pes_scan.discover_grids(tmp_path, "tip")
    assert {m for m, _ in tip} == {"h_MoS2"}

    data_dir = tmp_path / "data"
    grid_dir = tmp_path / "outputs" / "pes_scan"
    written = pes_scan.run(tmp_path, data_dir, grid_dir=grid_dir, kinds=["sheet"])

    assert "sheet" in written
    out = pd.read_csv(written["sheet"])
    assert list(out["material"]) == ["h_MoS2", "h_WS2"]          # sorted
    assert "pes_barrier_U" in out.columns
    assert (grid_dir / "sheet" / "h_MoS2.csv").exists()          # tidied grid copied


def test_refine_grids_are_ignored(tmp_path: Path):
    csv = _write_grid(tmp_path, 'pes_sheet', 'h-MoS2')
    # a refine grid alongside must not create a second/duplicate material row
    _cosine_grid(n=20).to_csv(csv.parent / "pes_scan_refine.csv", index=False)

    found = pes_scan.discover_grids(tmp_path, "sheet")

    assert found == [("h_MoS2", csv)]


def test_a_grid_without_a_provenance_record_is_not_found(tmp_path: Path):
    """discover_runs finds runs, and a run is one provenance.json. A stray
    CSV under a run-shaped path is not a run -- the old rglob picked it up
    and filed it under whatever directory name sat in the material slot."""
    stray = tmp_path / 'pes_sheet' / 'h-MoS2' / '30x_30y' / 'K0' / 'results'
    stray.mkdir(parents=True)
    _cosine_grid().to_csv(stray / 'pes_scan.csv', index=False)

    assert pes_scan.discover_grids(tmp_path, "sheet") == []


def test_a_potential_sweep_yields_one_grid_per_arm(tmp_path: Path):
    for pot in ('sw_a', 'sw_b'):
        run_dir = make_family_run(tmp_path, 'pes_sheet', material='h-MoS2',
                                  size=(30.0, 30.0), temp=0, pot=pot)
        _cosine_grid().to_csv(run_dir / 'results' / 'pes_scan.csv', index=False)

    found = pes_scan.discover_grids(tmp_path, "sheet")

    assert len(found) == 2
    assert {m for m, _ in found} == {"h_MoS2"}
    assert len({p for _, p in found}) == 2


def test_run_returns_empty_when_no_grids(tmp_path: Path):
    assert pes_scan.run(tmp_path, tmp_path / "data") == {}
