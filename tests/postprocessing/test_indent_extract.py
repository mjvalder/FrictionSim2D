"""Tests for indent delivery / postprocessing (frictionsim2d/postprocessing/indent.py)."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frictionsim2d.postprocessing import indent
from tests.helpers.results_tree import make_family_run


# --- unit helpers ----------------------------------------------------------

def test_material_id_underscores():
    assert indent.material_id("h-MoS2") == "h_MoS2"
    assert indent.material_id("a/b") == "a__b"


def test_load_token():
    assert indent._load_token(10.0) == "10"
    assert indent._load_token(30) == "30"
    assert indent._load_token(10.5) == "10.5"


def test_parse_wall_seconds():
    assert indent._parse_wall_seconds("0:00:12") == 12
    assert indent._parse_wall_seconds("1:02:03") == 3723
    assert indent._parse_wall_seconds("45") == 45


def test_split_by_load_groups_per_load_and_seed(tmp_path: Path):
    from frictionsim2d.postprocessing.read_data import DataReader

    _make_run(tmp_path)
    runs = DataReader(results_dir=str(tmp_path)).runs

    per_file = indent.split_by_load(runs, base_seed={"h_MoS2": 777})

    assert set(per_file) == {("f10.0", 777), ("f30.0", 777)}
    assert {r.load for r in per_file[("f10.0", 777)]} == {10.0}
    assert {r.tokens.layers for r in per_file[("f10.0", 777)]} == {1, 2}


def test_iter_hold_conditions_yields_material_layer_load(tmp_path: Path):
    from frictionsim2d.postprocessing.read_data import DataReader

    _make_run(tmp_path)
    runs = DataReader(results_dir=str(tmp_path)).runs

    assert set(indent.iter_hold_conditions(runs)) == {
        ("h_MoS2", 1, 10.0), ("h_MoS2", 1, 30.0),
        ("h_MoS2", 2, 10.0), ("h_MoS2", 2, 30.0)}


# --- retract_features reduction --------------------------------------------

def _synthetic_curve():
    """A physical pull-off curve: +repulsive wall -> zero -> adhesive well -> ~0."""
    z = np.linspace(0.0, 20.0, 101)
    fz = 6.0 * np.exp(-z / 0.6) - 2.0 * np.exp(-((z - 2.0) ** 2) / (2 * 0.9 ** 2))
    return pd.DataFrame({
        "step": np.arange(z.size), "z_tip": z + 80.0, "fz_tip": fz,
        "fx_tip": 0.0 * z, "fy_tip": 0.0 * z, "pe": np.cumsum(fz) * 0.2,
    })


def test_retract_features_pulloff_and_work_positive():
    feats = indent.retract_features(_synthetic_curve())
    assert feats["f_adh_min"] < 0
    assert feats["pulloff_force"] == -feats["f_adh_min"]
    assert 1.5 < feats["pulloff_force"] < 2.0
    assert feats["work_of_adhesion"] > 0
    assert feats["detached"] == 1.0


def test_work_of_adhesion_is_the_trapezoid_area_of_the_tensile_branch():
    # a triangular well 1 nN deep and 2 A wide: area 1 nN*A
    df = pd.DataFrame({"z_tip": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       "fz_tip": [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]})

    feats = indent.retract_features(df)

    assert feats["work_of_adhesion_nnA"] == pytest.approx(1.0)
    assert feats["work_of_adhesion"] == pytest.approx(1.0 / indent.EV_A_TO_NN)


def test_contact_stiffness_is_the_steep_wall_not_the_tail():
    feats = indent.retract_features(_synthetic_curve())
    assert 3.0 < feats["contact_stiffness"] < 7.0


def test_contact_stiffness_nan_without_repulsive_start():
    z = np.linspace(0.0, 5.0, 20)
    df = pd.DataFrame({"z_tip": z, "fz_tip": -np.ones_like(z)})  # purely attractive
    feats = indent.retract_features(df)
    assert math.isnan(feats["contact_stiffness"])
    assert feats["pulloff_force"] == 1.0


# --- fixture: a minimal indent run tree (hold files + retract curves) -------

_HOLD_HEADER = ("# Time-averaged data for fix fc_ave\n"
                "# TimeStep nf lfx lfy comx comy comz tipx tipy tipz\n")


def _write_hold(dir_: Path, load: int, layer: int, tipz: float, comz: float):
    dir_.mkdir(parents=True, exist_ok=True)
    # Every channel fluctuates a little: a constant one is `frozen` to the
    # integrity check, and the point would not be delivered.
    rows = "".join(
        f"{step} {30.0 + d} {d} {-d} {30.0 + d} {30.0 - d} {comz + d} "
        f"{30.0 + d} {30.0 - d} {tipz + d}\n"
        for step, d in zip((1000, 2000, 3000, 4000, 5000),
                           (0.01, -0.02, 0.015, -0.005, 0.0))
    )
    (dir_ / f"friction_f{load}_a0_s2_layer{layer}.txt").write_text(
        _HOLD_HEADER + rows, encoding="utf-8")


def _make_run(root: Path, seed=20260710, hold_steps=10000, z_step=0.2, n_steps=100):
    """A real indent tree: two layers x two loads, hold files and retract curves."""
    for layer in (1, 2):
        run_dir = make_family_run(root, 'indent', material='h-MoS2',
                                  size=(60.0, 60.0), temp=300, layers=layer,
                                  loads=(10.0, 30.0))
        results = run_dir / 'results'
        for stale in results.iterdir():
            stale.unlink()
        # Penetration (hold) files, one per load.
        _write_hold(results, load=10, layer=layer, tipz=12.0, comz=5.0)
        _write_hold(results, load=30, layer=layer, tipz=11.0, comz=5.0)
        # Adhesion (retract) curves, one per load.
        for load in (10, 30):
            _synthetic_curve().to_csv(
                results / f"indent_f{load}N_l{layer}.csv", index=False)
        prov = run_dir.parent / "provenance"
        prov.mkdir(parents=True, exist_ok=True)
        (prov / "indent_meta.json").write_text(json.dumps({
            "base_seed": seed, "hold_steps": hold_steps, "z_step": z_step,
            "n_steps": n_steps, "material": "h-MoS2",
            "layers": {"1": 111111, "2": 222222}, "loads": [10.0, 30.0],
        }), encoding="utf-8")
    return root


def test_load_indent_meta(tmp_path: Path):
    _make_run(tmp_path)
    meta = indent.load_indent_meta(tmp_path)
    assert meta["base_seed"]["h_MoS2"] == 20260710
    assert meta["hold_steps"]["h_MoS2"] == 10000
    assert meta["z_step"]["h_MoS2"] == 0.2
    assert meta["n_steps"]["h_MoS2"] == 100
    assert meta["seed"][("h_MoS2", 1)] == 111111
    assert meta["seed"][("h_MoS2", 2)] == 222222


def test_discover_curves_parses_material_layer_load(tmp_path: Path):
    _make_run(tmp_path)
    found = indent.discover_curves(tmp_path)
    conds = {(m, l, ld) for m, l, ld, _p in found}
    assert conds == {("h_MoS2", 1, 10.0), ("h_MoS2", 1, 30.0),
                     ("h_MoS2", 2, 10.0), ("h_MoS2", 2, 30.0)}


def test_run_writes_penetration_json(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    written = indent.run(tmp_path, out)

    # One penetration JSON per load, named by base seed.
    assert "output_indent_f10_s20260710.json" in written
    assert "output_indent_f30_s20260710.json" in written

    payload = json.loads(
        (out / "output_indent_f10_s20260710.json").read_text(encoding="utf-8"))
    assert payload["schema"] == 3
    assert payload["metadata"]["indent_load_nN"] == 10.0
    assert payload["metadata"]["indent_base_seed"] == 20260710
    assert {r["load"] for r in payload["runs"]} == {10.0}
    assert {r["layers"] for r in payload["runs"]} == {1, 2}
    columns = payload["runs"][0]["tiers"]["window"]["columns"]
    for col in ("nf", "comz", "tipz"):
        assert col in columns                                  # penetration channels


def test_run_writes_adhesion_features_and_curves(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    data = tmp_path / "data"
    written = indent.run(tmp_path, out, data_dir=data)

    assert "indent_features.csv" in written
    feats = pd.read_csv(data / "indent_features.csv")
    assert set(feats["layer"]) == {1, 2}
    assert {"pulloff_force", "work_of_adhesion", "contact_stiffness"} <= set(feats.columns)

    # Tidied per-condition curves; two loads per (material, layer) -> load token.
    assert (out / "indent" / "h_MoS2_l1_f10.csv").exists()
    assert (out / "indent" / "h_MoS2_l2_f30.csv").exists()


def test_run_writes_combined_manifest(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    written = indent.run(tmp_path, out)

    assert "indent_manifest.csv" in written
    with (out / "indent_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 2 layers x 2 loads = 4 conditions, deduped across hold + retract sources.
    assert len(rows) == 4
    keyed = {(r["layer"], r["load"]): r for r in rows}
    assert keyed[("1", "10.0")]["seed"] == "111111"
    assert keyed[("2", "30.0")]["seed"] == "222222"
    assert keyed[("1", "10.0")]["hold_steps"] == "10000"
    assert keyed[("2", "10.0")]["z_step"] == "0.2"
    assert keyed[("2", "10.0")]["n_steps"] == "100"


def test_run_returns_empty_when_no_indent(tmp_path: Path):
    assert indent.run(tmp_path, tmp_path / "out") == {}


def test_discover_curves_ignores_a_curve_outside_a_run(tmp_path: Path):
    """A retract CSV under a run-shaped path is not a run -- the old rglob
    picked it up and filed it under whatever sat in the material slot."""
    _make_run(tmp_path)
    stray = tmp_path / "indent" / "loose" / "results"
    stray.mkdir(parents=True)
    _synthetic_curve().to_csv(stray / "indent_f99N_l9.csv", index=False)

    conds = {(m, l, ld) for m, l, ld, _p in indent.discover_curves(tmp_path)}

    assert 99.0 not in {ld for _m, _l, ld in conds}


def test_halted_hold_is_not_delivered(tmp_path: Path):
    """A halted hold keeps its frames in DataReader, but never reaches the
    penetration JSON, the hold conditions or the manifest."""
    root = _make_run(tmp_path)
    run_dir = next(p.parent for p in root.rglob('results') if p.parent.name == 'L1')
    (run_dir / 'results' / 'halted.txt').write_text(
        '# HALT stage=slide load=30 angle=0 speed=2 T=9000 step=3000\n',
        encoding='utf-8')
    out = tmp_path / "outputs" / "indent_sims"
    indent.run(root, out)

    payload = json.loads(
        (out / "output_indent_f30_s20260710.json").read_text(encoding="utf-8"))
    assert {r["layers"] for r in payload["runs"]} == {2}
    assert {r["status"] for r in payload["runs"]} == {"complete"}
    meta = indent.load_indent_meta(root)
    (tmp_path / "o2").mkdir()
    _written, holds = indent._deliver_penetration(root, tmp_path / "o2", meta)  # pylint: disable=protected-access
    assert ("h_MoS2", 1, 30.0) not in holds and ("h_MoS2", 2, 30.0) in holds
