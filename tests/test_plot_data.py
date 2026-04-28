"""Focused tests for postprocessing Plotter safety and matching logic."""

import json

import numpy as np
import pandas as pd

from src.postprocessing.plot_data import Plotter


def _plotter(tmp_path):
    return Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))


def test_add_derived_columns_tipspeed_is_finite_with_zero_dt(tmp_path):
    """tipspeed should never contain NaN/inf when time does not advance."""
    plotter = _plotter(tmp_path)
    df = pd.DataFrame(
        {
            "tipx": [0.0, 1.0, 2.0],
            "tipy": [0.0, 0.0, 0.0],
            "time": [0.0, 0.0, 0.0],
        },
    )

    out = plotter._add_derived_columns(df)  # pylint: disable=protected-access

    assert "tipspeed" in out
    assert np.isfinite(out["tipspeed"]).all()
    assert out["tipspeed"].tolist() == [0.0, 0.0, 0.0]


def test_add_derived_columns_cof_avoids_infinite_values(tmp_path):
    """cof should use NaN for zero normal force rather than inf."""
    plotter = _plotter(tmp_path)
    df = pd.DataFrame(
        {
            "lf": [1.0, -2.0, 3.0],
            "nf": [2.0, 0.0, -3.0],
        },
    )

    out = plotter._add_derived_columns(df)  # pylint: disable=protected-access

    assert out["cof"].iloc[0] == 0.5
    assert np.isnan(out["cof"].iloc[1])
    assert out["cof"].iloc[2] == 1.0


def test_material_names_match_is_prefix_and_underscore_tolerant(tmp_path):
    """Material matching should remain permissive for legacy IDs."""
    plotter = _plotter(tmp_path)

    assert plotter._material_names_match("h_MoS2", "MoS2")  # pylint: disable=protected-access
    assert plotter._material_names_match("Mo_S2", "MoS2")  # pylint: disable=protected-access
    assert not plotter._material_names_match("h_MoS2", "WSe2")  # pylint: disable=protected-access


def test_generate_summary_plot_creates_output_png(tmp_path):
    """Plotter with synthetic data should write a PNG without error."""
    outputs_dir = tmp_path / "data" / "outputs"
    outputs_dir.mkdir(parents=True)

    synthetic = {
        "metadata": {},
        "results": {
            "MoS2": {
                "f10.0": {"a0": {"columns": ["lfx", "lfy", "nf"],
                                 "data": [[1.0, 0.0, 10.0]] * 5}},
                "f20.0": {"a0": {"columns": ["lfx", "lfy", "nf"],
                                 "data": [[2.0, 0.0, 20.0]] * 5}},
                "f30.0": {"a0": {"columns": ["lfx", "lfy", "nf"],
                                 "data": [[3.0, 0.0, 30.0]] * 5}},
            }
        },
    }
    (outputs_dir / "output_full_100x100y.json").write_text(
        json.dumps(synthetic), encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    plotter = Plotter(
        data_dirs=[str(tmp_path / "data")],
        labels=["test"],
        output_dir=str(out_dir),
    )

    plot_config = {
        "x_axis": "force",
        "y_axis": "lf",
        "filename": "summary_test.png",
        "title": "Integration test",
    }
    plotter._generate_summary_plot(plot_config)  # pylint: disable=protected-access

    assert (out_dir / "summary_test.png").exists()
