"""Tests for the run-taxonomy plot types (synthetic labels, no taxonomy run)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.postprocessing.plot_data import Plotter
from src.postprocessing.plot_taxonomy import (
    TAXONOMY_MODE_COLORS,
    aggregate_signature_by_channel,
    format_material_name,
    parse_taxonomy_run_id,
    pool_taxonomy_mode,
)

MATERIALS = ["h_MoS2", "t_WSe2", "indiene"]
LOADS = [2.0, 30.0, 100.0]



def _v2_run(material, columns, rows, *, load=10.0, angle=0.0, load_kind='force',
            layers=2, size='100x100y', family='afm', model='afm'):
    """One `runs[]` entry of a v2 export document (see read_data.EXPORT_SCHEMA)."""
    return {
        'model': model, 'family': family, 'material': material,
        'materials': [material.replace('_', '-')], 'flake_material': None,
        'size': size, 'size_x': 100.0, 'size_y': 100.0,
        'substrate': 'Si', 'substrate_amorphous': True,
        'tip_material': 'Si', 'tip_radius': 25.0,
        'temperature': 300.0, 'layers': layers, 'potential': None,
        'rel_path': f'{model}/{material}/100x_100y/K300',
        'load_kind': load_kind, 'load': load, 'angle': angle, 'speed': 2.0,
        'n_layers': layers,
        'tiers': {'window': {'columns': list(columns), 'data': rows,
                             'attrs': {'lf_schema': 2, 'lf_corrected': False}}},
        'post': {},
    }


def _v2_doc(runs, metadata=None, size='100x100y'):
    """A whole v2 export document around `runs`."""
    return {
        'schema': 2, 'size': size, 'generated_by': 'FrictionSim2D',
        'units': {'time': 'step', 'nf': 'nN', 'lfx': 'nN', 'lfy': 'nN'},
        'metadata': metadata or {},
        'runs': runs,
    }

def _labels() -> pd.DataFrame:
    modes = {
        ("h_MoS2", 2.0): "marginal_contact_chatter",
        ("h_MoS2", 100.0): "ploughing",
        ("t_WSe2", 100.0): "ploughing_b",
        ("indiene", 30.0): "emergent_family_3",
        ("t_WSe2", 2.0): "t1",
        ("indiene", 2.0): "missing",
    }
    rows = []
    for layer in (1, 2):
        for material in MATERIALS:
            for load in LOADS:
                mode = modes.get((material, load), "clean" if load == 30.0 else "unassigned")
                rows.append({
                    "run_id": f"{material}/l{layer}/f{load:g}/a0",
                    "mode": mode,
                    "severity": 1.0 + LOADS.index(load),
                    "extrapolated": load != 30.0,
                })
    return pd.DataFrame(rows)


def _signatures(labels: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame({"run_id": labels["run_id"]})
    ploughing = labels["mode"].str.startswith("ploughing").to_numpy()
    sig["tipz__drift"] = np.where(ploughing, -12.0, 0.0)
    sig["tipz__q99_dev"] = np.where(ploughing, 5.0, 0.0)
    sig["nf__resid_mean"] = np.where(labels["mode"] == "marginal_contact_chatter", -8.0, 0.0)
    sig["comz__drift"] = 0.0
    return sig


@pytest.fixture(name="taxonomy_files")
def fixture_taxonomy_files(tmp_path):
    labels = _labels()
    labels_path = tmp_path / "run_labels.csv"
    labels.to_csv(labels_path, index=False)
    sig_path = tmp_path / "run_signatures.csv"
    _signatures(labels).to_csv(sig_path, index=False)
    return labels_path, sig_path


def _plotter(tmp_path):
    return Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path / "out"))


def test_parse_run_id_splits_coordinates():
    assert parse_taxonomy_run_id("t_PtTe2/l2/f30/a54") == {
        "material": "t_PtTe2", "layer": 2, "force": 30.0, "angle": 54.0,
    }
    assert parse_taxonomy_run_id("not-a-run") is None


def test_pool_mode_keeps_template_splits_and_pools_emergent():
    known = TAXONOMY_MODE_COLORS
    assert pool_taxonomy_mode("ploughing", known) == "ploughing"
    assert pool_taxonomy_mode("ploughing_b", known) == "ploughing"
    assert pool_taxonomy_mode("sheet_lift_recovery_a", known) == "sheet_lift_recovery"
    assert pool_taxonomy_mode("emergent_family_3", known) == "other"
    assert pool_taxonomy_mode("t1", known) == "t1"


def test_channel_aggregation_keeps_sign_of_dominant_feature():
    medians = pd.DataFrame(
        {"tipz__drift": [-12.0, 1.0], "tipz__q99_dev": [5.0, 2.0], "nf__mean": [0.0, np.nan]},
        index=["ploughing", "other"],
    )
    out = aggregate_signature_by_channel(medians)
    assert out.loc["ploughing", "tipz"] == -12.0
    assert out.loc["other", "tipz"] == 2.0
    assert out.loc["other", "nf"] == 0.0


def test_format_material_name_subscripts_counts():
    assert format_material_name("h_MoS2") == "h-MoS$_{2}$"
    assert format_material_name("black_phosphorus") == "black phosphorus"


def test_labels_drop_missing_and_filter(tmp_path, taxonomy_files):
    labels_path, _ = taxonomy_files
    plotter = _plotter(tmp_path)
    df = plotter._taxonomy_labels(  # pylint: disable=protected-access
        {"labels_file": str(labels_path), "filter_layer": 2},
    )
    assert "missing" not in set(df["mode"])
    assert set(df["layer"]) == {2}
    assert set(df.loc[df["mode"] == "ploughing_b", "plot_mode"]) == {"ploughing"}


def test_prevalence_fractions_exclude_integrity_failures(tmp_path, taxonomy_files):
    labels_path, _ = taxonomy_files
    plotter = _plotter(tmp_path)
    plotter.generate_plot({
        "plot_type": "taxonomy_prevalence",
        "labels_file": str(labels_path),
        "filename": "prevalence.png",
    })
    assert (tmp_path / "out" / "prevalence.png").exists()


@pytest.mark.parametrize("column_level", ["channel", "feature"])
def test_signature_heatmap_writes_png(tmp_path, taxonomy_files, column_level):
    labels_path, sig_path = taxonomy_files
    plotter = _plotter(tmp_path)
    plotter.generate_plot({
        "plot_type": "taxonomy_signature_heatmap",
        "labels_file": str(labels_path),
        "signatures_file": str(sig_path),
        "min_runs": 1,
        "column_level": column_level,
        "filename": f"heatmap_{column_level}.png",
    })
    assert (tmp_path / "out" / f"heatmap_{column_level}.png").exists()


def test_regime_map_needs_a_single_layer(tmp_path, taxonomy_files):
    labels_path, _ = taxonomy_files
    plotter = _plotter(tmp_path)
    config = {
        "plot_type": "taxonomy_regime_map",
        "labels_file": str(labels_path),
        "material_selection": "any_mode",
        "pretty_material_names": True,
        "filename": "regime.png",
    }
    plotter.generate_plot(config)
    assert not (tmp_path / "out" / "regime.png").exists()

    plotter.generate_plot({**config, "filter_layer": 2})
    assert (tmp_path / "out" / "regime.png").exists()


def test_exemplar_reads_runs_from_a_dataset(tmp_path, taxonomy_files):
    labels_path, _ = taxonomy_files
    outputs_dir = tmp_path / "data" / "outputs"
    outputs_dir.mkdir(parents=True)
    columns = ["nf", "lfx", "lfy", "tipz", "comz", "lateral_force"]
    rows = [[100.0, 1.0, 0.0, 30.0 - 0.1 * i, 20.0, 1.0] for i in range(10)]
    synthetic = _v2_doc(
        [_v2_run(m, columns, rows, load=100.0) for m in ("h_MoS2", "t_WSe2")],
        metadata={"time_series": [1000 * (i + 1) for i in range(10)]})
    (outputs_dir / "output_full_100x100y.json").write_text(json.dumps(synthetic), encoding="utf-8")

    plotter = Plotter(
        data_dirs=[str(tmp_path / "data")], labels=["tip"], output_dir=str(tmp_path / "out"),
    )
    config = {
        "plot_type": "taxonomy_exemplar",
        "labels_file": str(labels_path),
        "datasets": ["tip"],
        "filter_size": "100x100y",
        "mode": "ploughing",
        "filter_layer": 2,
        "filename": "exemplar.png",
    }
    exemplars = plotter._select_taxonomy_exemplars(config)  # pylint: disable=protected-access
    assert sorted(exemplars["material"]) == ["h_MoS2", "t_WSe2"]

    series, time_axis = plotter._load_exemplar_series(  # pylint: disable=protected-access
        exemplars, config,
    )
    assert len(series) == 2
    assert time_axis is not None and len(time_axis) == 10

    plotter.generate_plot(config)
    assert (tmp_path / "out" / "exemplar.png").exists()


def test_exemplar_reads_a_long_timeseries_file(tmp_path, taxonomy_files):
    labels_path, _ = taxonomy_files
    long_df = pd.DataFrame({
        "run_id": ["h_MoS2/l1/f100/a0"] * 5,
        "step": range(5),
        "nf": 100.0, "lateral_force": 1.0, "tipz": 30.0, "comz": 20.0,
    })
    long_path = tmp_path / "long.csv"
    long_df.to_csv(long_path, index=False)

    plotter = _plotter(tmp_path)
    plotter.generate_plot({
        "plot_type": "taxonomy_exemplar",
        "labels_file": str(labels_path),
        "timeseries_file": str(long_path),
        "run_ids": ["h_MoS2/l1/f100/a0"],
        "sample_interval": 1000,
        "filename": "exemplar_long.png",
    })
    assert (tmp_path / "out" / "exemplar_long.png").exists()
