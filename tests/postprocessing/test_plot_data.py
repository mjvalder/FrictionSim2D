"""Focused tests for postprocessing Plotter safety and matching logic."""

import json

import numpy as np
import pandas as pd

from src.postprocessing.plot_data import Plotter



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


def test_extract_all_runs_keeps_a_negative_scan_angle(tmp_path):
    """The per-angle leaf keys carry the sign (a-15 for -15 deg), so the key
    parser has to as well -- otherwise a negative-angle run reaches the plots
    with no angle at all and is dropped by every angle filter."""
    outputs_dir = tmp_path / "data" / "outputs"
    outputs_dir.mkdir(parents=True)

    columns = ["lfx", "lfy", "nf"]
    rows = [[1.0, 0.0, 10.0]] * 5
    synthetic = _v2_doc([
        _v2_run("MoS2", columns, rows, load=10.0, angle=0.0),
        _v2_run("MoS2", columns, rows, load=10.0, angle=-15.0),
    ])
    (outputs_dir / "output_full_100x100y.json").write_text(
        json.dumps(synthetic), encoding="utf-8"
    )

    plotter = Plotter(
        data_dirs=[str(tmp_path / "data")],
        labels=["test"],
        output_dir=str(tmp_path / "out"),
    )

    runs = list(plotter._extract_all_runs("test", "100x100y"))  # pylint: disable=protected-access

    assert sorted(run["angle"] for run in runs) == [-15.0, 0.0]


def test_generate_summary_plot_creates_output_png(tmp_path):
    """Plotter with synthetic data should write a PNG without error."""
    outputs_dir = tmp_path / "data" / "outputs"
    outputs_dir.mkdir(parents=True)

    columns = ["lfx", "lfy", "nf"]
    synthetic = _v2_doc([
        _v2_run("MoS2", columns, [[1.0, 0.0, 10.0]] * 5, load=10.0),
        _v2_run("MoS2", columns, [[2.0, 0.0, 20.0]] * 5, load=20.0),
        _v2_run("MoS2", columns, [[3.0, 0.0, 30.0]] * 5, load=30.0),
    ])
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


def test_settings_lists_are_replaced_not_extended(tmp_path):
    """`--settings {"figure": {"size": [8, 6]}}` used to produce
    [10, 7, 8, 6], and matplotlib then read the first two -- so the option
    silently did nothing."""
    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path),
                      settings={'figure': {'size': [8, 6]},
                                'export': {'formats': ['pdf']}})

    assert plotter.settings['figure']['size'] == [8, 6]
    assert plotter.settings['export']['formats'] == ['pdf']
    assert plotter.figure_size == (8, 6)


def test_one_plotters_settings_do_not_leak_into_the_next(tmp_path):
    """DEFAULT_SETTINGS was copied shallowly, so every sub-dict was shared
    with the module-level default."""
    from src.postprocessing.plot_data import Plotter

    Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path),
            settings={'figure': {'dpi': 999}})
    second = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))

    assert second.settings['figure']['dpi'] == 150


def test_metadata_merging_still_unions_lists(tmp_path):
    """`_load_all_metadata` folds every dataset's metadata into one dict, and
    there a list is a set of values seen, not a value. It gets its own merge
    rather than riding on the settings one, which now replaces."""
    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    accumulated = {'materials': ['h_MoS2'], 'sizes': ['100x100y'],
                   'material_types': {'h_type': ['h_MoS2']}}

    plotter._merge_metadata(accumulated, {  # pylint: disable=protected-access
        'materials': ['h_WS2'], 'sizes': ['100x100y'],
        'material_types': {'h_type': ['h_WS2'], 't_type': ['t_CoTe2']}})

    assert accumulated['materials'] == ['h_MoS2', 'h_WS2']
    assert accumulated['sizes'] == ['100x100y']          # de-duplicated
    assert accumulated['material_types']['h_type'] == ['h_MoS2', 'h_WS2']
    assert accumulated['material_types']['t_type'] == ['t_CoTe2']


def test_the_two_merges_disagree_on_purpose(tmp_path):
    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    settings_side = {'formats': ['png']}
    metadata_side = {'formats': ['png']}

    plotter._deep_merge_dict(settings_side, {'formats': ['pdf']})  # pylint: disable=protected-access
    plotter._merge_metadata(metadata_side, {'formats': ['pdf']})  # pylint: disable=protected-access

    assert settings_side['formats'] == ['pdf']          # a value: replaced
    assert metadata_side['formats'] == ['pdf', 'png']   # a set seen: unioned


def test_nested_dicts_still_merge(tmp_path):
    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path),
                      settings={'fonts': {'title': 12}})

    assert plotter.settings['fonts']['title'] == 12
    assert plotter.settings['fonts']['legend'] == 16      # untouched


def test_constrained_layout_reaches_the_figure(tmp_path):
    import matplotlib.pyplot as plt

    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path),
                      settings={'layout': {'constrained_layout': True}})
    fig, _ax = plotter.new_figure()
    try:
        assert fig.get_layout_engine() is not None
    finally:
        plt.close(fig)


def test_save_plot_does_not_fight_constrained_layout(tmp_path):
    """subplots_adjust on a constrained-layout figure raises a matplotlib
    warning and disables the engine; _save_plot must skip it."""
    import matplotlib.pyplot as plt

    from src.postprocessing.plot_data import Plotter

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path),
                      settings={'layout': {'constrained_layout': True}})
    fig, ax = plotter.new_figure()
    ax.plot([0, 1], [0, 1])

    plotter._save_plot(fig, 'x.png')  # pylint: disable=protected-access

    assert (tmp_path / 'x.png').exists()
    assert not plt.get_fignums()
