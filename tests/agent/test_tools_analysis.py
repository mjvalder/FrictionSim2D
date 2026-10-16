"""Analysis tools: find result sets, summarise them, list plot types and schemas."""
import json

import pytest

from src.agent.changes import ToolInputError
from src.agent.sandbox import SandboxError
from src.agent.tools import RESULT_LIMIT
from src.agent.tools.analysis import (find_result_sets, plot_schema, plot_types,
                                      summarise_results)
from src.postprocessing.plot_config import PLOT_TYPES
from src.postprocessing.read_data import DataReader
from tests.helpers.agent import make_deps
from tests.helpers.results_tree import make_family_run, slide_config


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    ws = deps.roots.workspace
    campaign = ws / 'campaign'
    make_family_run(campaign, 'afm', loads=(10.0, 30.0),
                    config=slide_config('afm', loads=(10.0, 30.0, 50.0)))
    make_family_run(campaign, 'sheetonsheet', material='h-WS2')
    DataReader(results_dir=str(campaign)).export_full_data_to_json()
    make_family_run(ws / 'raw', 'afm', loads=(10.0,),
                    config=slide_config('afm', loads=(10.0, 30.0)))
    return deps


def _doc(deps):
    return deps.roots.workspace / 'campaign' / 'outputs' / 'output_full_100x100y.json'


# ---- find_result_sets --------------------------------------------------------

def test_find_lists_datasets_and_simulation_roots(deps):
    text = find_result_sets(deps)
    assert 'workspace:/campaign — dataset (sizes: 100x100y; 4 runs)' in text
    assert 'workspace:/campaign — simulation root (afm: 1 runs, sheetonsheet: 1 runs)' in text
    assert 'workspace:/raw — simulation root (afm: 1 runs)' in text
    assert 'workspace:/raw — dataset' not in text
    assert str(deps.roots.workspace) not in text


def test_find_sees_exported_sets_but_no_other_dot_folder(deps):
    ws = deps.roots.workspace
    for parent in (ws / '.preview' / '_datasets' / 'set_b', ws / '.hidden' / 'x'):
        (parent / 'outputs').mkdir(parents=True)
        (parent / 'outputs' / 'output_full_50x50y.json').write_text('{"runs": []}')
    text = find_result_sets(deps)
    assert 'workspace:/.preview/_datasets/set_b — dataset (sizes: 50x50y; 0 runs)' in text
    assert '.hidden' not in text


def test_find_walk_is_depth_bounded(deps):
    deep = deps.roots.workspace / 'a' / 'b' / 'c' / 'd' / 'e'
    (deep / 'outputs').mkdir(parents=True)
    (deep / 'outputs' / 'output_full_1x1y.json').write_text('{}')
    (deep.parent / 'outputs').mkdir()
    (deep.parent / 'outputs' / 'output_full_1x1y.json').write_text('{}')
    text = find_result_sets(deps)
    assert 'workspace:/a/b/c/d — dataset' in text
    assert 'workspace:/a/b/c/d/e' not in text


def test_find_caps_hits(deps):
    for i in range(5):
        out = deps.roots.workspace / f'set{i}' / 'outputs'
        out.mkdir(parents=True)
        (out / 'output_full_1x1y.json').write_text('{}')
    text = find_result_sets(deps, max_hits=3)
    assert text.count(' — ') == 3 and 'stopped after 3' in text


def test_find_says_when_nothing_is_there(tmp_path, mock_settings):
    assert 'no datasets' in find_result_sets(make_deps(tmp_path, mock_settings))


# ---- summarise_results: datasets ---------------------------------------------

def test_summarise_dataset_reports_metadata_and_rows(deps):
    text = summarise_results(deps, 'campaign')
    assert 'complete 3, no_output 1' in text
    assert 'h_MoS2' in text and 'h_WS2' in text
    assert 'force 10, 30' in text
    assert 'time_series' not in text and 'tiers' not in text
    rows = [line for line in text.splitlines() if line.startswith('h_')]
    assert len(rows) == 4
    assert rows[-1].endswith('no_output')      # complete first
    assert len(text) <= RESULT_LIMIT


def test_summarise_filters_by_material_and_size(deps):
    text = summarise_results(deps, 'workspace:/campaign', material='h-WS2')
    rows = [line for line in text.splitlines() if line.startswith('h_')]
    assert len(rows) == 1 and rows[0].startswith('h_WS2')
    assert 'no runs match' in summarise_results(deps, 'campaign', size='50x50y')
    sized = summarise_results(deps, 'campaign', size='100x_100y')
    assert len([ln for ln in sized.splitlines() if ln.startswith('h_')]) == 4


def test_summarise_caps_rows(deps):
    text = summarise_results(deps, 'campaign', max_rows=2)
    assert len([ln for ln in text.splitlines() if ln.startswith('h_')]) == 2
    assert '… 2 more rows' in text


def test_summarise_shows_scalar_post_entries_only(deps):
    doc = json.loads(_doc(deps).read_text())
    doc['runs'][0]['post'] = {'cof_mean': 0.0123456, 'peaks': [1, 2, 3]}
    _doc(deps).write_text(json.dumps(doc))
    text = summarise_results(deps, 'campaign')
    assert 'cof_mean=0.01235' in text and 'peaks' not in text


def test_summarise_refuses_oversized_json(deps):
    text = summarise_results(deps, 'campaign', max_json_mb=1e-6)
    assert 'output_full_100x100y.json' in text and 'too large' in text
    assert 'h_MoS2' not in text


def test_summarise_accepts_the_results_file_itself(deps):
    text = summarise_results(deps, 'campaign/outputs/output_full_100x100y.json')
    assert 'complete 3, no_output 1' in text


# ---- summarise_results: raw simulation roots ---------------------------------

def test_summarise_raw_root_uses_tokens_and_expected_points(deps):
    text = summarise_results(deps, 'raw')
    assert 'simulation root, 1 runs' in text
    assert 'afm h-MoS2 100x100y T300 L2' in text
    assert '2 points expected' in text and '1 results files' in text
    assert 'postprocess read' in text


def test_summarise_raw_root_filters(deps):
    assert 'no runs match' in summarise_results(deps, 'raw', material='h-WS2')


def test_summarise_refuses_outside_paths(deps):
    with pytest.raises(SandboxError):
        summarise_results(deps, '/etc')
    with pytest.raises(SandboxError):
        summarise_results(deps, '../')


def test_summarise_rejects_a_folder_without_results(deps):
    with pytest.raises(ToolInputError, match='find_result_sets'):
        summarise_results(deps, 'configs')


# ---- plot types / schema -----------------------------------------------------

def test_plot_types_lists_every_type_with_a_line():
    text = plot_types(None)
    lines = text.splitlines()
    for plot_type in PLOT_TYPES:
        assert any(line.startswith(f'{plot_type}: ') for line in lines)
    assert 'Rank materials' in text


def test_plot_schema_is_that_types_slice():
    text = plot_schema(None, 'timeseries')
    assert text.startswith('timeseries')
    assert 'x_axis' in text and 'src.postprocessing' not in text
    assert 'Optional[' not in text and '|' not in text
    assert 'add_linear_fit' not in text          # a summary-only field
    assert len(plot_schema(None, 'summary')) <= RESULT_LIMIT


def test_plot_schema_unknown_type():
    with pytest.raises(ToolInputError, match='rank_friction'):
        plot_schema(None, 'histogram')
