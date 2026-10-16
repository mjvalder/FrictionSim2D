"""Plot proposals: a plot validated against the plot schema, datasets from the workspace."""
import pytest

from src.agent.changes import ToolInputError, describe_change
from src.agent.plot_changes import PlotChange, apply_plot_change, build_plot_change
from src.agent.sandbox import SandboxError
from src.agent.tools import RESULT_LIMIT
from src.agent.tools.plots import PlotProposalRequest, propose_plot
from src.postprocessing.plot_config import PlotDocument
from tests.helpers.agent import make_deps


def _dataset(folder):
    (folder / 'outputs').mkdir(parents=True)
    (folder / 'outputs' / 'output_full_100x100y.json').write_text('{"runs": []}')
    return folder


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    _dataset(deps.roots.workspace / 'campaign')
    _dataset(deps.roots.workspace / 'other')
    (deps.roots.workspace / 'not_a_set').mkdir()
    return deps


def _request(**kw):
    kw.setdefault('plot_type', 'summary')
    kw.setdefault('datasets', ['workspace:/campaign'])
    kw.setdefault('fields', {'x_axis': 'force', 'y_axis': 'cof'})
    return PlotProposalRequest(**kw)


def test_valid_plot_builds_a_change(deps):
    change = build_plot_change(deps, 'COF vs load', 'summary',
                               {'x_axis': 'force', 'y_axis': 'cof'}, ['workspace:/campaign'])
    assert isinstance(change, PlotChange) and change.ok
    assert change.plot == {'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'cof',
                           'datasets': ['campaign']}
    assert change.add_datasets == ['campaign']
    assert str(deps.roots.workspace) not in describe_change(change)


def test_tool_returns_a_short_handle_and_records_the_change(deps):
    handle = propose_plot(deps, _request())
    assert 'summary' in handle and 'campaign' in handle and len(handle) <= RESULT_LIMIT
    assert 'do not say it is applied' in handle
    assert isinstance(deps.pending[-1], PlotChange)
    assert str(deps.roots.workspace) not in handle


def test_unknown_field_lists_the_valid_ones(deps):
    with pytest.raises(ToolInputError) as err:
        propose_plot(deps, _request(fields={'x_axs': 'force'}))
    text = str(err.value)
    assert 'x_axs' in text and 'x_axis' in text and 'y_axis' in text


def test_bad_value_is_a_retry_hint(deps):
    with pytest.raises(ToolInputError, match='legend_outside'):
        propose_plot(deps, _request(fields={'legend_outside': 'maybe'}))


def test_dataset_outside_the_workspace_is_refused(deps, tmp_path):
    outside = _dataset(tmp_path / 'elsewhere')
    for path in (str(outside), '../elsewhere'):
        with pytest.raises(SandboxError):
            propose_plot(deps, _request(datasets=[path]))
    assert deps.pending == []


def test_example_folder_is_not_a_plot_dataset(deps):
    with pytest.raises(SandboxError, match='workspace'):
        propose_plot(deps, _request(datasets=['examples:/materials']))


def test_folder_without_results_is_refused(deps):
    with pytest.raises(ToolInputError, match='find_result_sets'):
        propose_plot(deps, _request(datasets=['not_a_set']))


def test_plot_document_snapshot_is_visible_to_the_tool(deps):
    ws = deps.roots.workspace
    deps.plot_document = PlotDocument(data_dirs=[str(ws / 'campaign')],
                                      labels=['my-campaign']).model_dump(mode='json')
    change = build_plot_change(deps, 't', 'summary', {}, ['campaign', 'other'])
    assert change.add_datasets == ['other']
    assert change.plot['datasets'] == ['my-campaign', 'other']
    assert 'already loaded' in change.handle()


def test_direct_mode_moves_the_plot_snapshot_on(deps):
    deps.direct_apply = True
    handle = propose_plot(deps, _request())
    assert 'will be added' in handle
    assert deps.plot_document['labels'] == ['campaign']
    assert len(deps.plot_document['plots']) == 1
    change = build_plot_change(deps, 't', 'summary', {}, ['campaign'])
    assert change.add_datasets == []


def test_apply_adds_missing_datasets_and_appends_the_plot(deps):
    change = build_plot_change(deps, 't', 'timeseries', {'y_axis': 'lfx'},
                               ['campaign', 'other'])
    ws = deps.roots.workspace
    current = PlotDocument(data_dirs=[str(ws / 'other')], labels=['other']).model_dump(
        mode='json', exclude_unset=True)
    new, added = apply_plot_change(change, current, ws)
    document = PlotDocument.model_validate(new)
    assert document.data_dirs == [str(ws / 'other'), str(ws / 'campaign')]
    assert document.labels == ['other', 'campaign'] and added == ['campaign']
    assert document.plots[-1].plot_type == 'timeseries'
    assert current['data_dirs'] == [str(ws / 'other')]  # not mutated
