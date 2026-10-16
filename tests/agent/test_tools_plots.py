"""Plot proposals: a plot validated against the plot schema, datasets from the workspace."""
import pytest

from src.agent.changes import ToolInputError, describe_change
from src.agent.plot_changes import PlotChange, apply_plot_change, build_plot_change
from src.agent.sandbox import SandboxError
from src.agent.tools import RESULT_LIMIT
from src.agent.tools.plots import PlotField, PlotProposalRequest, propose_plot
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
    kw['fields'] = [PlotField(key=k, value=v) for k, v in kw['fields'].items()]
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


def test_fields_are_key_value_pairs_and_a_repeated_key_is_refused(deps):
    request = PlotProposalRequest(plot_type='summary', datasets=['campaign'],
                                  fields=[PlotField(key='y_axis', value='cof'),
                                          PlotField(key='y_axis', value='lf')])
    with pytest.raises(ToolInputError, match='y_axis'):
        propose_plot(deps, request)


def test_unknown_dataset_label_in_fields_is_refused(deps):
    with pytest.raises(ToolInputError, match='nope'):
        build_plot_change(deps, 't', 'summary', {'datasets': ['nope']}, ['campaign'])


# ---- applying against a document that moved on ---------------------------------

def _empty():
    return PlotDocument().model_dump(mode='json', exclude_unset=True)


def test_second_change_of_a_direct_turn_applies_after_the_first_was_refused(deps):
    deps.direct_apply = True
    propose_plot(deps, _request())          # the GUI refuses (or queues) this one ...
    second = build_plot_change(deps, 't', 'timeseries', {}, ['campaign'])
    assert second.add_datasets == []        # ... though the snapshot says it is loaded
    new, added = apply_plot_change(second, _empty(), deps.roots.workspace)
    document = PlotDocument.model_validate(new)
    assert document.labels == ['campaign'] and added == ['campaign']
    assert document.plots[-1].datasets == ['campaign']


def test_dataset_removed_before_add_is_added_back(deps):
    ws = deps.roots.workspace
    deps.plot_document = PlotDocument(data_dirs=[str(ws / 'campaign')],
                                      labels=['my-campaign']).model_dump(mode='json')
    change = build_plot_change(deps, 't', 'summary', {}, ['campaign'])
    new, _added = apply_plot_change(change, _empty(), ws)
    document = PlotDocument.model_validate(new)
    assert document.data_dirs == [str(ws / 'campaign')]
    assert document.plots[-1].datasets == document.labels


def test_renamed_dataset_label_is_followed(deps):
    ws = deps.roots.workspace
    change = build_plot_change(deps, 't', 'summary', {}, ['campaign'])
    current = PlotDocument(data_dirs=[str(ws / 'campaign')], labels=['renamed']).model_dump(
        mode='json', exclude_unset=True)
    new, added = apply_plot_change(change, current, ws)
    assert added == [] and new['plots'][-1]['datasets'] == ['renamed']


def test_a_label_that_cannot_be_resolved_any_more_refuses_the_apply(deps):
    ws = deps.roots.workspace
    deps.plot_document = PlotDocument(data_dirs=[str(ws / 'other')],
                                      labels=['x']).model_dump(mode='json')
    change = build_plot_change(deps, 't', 'summary', {'datasets': ['x', 'campaign']},
                               ['campaign'])
    with pytest.raises(ToolInputError, match="'x'"):
        apply_plot_change(change, _empty(), ws)


def test_same_named_folders_get_distinct_labels(deps):
    ws = deps.roots.workspace
    for parent in ('a', 'b', 'c'):
        _dataset(ws / parent / 'results')
    change = build_plot_change(deps, 't', 'summary', {}, ['a/results', 'b/results'])
    assert change.labels == ['results', 'b/results']
    new, _added = apply_plot_change(change, _empty(), ws)
    assert new['labels'] == ['results', 'b/results']
    assert new['plots'][-1]['datasets'] == ['results', 'b/results']


def test_new_label_avoids_the_documents_labels(deps):
    ws = deps.roots.workspace
    for parent in ('a', 'c', 'd'):
        _dataset(ws / parent / 'results')
    change = build_plot_change(deps, 't', 'summary', {}, ['a/results'])
    taken = PlotDocument(data_dirs=[str(ws / 'c' / 'results')], labels=['results'])
    new, added = apply_plot_change(change, taken.model_dump(mode='json'), ws)
    assert added == ['a/results'] and new['plots'][-1]['datasets'] == ['a/results']
    both = PlotDocument(data_dirs=[str(ws / 'c' / 'results'), str(ws / 'd' / 'results')],
                        labels=['results', 'a/results'])
    new, added = apply_plot_change(change, both.model_dump(mode='json'), ws)
    assert added == ['results-2'] and new['plots'][-1]['datasets'] == ['results-2']


def test_symlinked_workspace_matches_loaded_datasets(tmp_path, mock_settings):
    real = make_deps(tmp_path, mock_settings).roots.workspace
    _dataset(real / 'campaign')
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    deps = make_deps(tmp_path, mock_settings)
    from src.agent.sandbox import SearchRoots
    deps.roots = SearchRoots(link, None)
    deps.plot_document = PlotDocument(data_dirs=[str(real / 'campaign')],
                                      labels=['campaign']).model_dump(mode='json')
    change = build_plot_change(deps, 't', 'summary', {}, ['campaign'])
    assert change.add_datasets == [] and change.labels == ['campaign']
    new, added = apply_plot_change(change, deps.plot_document, link)
    assert added == [] and len(new['data_dirs']) == 1
    fresh = build_plot_change(deps, 't', 'summary', {}, ['workspace:/campaign'])
    assert fresh.labels == ['campaign']
