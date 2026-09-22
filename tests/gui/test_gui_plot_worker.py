"""The plot worker renders out of process (G2).

No Qt here: `main()` is a plain function over two JSON files, which is what
makes the QProcess protocol testable without a QApplication.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.gui.plot_worker import main, run
from src.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', loads=(10.0, 30.0, 50.0))
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    return root


def _request(tmp_path: Path, data_dir: Path, **over) -> Path:
    """A request file, writing a one-summary-plot document as its default.

    The default document is only written when the caller does NOT pass its
    own `document_file`: several tests write their own at the same path, and
    writing unconditionally would clobber it and silently test the default.
    """
    document = tmp_path / 'plots.json'
    if 'document_file' not in over:
        document.write_text(json.dumps({
            'data_dirs': [str(data_dir)], 'labels': ['ds'],
            'plots': [{'plot_type': 'summary', 'x_axis': 'force',
                       'y_axis': 'lf', 'filter_size': '100x100y',
                       'filename': 'lf.png'}],
        }), encoding='utf-8')
    payload = {'document_file': str(document),
               'output_dir': str(tmp_path / 'figures'),
               'preset': None, 'settings': None, 'plot_indices': None,
               'result_file': str(tmp_path / 'result.json')}
    payload.update(over)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_run_renders_and_reports_its_files(tmp_path):
    data_dir = _dataset(tmp_path)
    request = json.loads(
        _request(tmp_path, data_dir).read_text(encoding='utf-8'))

    result = run(request)

    assert result['ok'] is True
    assert result['warnings'] == []
    assert len(result['figures']) == 1
    figure = result['figures'][0]
    assert figure['index'] == 0
    assert figure['plot_type'] == 'summary'
    assert [Path(f).name for f in figure['files']] == ['lf.png']
    assert Path(figure['files'][0]).is_file()


def test_main_writes_the_result_file_and_exits_zero(tmp_path, capsys):
    data_dir = _dataset(tmp_path)
    request = _request(tmp_path, data_dir)

    code = main([str(request)])

    assert code == 0
    result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert result['ok'] is True
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.strip()]
    assert lines[0] == {'kind': 'progress', 'message': 'plot 1/1: summary'}
    assert lines[-1] == {'kind': 'done', 'ok': True}


def test_plot_indices_select_a_subset(tmp_path):
    data_dir = _dataset(tmp_path)
    document = tmp_path / 'plots.json'
    document.write_text(json.dumps({
        'data_dirs': [str(data_dir)], 'labels': ['ds'],
        'plots': [
            {'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'lf',
             'filter_size': '100x100y', 'filename': 'a.png'},
            {'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'cof',
             'filter_size': '100x100y', 'filename': 'b.png'}],
    }), encoding='utf-8')
    request = json.loads(_request(
        tmp_path, data_dir, document_file=str(document),
        plot_indices=[1]).read_text(encoding='utf-8'))

    result = run(request)

    assert [f['index'] for f in result['figures']] == [1]
    assert (tmp_path / 'figures' / 'b.png').is_file()
    assert not (tmp_path / 'figures' / 'a.png').exists()


def test_the_preset_and_settings_are_applied_in_order(tmp_path):
    data_dir = _dataset(tmp_path)
    request = json.loads(_request(
        tmp_path, data_dir, preset='publication',
        settings={'export': {'formats': ['png']}}).read_text(encoding='utf-8'))

    result = run(request)

    files = [Path(f).name for f in result['figures'][0]['files']]
    assert files == ['lf.png']          # settings won over the preset's pdf/svg


def test_a_broken_document_is_a_failure_result_not_a_crash(tmp_path, capsys):
    document = tmp_path / 'plots.json'
    document.write_text(json.dumps({
        'data_dirs': ['a'], 'labels': ['a'],
        'plots': [{'plot_type': 'summary', 'wobble': 1}]}), encoding='utf-8')
    request = _request(tmp_path, tmp_path, document_file=str(document))

    code = main([str(request)])

    assert code == 1
    result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert result['ok'] is False
    assert 'wobble' in result['error']
    assert 'Traceback' in result['traceback']
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        'kind': 'done', 'ok': False}


def test_one_failing_plot_becomes_a_warning_not_a_dead_batch(tmp_path):
    data_dir = _dataset(tmp_path)
    document = tmp_path / 'plots.json'
    document.write_text(json.dumps({
        'data_dirs': [str(data_dir)], 'labels': ['ds'],
        'plots': [
            {'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'lf',
             'filter_size': '100x100y', 'filename': 'ok.png'},
            {'plot_type': 'stick_slip_analysis', 'filter_size': 'nope',
             'filename': 'bad.png'}],
    }), encoding='utf-8')
    request = json.loads(_request(
        tmp_path, data_dir,
        document_file=str(document)).read_text(encoding='utf-8'))

    result = run(request)

    assert result['ok'] is True
    assert [f['index'] for f in result['figures']] == [0]
    assert len(result['warnings']) == 1
    assert 'plot 2' in result['warnings'][0]


def test_no_request_file_is_an_error_not_a_traceback(capsys):
    assert main([]) == 2
    assert 'no request file given' in capsys.readouterr().out
