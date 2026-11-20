"""documentation/publication/plots_publication.json: the article's figures."""

import json
from pathlib import Path

from src.postprocessing.plot_config import load_plot_document

PUB = Path(__file__).resolve().parents[2] / 'documentation' / 'publication'


def test_publication_document_validates():
    document = load_plot_document(PUB / 'plots_publication.json')
    assert len(document.plots) == 16
    assert len(document.data_dirs) == 11
    assert all(d.startswith(('aiida://dataset/', 'aiida://analysis/'))
               for d in document.data_dirs)
    assert ('aiida://analysis/period_summary/261107-sheet?geometry_dataset=260909-afm'
            in document.data_dirs)


def test_every_table_and_snapshot_is_read_from_the_store():
    """No `tables/` or `trees/` on disk: the analysis plots read stored
    analyses, the snapshot plots a dataset."""
    raw = json.loads((PUB / 'plots_publication.json').read_text(encoding='utf-8'))
    sources = {}
    for plot in raw['plots']:
        assert 'run_dir' not in plot, plot['filename']
        for key in ('stickslip_file', 'ranking_file', 'contact_file',
                    'traces_file', 'status_file'):
            if key in plot:
                assert isinstance(plot[key], dict), (plot['filename'], key)
                sources[(plot['filename'], key)] = plot[key]['analysis']
        if plot['plot_type'] in ('sheet_profile', 'integrity_panel'):
            assert plot['dataset'] == '260909-afm'
    assert set(sources.values()) == {'stick_slip', 'cof_ranking', 'contact',
                                     'integrity_traces', 'run_status'}


def test_every_shipped_input_it_names_is_there():
    raw = json.loads((PUB / 'plots_publication.json').read_text(encoding='utf-8'))
    names = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'file' and isinstance(value, str) and value.endswith('.json'):
                    names.add(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw['plots'])
    assert names and all((PUB / n).is_file() for n in names), names


def test_every_published_figure_has_a_plot():
    raw = json.loads((PUB / 'plots_publication.json').read_text(encoding='utf-8'))
    made = {p['filename'] for p in raw['plots']}
    made |= {p['spectrum_filename'] for p in raw['plots'] if p.get('spectrum_filename')}
    assert {f.name for f in (PUB / 'Images' / 'plots').glob('*.png')} == made


def _expand(name):
    """`a_{x,y}{,_s}.png` -> every file name it stands for."""
    import itertools  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel
    parts = re.split(r'\{([^}]*)\}', name)
    choices = [[p] if i % 2 == 0 else p.split(',') for i, p in enumerate(parts)]
    return {''.join(c) for c in itertools.product(*choices)}


def test_known_differences_name_published_figures():
    """README's known-differences list names real figures, and the comparison
    script points there instead of claiming any figure reproduces."""
    import re  # pylint: disable=import-outside-toplevel
    text = (PUB / 'README.md').read_text(encoding='utf-8')
    section = text.split('## Known differences from the published figures', 1)[1].split('\n## ', 1)[0]
    named = set()
    for line in section.splitlines():
        if line.startswith('- '):
            for name in re.findall(r'`([^`]+\.png)`', line.split(':', 1)[0]):
                named |= _expand(name)
    published = {f.name for f in (PUB / 'Images' / 'plots').glob('*.png')}
    assert named and named <= published, named - published
    assert len(named) == 14
    script = (PUB / 'compare_to_reference.py').read_text(encoding='utf-8')
    assert 'Known differences' in script and 'stick-slip' not in script


def test_mlip_series_read_the_whole_window_datasets():
    """Spec 7.2: the MLIP series are read from the store, not a frozen JSON."""
    raw = json.loads((PUB / 'plots_publication.json').read_text(encoding='utf-8'))
    dirs = dict(zip(raw['labels'], raw['data_dirs']))
    series = [s for p in raw['plots'] for s in p.get('external_series', []) if 'dataset' in s]
    assert len(series) == 6                       # three MLIPs, two figures
    for entry in series:
        assert dirs[entry['dataset']].endswith('-whole'), entry
        assert entry['y_key'] == 'lf_sx_sy'
    assert not (PUB / 'mos2_mlip_lf_nf.json').exists()
    assert (PUB / 'settings_default.yaml').is_file()
