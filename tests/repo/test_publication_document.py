"""documentation/publication/plots_publication.json: the article's figures."""

import json
from pathlib import Path

from src.postprocessing.plot_config import load_plot_document

PUB = Path(__file__).resolve().parents[2] / 'documentation' / 'publication'


def test_publication_document_validates():
    document = load_plot_document(PUB / 'plots_publication.json')
    assert len(document.plots) == 16
    assert len(document.data_dirs) == 11
    assert all(d.startswith('aiida://dataset/') for d in document.data_dirs)


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
