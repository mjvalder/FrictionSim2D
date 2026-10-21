"""The knowledge corpus: BM25 search and the search_knowledge tool."""
import json

import pytest

from src.agent.knowledge_index import KnowledgeIndex, knowledge_dir, load_knowledge
from src.agent.tools.knowledge import search_knowledge
from tests.helpers.agent import make_deps

PASSAGES = [
    {'id': 'w:1', 'title': 'Superlubricity', 'section': 'Introduction',
     'url': 'https://en.wikipedia.org/wiki/Superlubricity', 'licence': 'CC-BY-SA-4.0',
     'text': 'Superlubricity is a regime of motion in which friction vanishes or very '
             'nearly vanishes, for example between incommensurate graphite layers.'},
    {'id': 'w:2', 'title': 'Molybdenum disulfide', 'section': 'Structure',
     'url': 'https://en.wikipedia.org/wiki/Molybdenum_disulfide', 'licence': 'CC-BY-SA-4.0',
     'text': 'In MoS2 each molybdenum atom sits in a trigonal prismatic cage of sulfur '
             'atoms; the layers are held together by weak van der Waals forces.'},
    {'id': 'w:3', 'title': 'Stick-slip phenomenon', 'section': 'Introduction',
     'url': 'https://en.wikipedia.org/wiki/Stick-slip_phenomenon', 'licence': 'CC-BY-SA-4.0',
     'text': 'Stick-slip is the spontaneous jerking motion that can occur while two '
             'objects are sliding over each other. ' * 12},
    {'id': 'p:1', 'title': 'Friction of 2D materials', 'section': '',
     'url': 'https://doi.org/10.1/x', 'licence': 'cc-by',
     'text': 'Graphene and MoS2 show layer-dependent friction: thinner sheets pucker '
             'in front of the tip, which raises friction.'},
]


@pytest.fixture(name='corpus')
def _corpus(tmp_path):
    path = tmp_path / 'passages.jsonl'
    path.write_text('\n'.join(json.dumps(p) for p in PASSAGES) + '\n')
    return path


def test_search_ranks_the_matching_passage_first(corpus):
    index = KnowledgeIndex.from_file(corpus)
    assert index.search('why do incommensurate layers have vanishing friction')[0].id == 'w:1'
    assert index.search('MoS2 trigonal prismatic structure')[0].id == 'w:2'
    assert [p.id for p in index.search('puckering layer dependent friction', k=1)] == ['p:1']


def test_search_without_a_match_is_empty(corpus):
    assert KnowledgeIndex.from_file(corpus).search('quantum chromodynamics') == []


def test_load_knowledge_is_cached_until_the_file_changes(corpus):
    first = load_knowledge(corpus)
    assert load_knowledge(corpus) is first
    corpus.write_text(json.dumps(PASSAGES[0]) + '\n')
    assert len(load_knowledge(corpus)) == 1


def test_load_knowledge_without_a_file_is_none(tmp_path):
    assert load_knowledge(tmp_path / 'missing.jsonl') is None


def test_knowledge_dir_honours_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert knowledge_dir() == tmp_path / 'FrictionSim2D' / 'knowledge'


def test_tool_cites_title_and_source(tmp_path, mock_settings, corpus):
    deps = make_deps(tmp_path, mock_settings)
    deps.knowledge = lambda: load_knowledge(corpus)
    text = search_knowledge(deps, 'superlubricity incommensurate')
    assert text.splitlines()[0].startswith('[1] Superlubricity')
    assert 'CC-BY-SA-4.0' in text and 'wikipedia.org/wiki/Superlubricity' in text


def test_tool_fits_the_result_budget(tmp_path, mock_settings, corpus):
    deps = make_deps(tmp_path, mock_settings)
    deps.knowledge = lambda: load_knowledge(corpus)
    assert len(search_knowledge(deps, 'stick slip sliding friction objects')) <= 1500


def test_tool_without_a_corpus_says_so(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    assert 'no knowledge corpus' in search_knowledge(deps, 'friction').lower()
    deps.knowledge = lambda: None
    assert 'no knowledge corpus' in search_knowledge(deps, 'friction').lower()


def test_tool_without_a_hit_says_so(tmp_path, mock_settings, corpus):
    deps = make_deps(tmp_path, mock_settings)
    deps.knowledge = lambda: load_knowledge(corpus)
    assert 'no passage' in search_knowledge(deps, 'quantum chromodynamics')
