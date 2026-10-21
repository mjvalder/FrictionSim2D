"""Build the assistant's knowledge corpus: attributed passages from licensed sources.

Sources (spec §2, §4.2):

- **F** -- English Wikipedia articles listed in ``sources/wikipedia_titles.txt``
  (CC-BY-SA 4.0), fetched as plain-text extracts with their revision id.
- **G** -- the ``train``-tier papers of ``sources/papers.yaml`` (CC-BY, CC-BY-SA,
  CC0, public domain) that have an open PDF, text extracted with pypdf.

Every passage (~120-250 words, split on paragraph and sentence boundaries)
carries its title, section, URL, licence, revision or DOI and retrieval date.
Output: ``corpus/passages.jsonl`` and ``corpus/manifest.json`` (counts and every
source that failed, with the reason). ``corpus/`` is gitignored: CC-BY-SA text
stays out of the repository and the wheel.

    python tools/agent_data/corpus.py                 # both sources, cached
    python tools/agent_data/corpus.py --only wikipedia
    python tools/agent_data/corpus.py --install       # also copy to the app's data dir

Run with the fs2d-train env (needs pypdf and pyyaml).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
TITLES = HERE / 'sources' / 'wikipedia_titles.txt'
PAPERS = HERE / 'sources' / 'papers.yaml'
CACHE = HERE / '.cache'
OUT_DIR = HERE / 'corpus'

WIKI_API = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'FrictionSim2D-agent-data/0.1 (private research corpus; python-urllib)'
REQUEST_GAP = 2.0  # seconds between network requests
RETRIES = 5  # on HTTP 429/503, honouring Retry-After, else 10, 20, 40 ... s

MIN_WORDS, TARGET_WORDS, MAX_WORDS = 120, 200, 250
#: Wikipedia sections that are lists of links or citations, not prose.
SKIP_SECTIONS = {'references', 'external links', 'see also', 'further reading', 'notes',
                 'bibliography', 'sources', 'citations', 'footnotes'}
#: Where a paper's prose ends.
_PAPER_END = re.compile(r'^\s*(references|acknowledg(e)?ments?|data availability|'
                        r'author contributions|competing interests|conflicts? of interest)\b',
                        re.IGNORECASE | re.MULTILINE)
_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')
#: A formula whose subscript the PDF split off: 'MoS 2' -> 'MoS2' (two or more
#: element-like symbols, one digit), leaving 'Figure 2' and 'In 2019' alone.
_SPLIT_FORMULA = re.compile(r'\b((?:[A-Z][a-z]?){2,4}) (\d)\b')


# ---- network -----------------------------------------------------------------------

def _get(url: str, cache: Path) -> bytes:
    """``url``'s body, cached at ``cache`` (an empty cache file records a failure)."""
    if cache.exists():
        return cache.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(RETRIES + 1):
        time.sleep(REQUEST_GAP)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == RETRIES:
                raise IOError(f'HTTPError: {err}') from err
            wait = err.headers.get('Retry-After', '')
            time.sleep(float(wait) if wait.isdigit() else 10 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as err:
            raise IOError(f'{type(err).__name__}: {err}') from err
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(body)
    return body


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---- chunking ----------------------------------------------------------------------

def chunk(paragraphs: List[str]) -> Iterator[str]:
    """~200-word passages: whole paragraphs where they fit, sentences where they don't."""
    current: List[str] = []
    size = 0
    for paragraph in paragraphs:
        pieces = [paragraph] if len(paragraph.split()) <= MAX_WORDS else \
            _SENTENCE.split(paragraph)
        for piece in pieces:
            words = len(piece.split())
            if current and size + words > MAX_WORDS:
                yield ' '.join(current)
                current, size = [], 0
            current.append(piece)
            size += words
            if size >= TARGET_WORDS:
                yield ' '.join(current)
                current, size = [], 0
    if current and (size >= MIN_WORDS // 2):
        yield ' '.join(current)


# ---- Wikipedia (source F) ------------------------------------------------------------

def read_titles(path: Path = TITLES) -> List[str]:
    lines = (line.strip() for line in path.read_text(encoding='utf-8').splitlines())
    return [line for line in lines if line and not line.startswith('#')]


def strip_math(text: str) -> str:
    """Drop ``{\\displaystyle ...}`` LaTeX (balanced braces) from an extract."""
    out, pos = [], 0
    while True:
        start = text.find('{\\displaystyle', pos)
        if start < 0:
            return ''.join(out) + text[pos:]
        out.append(text[pos:start])
        depth, pos = 0, start
        while pos < len(text):
            depth += {'{': 1, '}': -1}.get(text[pos], 0)
            pos += 1
            if depth == 0:
                break


def is_prose(paragraph: str) -> bool:
    """Not the spaced-out symbol run a formula leaves behind ('v F σ → ⋅ ∇ ψ')."""
    tokens = paragraph.split()
    return bool(tokens) and sum(len(t) == 1 for t in tokens) <= 0.3 * len(tokens)


def wikipedia_sections(extract: str) -> Iterator[Tuple[str, List[str]]]:
    """``(section, paragraphs)`` of a plain-text extract, skipping link/citation lists."""
    section, paragraphs = 'Introduction', []
    for line in extract.splitlines():
        heading = re.match(r'^(=+)\s*(.*?)\s*\1$', line.strip())
        if heading:
            if paragraphs:
                yield section, paragraphs
            section, paragraphs = heading.group(2), []
            continue
        if line.strip():
            paragraphs.append(line.strip())
    if paragraphs:
        yield section, paragraphs


def fetch_wikipedia(title: str) -> Dict:
    """The article's plain text, canonical title, URL and revision id."""
    params = {'action': 'query', 'prop': 'extracts|info|revisions', 'explaintext': 1,
              'exsectionformat': 'wiki', 'redirects': 1, 'inprop': 'url', 'rvprop': 'ids',
              'titles': title, 'format': 'json', 'formatversion': 2}
    url = f'{WIKI_API}?{urllib.parse.urlencode(params)}'
    page = json.loads(_get(url, CACHE / 'wikipedia' / f'{_key(title)}.json'))['query']['pages'][0]
    if page.get('missing') or not page.get('extract'):
        raise LookupError('no such article')
    return {'title': page['title'], 'url': page['fullurl'], 'text': page['extract'],
            'revision': page['revisions'][0]['revid']}


def wikipedia_passages(title: str, retrieved: str) -> List[Dict]:
    article = fetch_wikipedia(title)
    passages = []
    for section, paragraphs in wikipedia_sections(strip_math(article['text'])):
        if section.casefold() in SKIP_SECTIONS:
            continue
        for text in chunk([p for p in paragraphs if is_prose(p)]):
            passages.append({
                'source': 'F', 'kind': 'wikipedia', 'title': article['title'],
                'section': section, 'url': article['url'], 'revision': article['revision'],
                'licence': 'CC-BY-SA-4.0', 'retrieved': retrieved, 'text': text})
    return passages


# ---- papers (source G) --------------------------------------------------------------

def pdf_text(body: bytes) -> str:
    """Plain text of a PDF, cut before its references/acknowledgements."""
    # pylint: disable=import-outside-toplevel
    from pypdf import PdfReader
    if not body.startswith(b'%PDF'):
        raise ValueError('not a PDF (publisher returned a web page)')
    reader = PdfReader(io.BytesIO(body))
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    ends = [m.start() for m in _PAPER_END.finditer(text) if m.start() > len(text) // 3]
    return text[:ends[0]] if ends else text


def paper_paragraphs(text: str) -> List[str]:
    """Undo PDF line wrapping: join hyphenated breaks and lines, split on blank lines
    or lines ending a sentence; drop fragments that are mostly digits or symbols."""
    text = unicodedata.normalize('NFKC', text)  # ligatures: 'ﬁ' -> 'fi'
    text = re.sub(r'-\n(?=[a-z])', '', text)
    blocks = re.split(r'\n\s*\n|(?<=[.!?])\n(?=[A-Z])', text)
    paragraphs = []
    for block in blocks:
        block = _SPLIT_FORMULA.sub(r'\1\2', re.sub(r'\s+', ' ', block).strip())
        letters = sum(c.isalpha() for c in block)
        if len(block.split()) >= 12 and letters > 0.6 * len(block):
            paragraphs.append(block)
    return paragraphs


def paper_passages(paper: Dict, retrieved: str) -> List[Dict]:
    url = paper.get('pdf_url')
    if not url:
        raise LookupError('no open PDF URL')
    body = _get(url, CACHE / 'papers' / f"{_key(paper['doi'])}.pdf")
    return [{'source': 'G', 'kind': 'paper', 'title': paper.get('title', paper['doi']),
             'section': '', 'url': url, 'doi': paper['doi'], 'licence': paper['licence'],
             'retrieved': retrieved, 'text': text}
            for text in chunk(paper_paragraphs(pdf_text(body)))]


def train_papers(path: Path = PAPERS) -> List[Dict]:
    return [p for p in yaml.safe_load(path.read_text())['papers'] if p['tier'] == 'train']


# ---- build -------------------------------------------------------------------------

def data_dir() -> Path:
    """Where the app reads the corpus: ``$XDG_DATA_HOME/FrictionSim2D/knowledge``."""
    base = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(base) / 'FrictionSim2D' / 'knowledge'


def build(only: Optional[str] = None) -> Dict:
    retrieved = dt.date.today().isoformat()
    passages: List[Dict] = []
    failures: List[Dict] = []
    jobs = []
    if only in (None, 'wikipedia'):
        jobs += [('wikipedia', t, lambda t=t: wikipedia_passages(t, retrieved))
                 for t in read_titles()]
    if only in (None, 'papers'):
        jobs += [('paper', p['doi'], lambda p=p: paper_passages(p, retrieved))
                 for p in train_papers()]
    for i, (kind, name, job) in enumerate(jobs, 1):
        try:
            got = job()
        except (IOError, LookupError, ValueError, KeyError) as err:
            failures.append({'kind': kind, 'source': name, 'reason': str(err)[:200]})
            got = []
        except Exception as err:  # pylint: disable=broad-except  # a bad PDF must not stop the build
            failures.append({'kind': kind, 'source': name,
                             'reason': f'{type(err).__name__}: {str(err)[:160]}'})
            got = []
        for n, passage in enumerate(got):
            passage['id'] = f"{kind}:{_key(name)}:{n:03d}"
        passages.extend(got)
        if sys.stderr.isatty():
            print(f'\r{i}/{len(jobs)}', end='', file=sys.stderr)
    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / 'passages.jsonl').open('w', encoding='utf-8') as out:
        for passage in passages:
            out.write(json.dumps(passage, ensure_ascii=False) + '\n')
    sources = {(p['kind'], p.get('doi') or p['title']) for p in passages}
    manifest = {
        'built': retrieved, 'passages': len(passages),
        'sources': {kind: sum(k == kind for k, _ in sources) for kind in ('wikipedia', 'paper')},
        'words': sum(len(p['text'].split()) for p in passages),
        'failures': failures}
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--only', choices=('wikipedia', 'papers'))
    parser.add_argument('--install', action='store_true',
                        help=f'copy passages.jsonl to {data_dir()}')
    args = parser.parse_args(argv)
    manifest = build(args.only)
    print(f"{manifest['passages']} passages, {manifest['words']} words from "
          f"{manifest['sources']}; {len(manifest['failures'])} sources failed "
          f"(corpus/manifest.json)")
    if args.install:
        target = data_dir()
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_DIR / 'passages.jsonl', target / 'passages.jsonl')
        shutil.copy2(OUT_DIR / 'manifest.json', target / 'manifest.json')
        print(f'installed to {target}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
