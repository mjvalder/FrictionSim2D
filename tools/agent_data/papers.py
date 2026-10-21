"""Classify the bibliography's papers by licence for the assistant training corpus.

Reads every ``*.bib`` under ``sources/bibliography/``, deduplicates by DOI, asks
OpenAlex for each work's open-access locations and licences, and writes
``sources/papers.yaml`` with one record per paper and its training tier:

- ``train``    -- CC-BY, CC-BY-SA, CC0 or public domain: full text may be used.
- ``nc``       -- CC-BY-NC(-SA): usable only while the model stays non-commercial.
- ``excluded`` -- any -ND licence (no derivatives) or a publisher-specific licence.
- ``closed``   -- no open licence found: bibliographic metadata only.

Spec: docs/superpowers/specs/TO-DO/2026-10-16-assistant-lora-finetune-design.md, §2 (source G).

    python tools/agent_data/papers.py            # uses the response cache
    python tools/agent_data/papers.py --refresh  # re-queries OpenAlex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
BIB_DIR = HERE / 'sources' / 'bibliography'
CACHE_DIR = HERE / '.cache' / 'openalex'
OUTPUT = HERE / 'sources' / 'papers.yaml'

OPENALEX = 'https://api.openalex.org/works/doi:'
USER_AGENT = 'FrictionSim2D-agent-data/0.1 (licence classification of a private bibliography)'
REQUEST_GAP = 0.15  # seconds between requests; OpenAlex allows 10/s

TRAIN = {'cc-by', 'cc-by-sa', 'cc0', 'public-domain'}
NONCOMMERCIAL = {'cc-by-nc', 'cc-by-nc-sa'}
TIER_RANK = {'train': 0, 'nc': 1, 'excluded': 2, 'closed': 3}

_FIELDS = ('doi', 'title', 'year', 'journal', 'author')
_SUFFIX = re.compile(r'/(figures|tables|supplemental)/.*$', re.IGNORECASE)


def iter_entries(text: str) -> Iterator[Dict[str, str]]:
    """Yield ``{'key': ..., field: value}`` for each ``@type{key, ...}`` entry.

    Brace-counting rather than a regex: abstracts contain nested braces."""
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        depth, pos = 1, match.end()
        while depth and pos < len(text):
            depth += {'{': 1, '}': -1}.get(text[pos], 0)
            pos += 1
        body = text[match.end():pos - 1]
        entry = {'key': match.group(2)}
        for name in _FIELDS:
            value = _field(body, name)
            if value:
                entry[name] = value
        yield entry


def _field(body: str, name: str) -> Optional[str]:
    """The brace- or quote-delimited value of ``name`` in an entry body."""
    match = re.search(rf'(?im)^\s*{name}\s*=\s*([{{"])', body)
    if not match:
        return None
    start, close = match.end(), '}' if match.group(1) == '{' else '"'
    depth, pos = 1, start
    while depth and pos < len(body):
        char = body[pos]
        if close == '}':
            depth += {'{': 1, '}': -1}.get(char, 0)
        elif char == '"':
            depth = 0
        pos += 1
    return re.sub(r'\s+', ' ', body[start:pos - 1].replace('{', '').replace('}', '')).strip()


def normalise_doi(doi: str) -> str:
    """Lower-case bare DOI: no resolver prefix, no APS ``/figures/...`` tail."""
    doi = re.sub(r'^(https?://(dx\.)?doi\.org/|doi:)', '', doi.strip(), flags=re.IGNORECASE)
    return _SUFFIX.sub('', doi).lower()


def load_bibliography(folder: Path) -> List[Dict[str, str]]:
    """Every entry with a DOI, first occurrence wins, with the files it came from."""
    papers: Dict[str, Dict[str, str]] = {}
    for path in sorted(folder.glob('*.bib')):
        for entry in iter_entries(path.read_text(encoding='utf-8', errors='replace')):
            if 'doi' not in entry:
                continue
            doi = normalise_doi(entry['doi'])
            record = papers.setdefault(doi, {**entry, 'doi': doi, 'bib_files': []})
            if path.name not in record['bib_files']:
                record['bib_files'].append(path.name)
    return list(papers.values())


def fetch_work(doi: str, refresh: bool = False) -> Optional[dict]:
    """The OpenAlex work for ``doi`` (cached on disk); None if OpenAlex lacks it."""
    cache = CACHE_DIR / (urllib.parse.quote(doi, safe='') + '.json')
    if cache.exists() and not refresh:
        return json.loads(cache.read_text()) or None
    request = urllib.request.Request(OPENALEX + urllib.parse.quote(doi, safe='/'),
                                     headers={'User-Agent': USER_AGENT})
    time.sleep(REQUEST_GAP)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            work = json.load(response)
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
        work = {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(work))
    return work or None


def tier_of(licence: Optional[str]) -> str:
    """The training tier a single licence string allows."""
    if not licence:
        return 'closed'
    if licence in TRAIN:
        return 'train'
    if licence in NONCOMMERCIAL:
        return 'nc'
    return 'excluded'


def classify(work: Optional[dict]) -> Dict[str, object]:
    """The most permissive location of ``work``: its tier, licence and URLs."""
    if not work:
        return {'tier': 'closed', 'licence': None, 'oa_status': 'unknown'}
    ranked = [(tier_of(loc.get('license')), loc) for loc in work.get('locations') or []]
    tier, location = min(ranked, key=lambda pair: TIER_RANK[pair[0]], default=('closed', {}))
    source = (location.get('source') or {}).get('display_name')
    # Every copy under a licence of the same tier, best location first: a
    # repository or PMC copy when the publisher refuses scripted downloads.
    same_tier = [loc for t, loc in ranked if t == tier and tier != 'closed']
    same_tier.sort(key=lambda loc: loc is not location)
    return {
        'tier': tier,
        'licence': location.get('license'),
        'version': location.get('version'),
        'host': source,
        'pdf_url': location.get('pdf_url') if tier != 'closed' else None,
        'landing_url': location.get('landing_page_url') if tier != 'closed' else None,
        'copies': [{'pdf_url': loc.get('pdf_url'), 'landing_url': loc.get('landing_page_url'),
                    'licence': loc.get('license'),
                    'host': (loc.get('source') or {}).get('display_name')}
                   for loc in same_tier if loc.get('pdf_url') or loc.get('landing_page_url')]
        or None,
        'oa_status': (work.get('open_access') or {}).get('oa_status'),
        'openalex_id': work.get('id'),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--refresh', action='store_true', help='ignore the response cache')
    args = parser.parse_args(argv)

    papers = load_bibliography(BIB_DIR)
    for i, paper in enumerate(papers, 1):
        paper.update({k: v for k, v in classify(fetch_work(paper['doi'], args.refresh)).items()
                      if v is not None})
        if sys.stderr.isatty():
            print(f'\r{i}/{len(papers)}', end='', file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)

    papers.sort(key=lambda p: (TIER_RANK[p['tier']], p['doi']))
    counts = {tier: sum(p['tier'] == tier for p in papers) for tier in TIER_RANK}
    OUTPUT.write_text(yaml.safe_dump(
        {'generated_by': 'tools/agent_data/papers.py', 'source': 'OpenAlex locations',
         'counts': counts, 'papers': papers},
        sort_keys=False, allow_unicode=True, width=100))
    print(f'{len(papers)} papers -> {OUTPUT.relative_to(HERE.parent.parent)}: {counts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
