"""Materials Project and 2DMatPedia (MPContribs) lookups for the assistant.

Headless and Qt-free. The network is reached only through injected callables
(``rester_factory``, ``http_get``), so tests pass fakes. Only a formula, a
2DMatPedia id or a structure id is ever sent; the key travels in a header and
is never logged, cached or put in an error message.

``mp_api`` and ``pymatgen`` are optional and imported inside the functions
that need them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

API = 'https://contribs-api.materialsproject.org'
PROJECT = '2dmatpedia'
PAGE_SIZE = 500
#: Most listing pages fetched before giving up (the project has ~6,400 entries).
MAX_PAGES = 100
CACHE_NAME = '2dmatpedia.json'
CACHE_SECONDS = 7 * 24 * 3600
TIMEOUT = 30
MP_API_MISSING = 'Materials Project lookups need the mp-api package (pip install mp-api)'
PYMATGEN_MISSING = 'Materials Project lookups need pymatgen (pip install pymatgen)'
NO_KEY = 'No Materials Project key: set MP_API_KEY or put it in ~/.mp_api_key'

SUMMARY_FIELDS = ['material_id', 'formula_pretty', 'energy_above_hull',
                  'formation_energy_per_atom', 'band_gap', 'is_metal', 'density',
                  'bulk_modulus', 'shear_modulus', 'symmetry']
LISTING_FIELDS = 'identifier,formula,data,structures'
#: MPContribs stores these under unicode keys; plain names for the table.
_DATA_KEYS = {'E': 'E', 'Ed': 'Ed', 'Eˣ': 'Ex', 'ΔE': 'dE', 'Evdw': 'Evdw', 'mu': 'mu'}
#: The 2DMatPedia id is the primary key: ``mp_id`` is shared by many entries.
_DM_ID = re.compile(r'2dm-\d+')
_PLAIN_ID = re.compile(r'^[A-Za-z0-9_-]+$')
_STATUS = re.compile(r'status code (\d{3})')


class MaterialsDBError(RuntimeError):
    """A lookup failed; the message is safe to show to the model."""


class MaterialsDBUnavailable(MaterialsDBError):
    """An optional package the lookup needs is not installed."""


def mp_api_key(settings=None, env: Optional[Mapping[str, str]] = None,
               home: Optional[Path] = None) -> Optional[str]:
    """The Materials Project key: ``$MP_API_KEY``, then ``~/.mp_api_key``, then
    ``PMG_MAPI_KEY`` in ``~/.config/.pmgrc.yaml``. None when there is none, or
    when ``settings`` turns Materials Project access off. Never logged."""
    if settings is not None and not settings.materials_project.enabled:
        return None
    env = os.environ if env is None else env
    home = Path.home() if home is None else Path(home)
    value = (env.get('MP_API_KEY') or '').strip()
    if value:
        return value
    try:
        path = home / '.mp_api_key'
        if path.is_file():
            value = path.read_text(encoding='utf-8').strip()
            if value:
                return value
        path = home / '.config' / '.pmgrc.yaml'
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            value = str(data.get('PMG_MAPI_KEY') or '').strip() if isinstance(data, dict) else ''
            return value or None
    except (OSError, yaml.YAMLError):
        return None
    return None


def default_materials_db(should_stop: Optional[Callable[[], bool]] = None):
    """The real client, as `AgentDeps.materials_db` expects: built per call
    from agent.yaml and the key lookup (``should_stop``: the turn's Stop);
    raises `MaterialsDBError` when off or
    when no key is set."""
    from .settings import load_agent_settings  # pylint: disable=import-outside-toplevel
    settings = load_agent_settings()
    if not settings.materials_project.enabled:
        raise MaterialsDBError('Materials Project lookups are turned off in agent.yaml '
                               '(materials_project.enabled).')
    key = mp_api_key(settings)
    if key is None:
        raise MaterialsDBError(NO_KEY)
    return MPClient(key, should_stop=should_stop)


def reduced_formula(formula: str) -> str:
    """``formula`` as Materials Project's reduced formula (``Mo2S4`` -> ``MoS2``)."""
    try:
        from pymatgen.core import Composition  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise MaterialsDBUnavailable(PYMATGEN_MISSING) from exc
    try:
        return Composition(str(formula).strip()).reduced_formula
    except Exception as exc:  # pylint: disable=broad-except
        raise MaterialsDBError(f'{formula!r} is not a chemical formula') from exc


def _cache_dir() -> Path:
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'FrictionSim2D'


def _default_rester(key: str):
    try:
        from mp_api.client import MPRester  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise MaterialsDBUnavailable(MP_API_MISSING) from exc
    return MPRester(api_key=key, mute_progress_bars=True)


def _default_http_get(url: str, params=None, headers=None):
    """``httpx.get``; any httpx failure becomes a `ConnectionError` naming
    only its type (so no URL or header reaches a message)."""
    import httpx  # pylint: disable=import-outside-toplevel
    try:
        return httpx.get(url, params=params, headers=headers, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        raise ConnectionError(type(exc).__name__) from exc


def _status_error(status: int) -> MaterialsDBError:
    if status in (401, 403):
        return MaterialsDBError('Materials Project key rejected')
    if status == 429:
        return MaterialsDBError('Materials Project rate-limited, retry later')
    return MaterialsDBError(f'Materials Project error (HTTP {status})')


def _get(doc: Any, name: str) -> Any:
    return doc.get(name) if isinstance(doc, dict) else getattr(doc, name, None)


def _modulus(value: Any) -> Optional[float]:
    """Bulk/shear moduli come as ``{'voigt', 'reuss', 'vrh'}``: vrh, else voigt."""
    if isinstance(value, dict):
        value = value.get('vrh', value.get('voigt'))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _summary_row(doc: Any) -> Dict[str, Any]:
    symmetry = _get(doc, 'symmetry')
    return {'material_id': str(_get(doc, 'material_id')),
            'formula': _get(doc, 'formula_pretty'),
            'e_above_hull': _get(doc, 'energy_above_hull'),
            'formation_energy': _get(doc, 'formation_energy_per_atom'),
            'band_gap': _get(doc, 'band_gap'), 'is_metal': _get(doc, 'is_metal'),
            'density': _get(doc, 'density'),
            'bulk_modulus': _modulus(_get(doc, 'bulk_modulus')),
            'shear_modulus': _modulus(_get(doc, 'shear_modulus')),
            'spacegroup': _get(symmetry, 'symbol') if symmetry is not None else None}


def _flatten(contribution: dict) -> Dict[str, Any]:
    """One MPContribs 2DMatPedia contribution as plain scalar columns."""
    data = contribution.get('data') or {}
    details = data.get('details')
    match = _DM_ID.search(details) if isinstance(details, str) else None
    structures = contribution.get('structures') or []
    row = {'dm_id': match.group(0) if match else None,
           'mp_id': contribution.get('identifier'),
           'formula': contribution.get('formula'),
           'structure_id': structures[0].get('id') if structures else None,
           'process': data.get('process')}
    for source, name in _DATA_KEYS.items():
        value = data.get(source)
        row[name] = value.get('value') if isinstance(value, dict) else None
    try:
        row['reduced'] = reduced_formula(row['formula'] or '')
    except MaterialsDBUnavailable:
        raise
    except MaterialsDBError:
        row['reduced'] = row['formula']
    return row


class MPClient:
    """Materials Project summaries and the 2DMatPedia project on MPContribs."""

    def __init__(self, key: str, rester_factory: Optional[Callable[[str], Any]] = None,
                 http_get: Optional[Callable[..., Any]] = None,
                 cache_dir: Optional[Path] = None,
                 clock: Callable[[], float] = time.time,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
        self._key = key
        self._should_stop = should_stop or (lambda: False)
        #: Whether the last `listing` call downloaded (rather than read the cache).
        self.downloaded_listing = False
        self._rester_factory = rester_factory or _default_rester
        self._http_get = http_get or _default_http_get
        self._cache = Path(cache_dir or _cache_dir()) / CACHE_NAME
        self._clock = clock

    def __repr__(self) -> str:  # never show the key
        return 'MPClient()'

    # -- Materials Project ---------------------------------------------------

    def summary(self, formula: str) -> List[Dict[str, Any]]:
        """Summary rows for ``formula`` (reduced), most stable first."""
        reduced = reduced_formula(formula)
        try:
            with self._rester_factory(self._key) as rester:
                docs = rester.materials.summary.search(formula=[reduced],
                                                       fields=SUMMARY_FIELDS)
        except MaterialsDBError:
            raise
        except OSError as exc:
            raise MaterialsDBError('Materials Project unreachable') from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise self._rester_error(exc) from exc
        rows = [_summary_row(doc) for doc in docs]
        return sorted(rows, key=lambda r: (r['e_above_hull'] is None,
                                           r['e_above_hull'] or 0.0))

    def _rester_error(self, exc: Exception) -> MaterialsDBError:
        text = str(exc)
        match = _STATUS.search(text)
        if match:
            return _status_error(int(match.group(1)))
        lowered = text.lower()
        if 'api key' in lowered or 'api_key' in lowered:
            return MaterialsDBError('Materials Project key rejected')
        if any(word in lowered for word in ('connection', 'max retries', 'timed out')):
            return MaterialsDBError('Materials Project unreachable')
        return MaterialsDBError(f'Materials Project query failed ({type(exc).__name__})')

    # -- MPContribs ------------------------------------------------------------

    def _check_stop(self) -> None:
        if self._should_stop():
            raise MaterialsDBError('stopped')

    def _get_json(self, url: str, params: Dict[str, Any]) -> Any:
        self._check_stop()
        try:
            response = self._http_get(url, params=params, headers={'X-API-KEY': self._key})
        except OSError as exc:  # ConnectionError, TimeoutError, ...
            raise MaterialsDBError('Materials Project unreachable') from exc
        status = int(response.status_code)
        if status >= 400:
            raise _status_error(status)
        try:
            return response.json()
        except ValueError as exc:
            raise MaterialsDBError('Materials Project sent an unreadable reply') from exc

    def _fetch_listing(self) -> List[Dict[str, Any]]:
        """Every 2DMatPedia contribution, paged with ``_skip`` (the server
        ignores ``_page`` and would return page 1 forever)."""
        rows: List[Dict[str, Any]] = []
        for page in range(MAX_PAGES):
            body = self._get_json(f'{API}/contributions/', {
                'project': PROJECT, '_limit': PAGE_SIZE, '_skip': page * PAGE_SIZE,
                '_fields': LISTING_FIELDS})
            if not isinstance(body, dict):
                raise MaterialsDBError('Materials Project sent an unexpected listing')
            data = body.get('data') or []
            rows.extend(_flatten(c) for c in data if isinstance(c, dict))
            if not body.get('has_more') or not data:
                return rows
        raise MaterialsDBError(f'2DMatPedia listing did not end after {MAX_PAGES} pages')

    def _read_cache(self) -> Optional[List[Dict[str, Any]]]:
        try:
            payload = json.loads(self._cache.read_text(encoding='utf-8'))
            if self._clock() - float(payload['fetched']) < CACHE_SECONDS:
                return list(payload['rows'])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _write_cache(self, rows: List[Dict[str, Any]]) -> None:
        try:
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            partial = self._cache.with_name(f'.{CACHE_NAME}.{os.getpid()}')
            partial.write_text(json.dumps({'fetched': self._clock(), 'rows': rows}),
                               encoding='utf-8')
            os.replace(partial, self._cache)
        except OSError as exc:
            logger.info('could not cache the 2DMatPedia listing: %s', type(exc).__name__)

    def listing(self) -> List[Dict[str, Any]]:
        """The flattened 2DMatPedia listing, cached for a week."""
        self.downloaded_listing = False
        rows = self._read_cache()
        if rows is None:
            rows = self._fetch_listing()
            self._write_cache(rows)
            self.downloaded_listing = True
        return rows

    def twodmatpedia(self, formula: str) -> List[Dict[str, Any]]:
        """2DMatPedia entries whose reduced formula is that of ``formula``."""
        reduced = reduced_formula(formula)
        return [row for row in self.listing() if row.get('reduced') == reduced]

    def twodmatpedia_entry(self, dm_id: str) -> Optional[Dict[str, Any]]:
        """The entry with 2DMatPedia id ``dm_id`` (``2dm-1234``), or None."""
        dm_id = str(dm_id).strip()
        return next((row for row in self.listing() if row.get('dm_id') == dm_id), None)

    def structure_cif(self, structure_id: str) -> str:
        """CIF text of an MPContribs structure, built from lattice and sites
        (``Structure.from_dict`` fails: the payload has no @module/@class)."""
        if not _PLAIN_ID.match(str(structure_id)):
            raise MaterialsDBError(f'{structure_id!r} is not a structure id')
        payload = self._get_json(f'{API}/structures/{structure_id}/', {'_fields': '_all'})
        try:
            # pylint: disable=import-outside-toplevel
            from pymatgen.core import Lattice, Structure
            from pymatgen.io.cif import CifWriter
        except ImportError as exc:
            raise MaterialsDBUnavailable(PYMATGEN_MISSING) from exc
        try:
            sites = payload['sites']
            if not sites:
                raise ValueError('no sites')
            structure = Structure(Lattice(payload['lattice']['matrix']),
                                  [site['species'][0]['element'] for site in sites],
                                  [site['abc'] for site in sites])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise MaterialsDBError('unexpected structure payload from MPContribs') from exc
        return str(CifWriter(structure))
