"""Materials Project and 2DMatPedia: look up a formula, fetch a 2D structure.

Only formulas and ids leave the machine. Failures (no key, key rejected,
rate limit, no network) come back as plain text for the model to relay.
Fetching a structure proposes a new CIF in the workspace; it never writes.
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Optional

from ..changes import ToolInputError, build_fs_changeset
from ..materials_db import MaterialsDBError
from . import compact

MAX_ROWS = 10
_DM_ID = re.compile(r'^2dm-\d+$')


def _num(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}'
    return '-'


def _with_client(deps, action: Callable[[Any], str]) -> str:
    """Run ``action(client)``; a missing factory or a lookup failure is text."""
    factory: Optional[Callable[[], Any]] = getattr(deps, 'materials_db', None)
    if factory is None:
        return 'Materials Project lookups are not available here.'
    try:
        return action(factory())
    except MaterialsDBError as exc:
        return str(exc)


def _more(total: int) -> List[str]:
    return [f'... {total - MAX_ROWS} more'] if total > MAX_ROWS else []


def query_materials_project(deps, formula: str) -> str:
    """Materials Project entries for a formula, most stable first."""
    def run(client) -> str:
        rows = client.summary(formula)
        if not rows:
            return f'no Materials Project entry for {formula}'
        lines = [f'Materials Project, {formula}: {len(rows)} entries (most stable first)',
                 'id | spacegroup | E_hull eV/at | E_form eV/at | gap eV | metal | '
                 'density g/cm3 | K GPa | G GPa']
        for row in rows[:MAX_ROWS]:
            lines.append(' | '.join([
                row['material_id'], str(row.get('spacegroup') or '-'),
                _num(row.get('e_above_hull'), 3), _num(row.get('formation_energy'), 3),
                _num(row.get('band_gap')), _num(row.get('is_metal')),
                _num(row.get('density')), _num(row.get('bulk_modulus'), 0),
                _num(row.get('shear_modulus'), 0)]))
        return compact('\n'.join(lines + _more(len(rows))))
    return _with_client(deps, run)


def query_2dmatpedia(deps, formula: str) -> str:
    """2DMatPedia monolayers for a formula, with their source MP ids."""
    def run(client) -> str:
        rows = client.twodmatpedia(formula)
        if not rows:
            return f'no 2DMatPedia entry for {formula}'
        lines = [f'2DMatPedia, {formula}: {len(rows)} entries '
                 '(fetch one with fetch_2dmatpedia_structure)',
                 'dm_id | from mp_id | process | E eV/at | Ed eV/at | Ex eV/at | dE | Evdw']
        for row in rows[:MAX_ROWS]:
            lines.append(' | '.join([
                str(row.get('dm_id') or '-'), str(row.get('mp_id') or '-'),
                str(row.get('process') or '-'), _num(row.get('E'), 3),
                _num(row.get('Ed'), 3), _num(row.get('Ex'), 3), _num(row.get('dE'), 3),
                _num(row.get('Evdw'), 3)]))
        return compact('\n'.join(lines + _more(len(rows))))
    return _with_client(deps, run)


def fetch_2dmatpedia_structure(deps, dm_id: str, dest_dir: str = 'materials') -> str:
    """Propose writing a 2DMatPedia structure as ``<formula>_<dm_id>.cif``."""
    dm_id = str(dm_id).strip()
    if not _DM_ID.match(dm_id):
        raise ToolInputError(f'{dm_id!r} is not a 2DMatPedia id (2dm-<number>); '
                             'use one from query_2dmatpedia')

    def run(client) -> str:
        entry = client.twodmatpedia_entry(dm_id)
        if entry is None or not entry.get('structure_id'):
            return f'no 2DMatPedia entry {dm_id} with a structure'
        text = client.structure_cif(entry['structure_id'])
        formula = re.sub(r'[^A-Za-z0-9]', '', str(entry.get('formula') or '')) or 'structure'
        name = f'{formula}_{dm_id}.cif'
        ops = [('mkdir', dest_dir), ('write_text', f'{dest_dir}/{name}', text)]
        return deps.propose(build_fs_changeset(
            deps, f'Write 2DMatPedia {dm_id} as {dest_dir}/{name}', ops))
    return _with_client(deps, run)
