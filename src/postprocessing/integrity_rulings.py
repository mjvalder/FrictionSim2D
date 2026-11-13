"""User rulings for integrity-quarantined points.

`<results_dir>/integrity_rulings.yaml` holds one entry per ruled point::

    - point: "afm/.../L1|force=30|angle=0|speed=2"
      ruling: pass            # or fail
      note: looked at the snapshot, the hole is a viewer artefact
      date: 2026-11-12

A ruling only decides a `quarantined` point. On any other status, or on a
point that does not exist, it changes nothing and yields one warning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

RULINGS_FILE = 'integrity_rulings.yaml'
_VALID = ('pass', 'fail')
_ID_RE = re.compile(
    r'^(?P<rel>[^|]+)\|(?P<kind>[A-Za-z]+)=(?P<load>[^|]+)'
    r'\|angle=(?P<angle>[^|]+)\|speed=(?P<speed>[^|]+)$')


@dataclass(frozen=True)
class Ruling:
    """One entry. A non-empty `error` marks an entry that could not be read."""
    ruling: str = ''
    note: str = ''
    date: str = ''
    error: str = ''


def _num(value) -> str:
    """Canonical number text: `30.0` and `30` both give `30`, `0.5` stays."""
    text = repr(float(value) + 0.0)          # +0.0 turns -0.0 into 0.0
    return text[:-2] if text.endswith('.0') else text


def format_point_id(rel_path: str, load_kind: str, load: float,
                    angle: float, speed: float) -> str:
    """`<rel path>|<load_kind>=<load>|angle=<a>|speed=<s>`, numbers canonical."""
    return (f'{rel_path}|{load_kind}={_num(load)}'
            f'|angle={_num(angle)}|speed={_num(speed)}')


def point_id(point, root: Path) -> str:
    """The ruling key of a `RunFrames`: its run's path under `root`."""
    rel = Path(point.locator.run_dir).relative_to(root).as_posix()
    return format_point_id(rel, point.load_kind, point.load, point.angle,
                           point.speed)


def _canonical(text: str) -> str:
    """Re-spell a hand-written id so it matches `point_id`; ValueError if bad."""
    match = _ID_RE.match(text.strip())
    if match is None:
        raise ValueError(f'unparseable point id {text!r}')
    return format_point_id(match['rel'].strip(), match['kind'],
                           float(match['load']), float(match['angle']),
                           float(match['speed']))


def _entry(raw) -> tuple:
    if not isinstance(raw, dict):
        raise ValueError('entry is not a mapping')
    key = _canonical(str(raw.get('point', '')))
    ruling = str(raw.get('ruling', '')).strip().lower()
    if ruling not in _VALID:
        raise ValueError(f'ruling must be pass or fail, got {raw.get("ruling")!r}')
    return key, Ruling(ruling=ruling, note=str(raw.get('note') or ''),
                       date=str(raw.get('date') or ''))


def load_rulings(root: Path) -> Dict[str, Ruling]:
    """Rulings by canonical point id; `{}` when there is no file.

    Entries that cannot be read are kept under `<malformed #n>` (or `<file>`)
    with `error` set, so `apply_rulings` reports each one; nothing raises.
    """
    path = Path(root) / RULINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return {'<file>': Ruling(error=f'cannot read {RULINGS_FILE}: {exc}')}
    if data is None:
        return {}
    if not isinstance(data, list):
        return {'<file>': Ruling(
            error=f'{RULINGS_FILE}: top level must be a list of entries')}
    out: Dict[str, Ruling] = {}
    for i, raw in enumerate(data):
        try:
            key, ruling = _entry(raw)
        except (ValueError, TypeError) as exc:
            out[f'<malformed #{i}>'] = Ruling(error=f'entry {i}: {exc}')
            continue
        out[key] = ruling
    return out


def apply_rulings(points: Iterable, rulings: Dict[str, Ruling],
                  root: Path) -> List[str]:
    """Apply rulings to `quarantined` points in place; return the warnings.

    `pass` -> `complete`, `fail` -> `failed_integrity`; the verdict gains
    `ruled_by_user`. One warning per ruling that could not apply.
    """
    warnings: List[str] = []
    by_id = {point_id(p, root): p for p in points}
    for key, ruling in rulings.items():
        if ruling.error:
            warnings.append(f'integrity ruling ignored: {ruling.error}')
            continue
        point = by_id.get(key)
        if point is None:
            warnings.append(f'integrity ruling ignored: no such point {key}')
            continue
        if point.status != 'quarantined':
            warnings.append(f'integrity ruling ignored: {key} is '
                            f'{point.status}, not quarantined')
            continue
        status = 'complete' if ruling.ruling == 'pass' else 'failed_integrity'
        text = f'ruled {ruling.ruling} by user {ruling.date}: {ruling.note}'
        point.status = status
        point.status_detail = '; '.join(
            part for part in (point.status_detail, text) if part)
        if point.integrity is not None:
            point.integrity = replace(
                point.integrity, status=status, failed=status != 'complete',
                ruled_by_user=True)
    return warnings
