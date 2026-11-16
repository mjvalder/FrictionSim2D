"""User rulings for integrity-quarantined points.

`<results_dir>/integrity_rulings.yaml` holds one entry per ruled point::

    - point: "afm/.../L1|force=30|angle=0|speed=2"
      ruling: pass            # or fail
      note: looked at the snapshot, the hole is a viewer artefact
      date: 2026-11-12

A ruling decides a `quarantined` point outright. On any other status it needs
`override: true` and a non-empty `note` (the expert may overrule any verdict);
without them, or on a point that does not exist, it changes nothing and yields
one warning. `decide` is the pure decision, shared with other appliers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    override: bool = False
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
    run_dir, base = Path(point.locator.run_dir), Path(root)
    try:
        rel = run_dir.resolve().relative_to(base.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        try:
            rel = run_dir.relative_to(base).as_posix()
        except ValueError:
            rel = run_dir.as_posix()
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
    override = raw.get('override', False)
    if not isinstance(override, bool):
        raise ValueError(f'override must be true or false, got {override!r}')
    return key, Ruling(ruling=ruling, note=str(raw.get('note') or ''),
                       date=str(raw.get('date') or ''), override=override)


def load_rulings(root: Path) -> Dict[str, Ruling]:
    """Rulings by canonical point id; `{}` when there is no file.

    `root` is the directory holding `RULINGS_FILE`, or a rulings file itself.
    Entries that cannot be read are kept under `<malformed #n>` (or `<file>`)
    with `error` set, so `apply_rulings` reports each one; nothing raises.
    """
    path = Path(root) if Path(root).is_file() else Path(root) / RULINGS_FILE
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
    duplicates: List[str] = []
    for i, raw in enumerate(data):
        try:
            key, ruling = _entry(raw)
        except (ValueError, TypeError) as exc:
            out[f'<malformed #{i}>'] = Ruling(error=f'entry {i}: {exc}')
            continue
        if key in out:
            duplicates.append(key)
        out[key] = ruling
    for n, key in enumerate(dict.fromkeys(duplicates)):
        out[f'<duplicate #{n}>'] = Ruling(
            error=f'{key} is ruled more than once; the last entry is used')
    return out


#: Statuses with no frames to accept: a `pass` has nothing to stand on.
_NO_DATA = ('no_output', 'not_reached')


def decide(status: str, ruling: Ruling) -> Tuple[Optional[str], Optional[str]]:
    """What `ruling` does to a point of `status`: `(new_status, warning)`.

    Exactly one is not None. `quarantined` takes either ruling. Any other
    status needs `override` and a non-empty note. `pass` -> `complete`; `fail`
    -> `failed_integrity` on a stage-3 verdict (`complete`, `failed_integrity`)
    and the status unchanged on a gate status. A `pass` on a point with no
    data is refused.
    """
    if status != 'quarantined':
        missing = []
        if not ruling.override:
            missing.append('`override: true`')
        if not ruling.note.strip():
            missing.append('a non-empty `note`')
        if missing:
            return None, (f'{status} point needs {" and ".join(missing)} '
                          'to be overruled')
        if ruling.ruling == 'pass' and status in _NO_DATA:
            return None, f'{status} point has no data to accept as pass'
    if ruling.ruling == 'pass':
        return 'complete', None
    if status in ('quarantined', 'complete', 'failed_integrity'):
        return 'failed_integrity', None
    return status, None


def rule_verdict(verdict, ruling: Ruling):
    """`verdict` (an `IntegrityVerdict`) as `ruling` decides it, or
    `(None, warning)` when the ruling cannot apply (see `decide`).

    The new verdict gains `ruled_by_user`, `original_status`, the ruling's
    `note` and `date` and, when the point was not quarantined, `overridden`.
    `failed` follows the new status, or stays when the status is unchanged.
    """
    original = verdict.status
    status, why = decide(original, ruling)
    if why is not None:
        return None, why
    return replace(
        verdict, status=status,
        failed=verdict.failed if status == original else status != 'complete',
        ruled_by_user=True, overridden=original != 'quarantined',
        original_status=original, note=ruling.note, date=ruling.date), None


def apply_rulings(points: Iterable, rulings: Dict[str, Ruling],
                  root: Path) -> List[str]:
    """Apply rulings to points in place (see `decide`); return the warnings.

    The verdict is replaced by `rule_verdict`'s. One warning per ruling that
    could not apply.
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
        original = point.status
        status, why = decide(original, ruling)
        if why is not None:
            warnings.append(f'integrity ruling ignored: {key}: {why}')
            continue
        if original != 'quarantined':
            text = f'overridden by user {ruling.date}: {ruling.note}'
        else:
            text = f'ruled {ruling.ruling} by user {ruling.date}: {ruling.note}'
        point.status = status
        point.status_detail = '; '.join(
            part for part in (point.status_detail, text) if part)
        if point.integrity is not None:
            # Decided on the point's status, as above, whatever the verdict says.
            point.integrity = rule_verdict(
                replace(point.integrity, status=original), ruling)[0]
    return warnings
