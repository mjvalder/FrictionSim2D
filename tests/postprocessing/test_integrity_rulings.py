"""User rulings for quarantined integrity points."""
from __future__ import annotations

import pytest

pytest.importorskip('scipy')

from src.postprocessing.integrity_rulings import (  # noqa: E402
    RULINGS_FILE, apply_rulings, format_point_id, load_rulings, point_id)
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.postprocessing.test_read_data_integrity import (  # noqa: E402
    _afm_tree, _point)


def _quarantined_tree(root):
    """30 N point quarantined (deck writes a snapshot, file absent)."""
    run_dir = _afm_tree(root, loads=(10.0, 30.0))
    for path in (run_dir / 'data').glob('final_f30N*'):
        path.unlink()
    return run_dir


def _write(root, text):
    (root / RULINGS_FILE).write_text(text, encoding='utf-8')


def _ruling(pid, ruling, note='n', date='2026-11-12'):
    return (f'- point: "{pid}"\n  ruling: {ruling}\n  note: {note}\n'
            f'  date: {date}\n')


def _id(root, load, reader=None):
    r = reader or DataReader(str(root))
    return point_id(_point(r, load, points=True), root)


def test_format_point_id_canonical_numbers():
    a = format_point_id('afm/x/L1', 'force', 30.0, 0, 2.0)
    assert a == 'afm/x/L1|force=30|angle=0|speed=2'
    assert format_point_id('afm/x/L1', 'force', 30, 0.0, 2) == a
    assert format_point_id('r', 'pressure', 0.5, -0.0, 1e-5) == \
        'r|pressure=0.5|angle=0|speed=1e-05'


def test_point_id_uses_run_path_under_root(tmp_path):
    run_dir = _quarantined_tree(tmp_path)
    pid = _id(tmp_path, 30.0)
    rel = run_dir.relative_to(tmp_path).as_posix()
    assert pid == f'{rel}|force=30|angle=0|speed=2'


def test_pass_ruling_makes_complete(tmp_path):
    _quarantined_tree(tmp_path)
    base = DataReader(str(tmp_path))
    assert _point(base, 30.0).status == 'quarantined'
    pid = _id(tmp_path, 30.0, base)
    _write(tmp_path, _ruling(pid.replace('=30|', '=30.0|'), 'pass', 'fine'))
    r = DataReader(str(tmp_path))
    p = _point(r, 30.0)
    assert p.status == 'complete' and p.integrity.status == 'complete'
    assert p.integrity.ruled_by_user and not p.integrity.failed
    assert 'ruled pass by user 2026-11-12: fine' in p.status_detail
    assert r.ruling_warnings == []
    assert _point(r, 10.0).status == 'complete'          # untouched


def test_fail_ruling_makes_failed_integrity(tmp_path):
    _quarantined_tree(tmp_path)
    _write(tmp_path, _ruling(_id(tmp_path, 30.0), 'fail'))
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'failed_integrity' and p.integrity.failed
    assert p.integrity.ruled_by_user
    assert 'ruled fail by user 2026-11-12' in p.status_detail


def test_ruling_on_non_quarantined_or_missing_point_warns_once_each(tmp_path):
    _afm_tree(tmp_path, loads=(10.0, 30.0), eject=(30.0,))   # 30 failed, 10 complete
    run_id = _id(tmp_path, 10.0).split('|')[0]
    text = (_ruling(_id(tmp_path, 30.0), 'pass')
            + _ruling(_id(tmp_path, 10.0), 'fail')
            + _ruling(f'{run_id}|force=99|angle=0|speed=2', 'pass'))
    _write(tmp_path, text)
    r = DataReader(str(tmp_path))
    assert _point(r, 30.0).status == 'failed_integrity'
    assert _point(r, 10.0).status == 'complete'
    assert not _point(r, 30.0).integrity.ruled_by_user
    assert len(r.ruling_warnings) == 3


@pytest.mark.parametrize('text', [
    '- point: 5\n  ruling: maybe\n',
    '- just a string\n',
    '- point: "no-bars"\n  ruling: pass\n',
    'not: a list\n',
    'a: [unclosed\n',
])
def test_malformed_entry_warns_without_raising(tmp_path, text):
    _quarantined_tree(tmp_path)
    _write(tmp_path, text)
    r = DataReader(str(tmp_path))
    assert len(r.ruling_warnings) == 1
    assert _point(r, 30.0).status == 'quarantined'


def test_no_file_no_change(tmp_path):
    _quarantined_tree(tmp_path)
    assert load_rulings(tmp_path) == {}
    r = DataReader(str(tmp_path))
    assert r.ruling_warnings == []
    assert _point(r, 30.0).status == 'quarantined'
    assert apply_rulings(r.points, {}, tmp_path) == []
