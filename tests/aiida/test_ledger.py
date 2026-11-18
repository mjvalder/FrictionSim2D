"""The ruling ledger: which ruling should be in force, and how a source's
rulings file updates it. Plain dicts only -- no AiiDA needed."""
from datetime import datetime, timezone

from src.aiida.ledger import (Clock, conflicts, desired_ruling, drop_source,
                              read_source, ruling_order, stamp, why_in_force)
from src.postprocessing.integrity_rulings import Ruling

OWN, O1, O2 = '/camp/integrity_rulings.yaml', '/rev/o1.yaml', '/rev/o2.yaml'
PID = 'afm/x/L1|force=30|angle=0|speed=2'


def _entry(ruling, seen_at, status='present', note='n'):
    return {'ruling': ruling, 'note': note, 'date': '', 'override': False,
            'seen_at': seen_at, 'status': status}


# --- desired_ruling ------------------------------------------------------------

def test_the_own_file_wins_whenever_it_rules():
    entries = {O1: _entry('pass', '2026-11-18T10'), OWN: _entry('fail', '2026-01-01T00')}
    assert desired_ruling(entries, OWN) == OWN


def test_otherwise_the_most_recently_seen_wins():
    entries = {O1: _entry('pass', '2026-11-18T10'), O2: _entry('fail', '2026-11-18T11')}
    assert desired_ruling(entries, OWN) == O2
    assert ruling_order(entries, OWN) == [O2, O1]


def test_a_missing_source_still_counts():
    entries = {O1: _entry('pass', '2026-11-18T12', status='missing'),
               O2: _entry('fail', '2026-11-18T11')}
    assert desired_ruling(entries, OWN) == O1


def test_a_ruling_that_does_not_apply_gives_way():
    entries = {OWN: _entry('pass', 'a'), O1: _entry('pass', 'b')}
    assert desired_ruling(entries, OWN, applies=lambda s, e: s != OWN) == O1
    assert desired_ruling(entries, OWN, applies=lambda s, e: False) is None


def test_no_entries_no_ruling():
    assert desired_ruling({}, OWN) is None


def test_a_tie_is_broken_by_source_path():
    entries = {O1: _entry('pass', 'same'), O2: _entry('fail', 'same')}
    assert desired_ruling(entries, None) == desired_ruling(dict(reversed(entries.items())),
                                                           None)


# --- conflicts -----------------------------------------------------------------

def test_agreeing_sources_are_no_conflict():
    entries = {OWN: _entry('pass', 'a'), O1: _entry('pass', 'b', note='other note')}
    assert conflicts(entries, OWN, OWN) is None


def test_disagreeing_sources_name_the_one_in_force_and_why():
    entries = {OWN: _entry('fail', 'a'), O1: _entry('pass', 'b'), O2: _entry('fail', 'c')}
    in_force, why, others = conflicts(entries, OWN, OWN)
    assert in_force == (OWN, 'fail') and 'own' in why
    assert others == [(O1, 'pass')]
    entries = {O1: _entry('pass', 'b'), O2: _entry('fail', 'c')}
    in_force, why, others = conflicts(entries, OWN, O2)
    assert in_force == (O2, 'fail') and 'most recently' in why
    assert others == [(O1, 'pass')]


# --- read_source ---------------------------------------------------------------

def _rulings(**by_pid):
    return {pid: Ruling(ruling=r, note='n') for pid, r in by_pid.items()}


def test_reading_a_file_replaces_that_sources_entries():
    clock = Clock('2026-11-18T10:00:00.000000Z')
    ledger, warnings = {}, []
    read_source(ledger, O1, {PID: Ruling(ruling='pass', note='n'), 'p2': Ruling('fail')},
                own=False, now=clock(), warnings=warnings)
    assert set(ledger) == {PID, 'p2'} and ledger[PID][O1]['status'] == 'present'
    first = ledger[PID][O1]['seen_at']
    retracted = read_source(ledger, O1, {PID: Ruling(ruling='pass', note='n')},
                            own=False, now=clock(), warnings=warnings)
    assert retracted == ['p2'] and set(ledger) == {PID}
    assert ledger[PID][O1]['seen_at'] == first            # unchanged content: not re-seen
    read_source(ledger, O1, {PID: Ruling(ruling='pass', note='new note')},
                own=False, now=clock(), warnings=warnings)
    assert ledger[PID][O1]['note'] == 'new note'
    assert ledger[PID][O1]['seen_at'] == first            # a note edit does not re-rank
    read_source(ledger, O1, {PID: Ruling(ruling='fail', note='new note')},
                own=False, now=clock(), warnings=warnings)
    assert ledger[PID][O1]['seen_at'] > first             # a new decision: seen now
    second = ledger[PID][O1]['seen_at']
    read_source(ledger, O1, {PID: Ruling(ruling='fail', note='n', override=True)},
                own=False, now=clock(), warnings=warnings)
    assert ledger[PID][O1]['seen_at'] > second            # override changed: seen now
    assert warnings == []


def test_reading_keeps_the_in_force_mark():
    ledger = {PID: {O1: {**_entry('pass', 'a'), 'in_force': True}}}
    read_source(ledger, O1, {PID: Ruling(ruling='pass', note='n')}, own=False, now='b',
                warnings=[])
    assert ledger[PID][O1]['in_force'] is True


def test_a_missing_source_is_marked_and_kept():
    ledger, warnings = {}, []
    read_source(ledger, O1, {PID: Ruling(ruling='pass')}, own=False, now='a',
                warnings=warnings)
    assert read_source(ledger, O1, None, own=False, now='b', warnings=warnings) == []
    assert ledger[PID][O1]['status'] == 'missing' and ledger[PID][O1]['seen_at'] == 'a'
    read_source(ledger, O1, {PID: Ruling(ruling='pass')}, own=False, now='c',
                warnings=warnings)
    assert ledger[PID][O1]['status'] == 'present' and ledger[PID][O1]['seen_at'] == 'a'


def test_an_unreadable_file_changes_nothing():
    ledger, warnings = {}, []
    read_source(ledger, O1, {PID: Ruling(ruling='pass')}, own=False, now='a',
                warnings=warnings)
    before = {k: dict(v) for k, v in ledger.items()}
    read_source(ledger, O1, {'<file>': Ruling(error='cannot read')}, own=False, now='b',
                warnings=warnings)
    assert {k: dict(v) for k, v in ledger.items()} == before
    assert any(O1 in w and 'cannot read' in w for w in warnings)


def test_a_malformed_entry_is_skipped_and_the_rest_counts():
    ledger, warnings = {}, []
    read_source(ledger, O1, {'p2': Ruling(ruling='fail')}, own=False, now='a',
                warnings=warnings)
    read_source(ledger, O1, {PID: Ruling(ruling='pass'),
                             '<malformed #1>': Ruling(error='entry 1: not a mapping')},
                own=False, now='b', warnings=warnings)
    assert ledger[PID][O1]['ruling'] == 'pass'
    assert 'p2' in ledger                       # not withdrawn on a partial read
    assert any('entry 1' in w for w in warnings)
    assert any('no ruling of' in w and O1 in w for w in warnings)


def test_drop_source_lists_the_points():
    ledger = {PID: {O1: _entry('pass', 'a'), O2: _entry('fail', 'b')},
              'p2': {O1: _entry('pass', 'a')}}
    assert drop_source(ledger, O1) == [PID, 'p2']
    assert ledger == {PID: {O2: _entry('fail', 'b')}}


def test_clock_is_strictly_increasing():
    clock = Clock('2026-11-18T10:00:00.000000Z')
    stamps = [clock() for _ in range(5)]
    assert stamps == sorted(set(stamps)) and len(stamps) == 5


def test_why_in_force():
    entries = {OWN: _entry('pass', 'a'), O1: _entry('fail', '2026-11-18T10')}
    assert 'own' in why_in_force(entries, OWN, OWN)
    assert why_in_force(entries, OWN, O1) == (
        f'most recently seen (2026-11-18T10); {OWN} does not apply')
    assert why_in_force(entries, OWN, None) == 'no ruling applies'
    assert why_in_force({}, OWN, None) == 'no file rules it'


def test_stamp_orders_as_text():
    early = stamp(datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = stamp(datetime(2026, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc))
    assert early < late and early.endswith('Z')
