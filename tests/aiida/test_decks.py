"""Deck discovery -- aiida-free."""
from types import SimpleNamespace

import pytest

from src.aiida.decks import deck_log, discover_decks, points_for
from src.run.tree import discover_runs
from tests.helpers.results_tree import make_family_run


def _locator(root, family, **kw):
    run_dir = make_family_run(root, family, **kw)
    (loc,) = [r for r in discover_runs(root) if r.run_dir == run_dir]
    return loc


def _touch(path, text=''):
    path.write_text(text or '# deck\n', encoding='utf-8')
    return path


def test_afm_decks_and_filters(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide_f30N.in')
    _touch(loc.lammps_dir / 'slide_f10N.in')
    decks = discover_decks(loc)
    assert [d.name for d in decks] == ['system', 'slide_f10N', 'slide_f30N']
    assert [d.point_filter for d in decks] == [{}, {'load': 10.0}, {'load': 30.0}]
    assert [d.writes_structures for d in decks] == [True, False, False]


def test_sheet_pressure_deck(tmp_path):
    loc = _locator(tmp_path, 'sheetonsheet')
    _touch(loc.lammps_dir / 'slide_p0p5gpa.in')
    deck = [d for d in discover_decks(loc) if d.name == 'slide_p0p5gpa'][0]
    assert deck.point_filter == {'load': 0.5}


@pytest.mark.parametrize('stem, expected', [
    ('slide_a90', {'angle': 90.0}),
    ('slide_am45', {'angle': -45.0}),
    ('slide_a22p5', {'angle': 22.5}),
    ('slide_2ms', {'speed': 2.0}),
    ('slide_0p5ms', {'speed': 0.5}),
    ('slide_p1gpa', {'load': 1.0}),
    ('slide_refine', {}),
    ('slide', {}),
])
def test_filter_tokens(tmp_path, stem, expected):
    loc = _locator(tmp_path, 'sheetonsheet')
    _touch(loc.lammps_dir / f'{stem}.in')
    deck = [d for d in discover_decks(loc) if d.name == stem][0]
    assert deck.point_filter == expected


def test_pes_sheet_refine(tmp_path):
    loc = _locator(tmp_path, 'pes_sheet')
    _touch(loc.lammps_dir / 'slide.in')
    _touch(loc.lammps_dir / 'slide_refine.in')
    decks = discover_decks(loc)
    assert [d.name for d in decks] == ['system', 'slide', 'slide_refine']
    assert decks[2].point_filter == {}


def test_points_for(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide_f30N.in')
    _touch(loc.lammps_dir / 'slide_a90.in')
    pts = [SimpleNamespace(load=10.0, angle=0.0, speed=2.0),
           SimpleNamespace(load=30.0, angle=0.0, speed=2.0),
           SimpleNamespace(load=30.0, angle=90.0, speed=2.0)]
    by_name = {d.name: d for d in discover_decks(loc)}
    assert points_for(by_name['system'], pts) == pts
    assert points_for(by_name['slide_f30N'], pts) == pts[1:]
    assert points_for(by_name['slide_a90'], pts) == pts[2:]


def test_log_array_layout(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide_f30N.in')
    a = _touch(loc.lammps_dir / 'run_system.log')
    b = _touch(loc.lammps_dir / 'run_slide_f30N.log')
    _touch(loc.lammps_dir / 'run_slide.log')   # combined name must lose
    assert deck_log(loc, 'slide_f30N') == b
    assert deck_log(loc, 'system') == a


def test_log_combined_layout(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide.in')
    s = _touch(loc.lammps_dir / 'run_system.log')
    sl = _touch(loc.lammps_dir / 'run_slide.log')
    assert deck_log(loc, 'system') == s
    assert deck_log(loc, 'slide') == sl


def test_log_local_layout(tmp_path):
    loc = _locator(tmp_path, 'sheetonsheet')
    (loc.lammps_dir / 'system.in').unlink()
    _touch(loc.lammps_dir / 'slide.in')
    log = _touch(loc.run_dir / 'log_slide.lammps')
    assert deck_log(loc, 'slide') == log
    assert [d.log_path for d in discover_decks(loc)] == [log]


def test_local_log_ignored_with_several_decks(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide.in')
    _touch(loc.run_dir / 'log_slide.lammps')
    assert deck_log(loc, 'slide') is None


def test_no_log(tmp_path):
    loc = _locator(tmp_path, 'afm')
    assert deck_log(loc, 'system') is None


def test_sync_conflict_copies_ignored(tmp_path):
    loc = _locator(tmp_path, 'afm')
    _touch(loc.lammps_dir / 'slide.in')
    _touch(loc.lammps_dir / 'slide.sync-conflict-20261014-173634-U5EC3BU.in')
    _touch(loc.lammps_dir / 'system.sync-conflict-20261014-173634-U5EC3BU.in')
    _touch(loc.lammps_dir / 'run_slide.sync-conflict-20261014-173634-U5EC3BU.log')
    sl = _touch(loc.lammps_dir / 'run_slide.log')
    assert [d.name for d in discover_decks(loc)] == ['system', 'slide']
    assert deck_log(loc, 'slide') == sl
    # a conflicted local log must not count either
    only = _locator(tmp_path / 'b', 'sheetonsheet')
    (only.lammps_dir / 'system.in').unlink()
    _touch(only.lammps_dir / 'slide.in')
    _touch(only.run_dir / 'log_slide.sync-conflict-1.lammps')
    assert deck_log(only, 'slide') is None
