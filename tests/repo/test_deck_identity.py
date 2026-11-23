"""Every example's rendered decks must match the committed snapshot."""
import pytest

from tests.helpers.deck_snapshot import (
    diff_files, normalise_deck, read_snapshot, render, render_set, stored_form, write_snapshot)

_CASES = render_set()
_IDS = [c.name for c in _CASES]
_CACHE = {}


def _cached(case, factory):
    """One render per case per session, shared by the tests below."""
    if case.name not in _CACHE:
        _CACHE[case.name] = render(case, factory.mktemp(case.name))
    return _CACHE[case.name]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_decks_match_snapshot(case, lammps_replay, tmp_path_factory, request):
    files = _cached(case, tmp_path_factory)
    if request.config.getoption("--update-deck-snapshot"):
        write_snapshot(case, files)
        return
    actual = {rel: stored_form(t) for rel, t in files.items()}
    expected = read_snapshot(case)
    if actual != expected:
        pytest.fail("rendered decks changed:\n" + diff_files(expected, actual), pytrace=False)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_normalise_is_idempotent(case, lammps_replay, tmp_path_factory):
    for rel, text in _cached(case, tmp_path_factory).items():
        assert normalise_deck(text) == text, rel


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_two_renders_identical(case, lammps_replay, tmp_path, tmp_path_factory):
    first = _cached(case, tmp_path_factory)
    assert render(case, tmp_path) == first
