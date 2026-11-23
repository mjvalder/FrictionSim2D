"""Every example's rendered decks must match the committed snapshot."""
import pytest

from tests.helpers.deck_snapshot import (
    diff_files, normalise_deck, read_snapshot, render, render_set, stored_form, write_snapshot)

_CASES = render_set()


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_decks_match_snapshot(case, lammps_replay, tmp_path, request):
    rendered = render(case, tmp_path)
    if request.config.getoption("--update-deck-snapshot"):
        write_snapshot(case, rendered)
        return
    actual = {rel: stored_form(t) for rel, t in rendered.items()}
    expected = read_snapshot(case)
    assert actual == expected, "rendered decks changed:\n" + diff_files(expected, actual)


@pytest.mark.parametrize("case", _CASES[:1], ids=[c.name for c in _CASES[:1]])
def test_normalise_is_idempotent(case, lammps_replay, tmp_path):
    for text in render(case, tmp_path).values():
        assert normalise_deck(text) == text


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_two_renders_identical(case, lammps_replay, tmp_path):
    assert render(case, tmp_path / "a") == render(case, tmp_path / "b")
