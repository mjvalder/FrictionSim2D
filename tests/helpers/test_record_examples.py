"""Record (``--record-lammps -m lammps``) and replay the example builds' LAMMPS calls.

Task 2 owns the full render set; this covers what ``examples/*.ini`` trigger.
"""
import pytest

from tests.helpers.deck_snapshot import render, render_set
from tests.helpers.examples import CASES, HETERO, build

_CASES = CASES + [HETERO]


@pytest.mark.lammps
@pytest.mark.parametrize("name,model,_n", _CASES, ids=[c[0] for c in _CASES])
def test_record_example(name, model, _n, lammps_replay, tmp_path, request):
    if not request.config.getoption("--record-lammps"):
        pytest.skip("recording needs --record-lammps")
    build(name, model, tmp_path)


@pytest.mark.parametrize("name,model,_n", _CASES, ids=[c[0] for c in _CASES])
def test_replay_example(name, model, _n, lammps_replay, tmp_path, request):
    if request.config.getoption("--record-lammps"):
        pytest.skip("recording run")
    build(name, model, tmp_path)


@pytest.mark.lammps
@pytest.mark.parametrize("case", render_set(), ids=lambda c: c.name)
def test_record_render_set(case, lammps_replay, tmp_path, request):
    """Records everything the deck-identity render set (incl. campaigns) calls."""
    if not request.config.getoption("--record-lammps"):
        pytest.skip("recording needs --record-lammps")
    render(case, tmp_path)
