"""Tiny end-to-end runs with a fake LAMMPS: build -> outputs -> read -> import.

One row per (family, potential, condition) of the matrix,
``tests.helpers.deck_snapshot.SCENARIO_CASES`` (so the deck snapshot pins the
same builds). Each row builds its config in the snapshot's layout (build-time LAMMPS replayed by ``lammps_replay``), lets ``fake_lmp.fake_run`` write every file
every deck would write, then drives the real CLI: ``postprocess read
--export`` must produce ``output_full_*.json`` carrying the family's
observables columns, and ``aiida import`` into ``aiida_profile_clean`` must
give one ``point_table()`` row per read point. The quench row goes ``quench
import`` -> ``find_block`` instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from frictionsim2d import cli as cli_module
from frictionsim2d.core.observables import PES_FAMILIES, rename_map
from frictionsim2d.run.artifacts import generate_package
from tests.helpers.deck_snapshot import SCENARIO_CASES, RenderCase
from tests.scenarios.conftest import tier_elapsed
from tests.scenarios.fake_lmp import fake_run

pytestmark = pytest.mark.scenario

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The friction rows; the quench row has its own test.
SCENARIOS = [c for c in SCENARIO_CASES if c.run_type != 'quench']
QUENCH = next(c for c in SCENARIO_CASES if c.run_type == 'quench')

#: Rows that also go through `aiida import`: one per family (afm, indent,
#: sheetonsheet, hetero, flake, pes_tip, pes_sheet) plus the MLIP row. The
#: others repeat a family's import path, and importing all of them would
#: break the budget (the two hetero builds alone take ~30 s).
AIIDA_ROWS = frozenset({'afm_sw_load', 'indent_sw', 'sheet_reaxff_speed', 'sheet_hetero_twist',
                        'flake_slide_sw', 'pes_tip_sw', 'pes_sheet_ljg', 'sheet_mace_d3'})

#: The tier's wall-time budget, seconds.
BUDGET_S = 60.0

try:
    from frictionsim2d.aiida import AIIDA_AVAILABLE
except ImportError:  # pragma: no cover - the package always ships the flag
    AIIDA_AVAILABLE = False


def _build(case: RenderCase, out: Path) -> Path:
    """Build ``case`` into ``out`` in the deck snapshot's layout (``out/sim``,
    from the repo root), so the build-time LAMMPS it recorded replays; return
    the simulation root. ``render`` itself is not used: it also reads and
    normalises every file, which the budget cannot afford."""
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        result = generate_package(case.ini, case.run_type, output_root=out,
                                  generate_hpc=False, simulation_root_name='sim')
    finally:
        os.chdir(cwd)
    assert not result.failures, f'{case.name}: {result.failures[0].error!r}'
    return out / 'sim'


def _fake_all(root: Path) -> list:
    """``fake_run`` every deck of ``root``, logs where the array job puts them."""
    decks = sorted(root.rglob('lammps/*.in'))
    assert decks, f'no decks under {root}'
    # system decks first: the slide decks read what they write
    decks.sort(key=lambda d: (not d.name.startswith('system'), str(d)))
    for deck in decks:
        fake_run(deck, cwd=root, log=deck.parent / f'run_{deck.stem}.log')
    return decks


def _invoke(*args):
    result = CliRunner().invoke(cli_module.cli, [str(a) for a in args])
    assert result.exit_code == 0, (
        f'{args}\nexit {result.exit_code}\n{result.output}\n{result.exception!r}')
    return result


def test_fake_run_writes_expanded_files(tmp_path):
    deck = tmp_path / 'deck.in'
    deck.write_text(
        'variable p index 1 2\n'
        'fix fc all ave/time 1 10 10 v_fz c_t file out/f_${p}.txt title1 "# schema=2"\n'
        'run 10\n', encoding='utf-8')

    written = fake_run(deck, rows=3, cwd=tmp_path)

    files = [tmp_path / 'out' / 'f_1.txt', tmp_path / 'out' / 'f_2.txt']
    assert [p for p in written if p.suffix == '.txt'] == files
    for path in files:
        lines = path.read_text(encoding='utf-8').splitlines()
        assert lines[:2] == ['# schema=2', '# TimeStep v_fz c_t']
        assert [line.split()[0] for line in lines[2:]] == ['10', '20', '30']
    assert 'Loop time' in (tmp_path / 'log.lammps').read_text(encoding='utf-8')


def test_fake_run_repeats_print_rows_and_never_runs_an_if(tmp_path):
    deck = tmp_path / 'deck.in'
    deck.write_text(
        'variable f equal 5.0\n'
        'print "x,e" file out/g_$(v_f).csv screen no\n'
        'print "$(v_dx),$(v_pe)" append out/g_$(v_f).csv screen no\n'
        'if "${halt} > 0" then &\n'
        '    "print \'# HALT\' append out/halted.txt"\n', encoding='utf-8')

    fake_run(deck, rows=2, cwd=tmp_path)

    lines = (tmp_path / 'out' / 'g_5.csv').read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'x,e' and len(lines) == 3
    assert not (tmp_path / 'out' / 'halted.txt').exists()


@pytest.mark.parametrize('case', SCENARIOS, ids=[c.name for c in SCENARIOS])
def test_scenario(case, lammps_replay, tmp_path, request):
    root = _build(case, tmp_path)
    _fake_all(root)

    result = _invoke('postprocess', 'read', root, '--export')
    exports = sorted((root / 'outputs').glob('output_full_*.json'))
    assert exports, result.output
    entries = [entry for export in exports
               for entry in json.loads(export.read_text(encoding='utf-8'))['runs']]
    assert entries, result.output
    for entry in entries:
        _check_columns(entry)

    if AIIDA_AVAILABLE and case.name in AIIDA_ROWS:
        request.getfixturevalue('scenario_store')
        _invoke('aiida', 'import', root, '-l', case.name)
        from frictionsim2d.aiida.query import point_table  # pylint: disable=import-outside-toplevel
        table = point_table()
        assert len(table) == len(entries), (len(table), len(entries))
        assert sorted(zip(table['load'], table['angle'], table['speed'])) == sorted(
            (e['load'], e['angle'], e['speed']) for e in entries)


#: The columns of the one grid CSV each PES family writes (its template header).
_PES_COLUMNS = {'pes_tip': ['x', 'y', 'energy_eV', 'fx', 'fy', 'fz'],
                'pes_sheet': ['dx', 'dy', 'energy_eV']}


#: Columns no results file carries: the step, and what the reader derives
#: (`frictionsim2d.data.models.compute_derived_columns`).
_NOT_PARSED = {'time', 'lateral_force', 'cof'}


def _check_columns(entry: dict) -> None:
    """``entry``'s tiers carry its family's observables, every column parsed back.

    A PES grid has its template's columns. Every other tier's columns are
    `_NOT_PARSED` plus registry channel names for (family, tier, n_layers): a native
    `v_...` left un-renamed means the deck and the registry disagree. The
    window must carry the friction label's ingredients.
    """
    family, tiers = entry['family'], entry['tiers']
    assert 'window' in tiers, (entry['rel_path'], sorted(tiers))
    if family in PES_FAMILIES:
        assert tiers['window']['columns'] == _PES_COLUMNS[family], entry['rel_path']
        return
    for tier, frame in tiers.items():
        known = set(rename_map(family, tier, entry['n_layers']).values()) | _NOT_PARSED
        stray = set(frame['columns']) - known
        assert not stray, (entry['rel_path'], tier, sorted(stray))
    assert {'nf', 'lfx', 'lfy'} <= set(tiers['window']['columns']), entry['rel_path']


def test_quench_scenario(lammps_replay, tmp_path, monkeypatch):
    """`run quench` -> fake melt-quench -> `quench import` -> the block is found."""
    from frictionsim2d.builders.construction.amorphous import (  # pylint: disable=import-outside-toplevel
        check_member, find_block, member_dirs)
    # The import writes the user library; a session-wide one would let the
    # imported block satisfy the deck snapshot's own quench_si render.
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg_data'))
    root = _build(QUENCH, tmp_path)
    assert [d.name for d in _fake_all(root)] == ['quench.in']

    _invoke('quench', 'import', root)

    members = member_dirs(root)
    assert members
    for member in members:
        _record, key, _block, box, _natoms = check_member(member)
        found = find_block(key, box)
        assert found is not None and found.is_relative_to(tmp_path / 'xdg_data'), found


def test_tier_budget(request):
    elapsed = tier_elapsed(request.config)
    if elapsed is None:
        pytest.skip('no other scenario test ran in this session')
    assert elapsed < BUDGET_S, f'scenario tier took {elapsed:.1f} s (budget {BUDGET_S:.0f} s)'
