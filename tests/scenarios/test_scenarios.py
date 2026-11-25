"""Tiny end-to-end runs with a fake LAMMPS: build -> outputs -> read -> import.

One row per (family, potential, condition) of the matrix,
``tests.helpers.deck_snapshot.SCENARIO_CASES`` (so the deck snapshot pins the
same builds). Each row builds its config in the snapshot's layout (build-time
LAMMPS replayed by ``lammps_replay``), lets ``fake_lmp.fake_run`` write every
file every deck would write, then drives the real CLI: ``postprocess read
--export`` must produce ``output_full_*.json`` with the row's expected number
of points and the family's observables columns, and ``aiida import`` into
``aiida_profile_clean`` must give one ``point_table()`` row per point. The
quench row goes ``quench import`` -> ``find_block`` instead.

Substitutions in the matrix (each also named at ``SCENARIO_CASES``):

- pes-sheet + kc is ``pes_sheet_ljg`` (lj_g): kc has bundled C/H parameters
  only, and no carbon material ships;
- the MACE model is ``configs/fake_mace.json``, junk bytes the build stages and
  nothing parses;
- afm-freestanding + tersoff is a freestanding ``pes_tip`` run on b-SiGe
  (1989_SiGe.tersoff); freestanding exists only as a pes-tip variant;
- afm-hetero (lj, the default interlayer), sheet-hetero twist and quench reuse
  the deck cases ``afm_hetero_finite``, ``sheet_hetero_twist``, ``quench_si``;
- four rows skip ``aiida import`` (``AIIDA_ROWS``): they repeat a family's
  import path, and importing them breaks the tier's 60 s budget
  (``conftest.py``).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from frictionsim2d import cli as cli_module
from frictionsim2d.core.observables import PES_FAMILIES, REGISTRY, layer_channels, rename_map
from frictionsim2d.run.artifacts import generate_package
from tests.helpers.deck_snapshot import SCENARIO_CASES, RenderCase
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

#: Load points each row's export must hold: the product of its sweeps.
EXPECTED_POINTS = {
    'afm_sw_load': 2,               # force [5, 10]
    'afm_rebomos_angle_finite': 2,  # scan_angle [0, 30]
    'afm_freestanding_tersoff': 1,  # one PES grid
    'indent_sw': 2,                 # force [5, 10]
    'sheet_reaxff_speed': 2,        # scan_speed [1, 2]
    'sheet_sw_ilp': 1,
    'flake_slide_sw': 1,
    'pes_tip_sw': 1,                # one PES grid
    'pes_sheet_ljg': 1,             # one PES grid
    'sheet_mace_d3': 2,             # pressure [0.1, 1]
    'afm_hetero_finite': 1,
    'sheet_hetero_twist': 1,
}

#: Registry channels a row's deck legitimately does not emit, by row: an ilp
#: interlayer has no group/group interlayer energy (lj/lj_g only), and a
#: hetero AFM sheet's layers are z-bands with no per-layer COM groups.
_ROW_OPTIONAL = {
    'sheet_sw_ilp': {'E_int_interlayer'},
    'afm_hetero_finite': {f'com{ax}_L{n}' for ax in 'xyz' for n in (1, 2)},
}

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


def test_fake_lmp_command_line_writes_the_deck_files(tmp_path):
    deck = tmp_path / 'deck.in'
    deck.write_text('variable p index 1 2\n'
                    'fix fc all ave/time 1 10 10 v_fz file f_${p}.txt\n'
                    'run 10\n', encoding='utf-8')
    script = Path(__file__).resolve().parent / 'fake_lmp.py'

    done = subprocess.run([sys.executable, str(script), '-in', str(deck), '-log', 'run.log'],
                          cwd=tmp_path, capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr
    assert (tmp_path / 'f_1.txt').is_file() and (tmp_path / 'f_2.txt').is_file()
    assert 'Total wall time' in (tmp_path / 'run.log').read_text(encoding='utf-8')
    assert str(tmp_path / 'f_2.txt') in done.stdout


def test_fake_run_refuses_a_power_and_warns_on_an_unevaluable_name(tmp_path):
    deck = tmp_path / 'deck.in'
    deck.write_text('variable big equal 9**9**9\n'
                    'variable g equal f_force[3]\n'
                    'print "$(v_big)" file out/big.txt\n'
                    'fix fc all ave/time 1 10 10 v_fz file out/f_${g}.txt\n', encoding='utf-8')

    with pytest.warns(UserWarning, match='cannot be evaluated'):
        fake_run(deck, rows=1, cwd=tmp_path)

    assert (tmp_path / 'out' / 'big.txt').is_file()
    assert (tmp_path / 'out' / 'f_0.txt').is_file()


@pytest.mark.parametrize('case', SCENARIOS, ids=[c.name for c in SCENARIOS])
def test_scenario(case, lammps_replay, tmp_path, request):
    root = _build(case, tmp_path)
    _fake_all(root)

    result = _invoke('postprocess', 'read', root, '--export')
    exports = sorted((root / 'outputs').glob('output_full_*.json'))
    assert exports, result.output
    entries = [entry for export in exports
               for entry in json.loads(export.read_text(encoding='utf-8'))['runs']]
    assert len(entries) == EXPECTED_POINTS[case.name], [
        (e['load'], e['angle'], e['speed']) for e in entries]
    for entry in entries:
        _check_columns(entry, _ROW_OPTIONAL.get(case.name, set()))

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


def _optional_channels(n_layers: int) -> set:
    """Registry channels no default deck emits: the opt-in ones, and the
    static per-layer entries for layers the stack does not have."""
    names = {ob.name for ob in list(REGISTRY.values()) + layer_channels(n_layers)
             if ob.opt_in}
    for ob in REGISTRY.values():
        match = re.search(r'_L(\d+)$', ob.name)
        if match and int(match.group(1)) > n_layers:
            names.add(ob.name)
    return names


def _check_columns(entry: dict, row_optional: set) -> None:
    """``entry``'s tiers carry its family's observables, and only those.

    A PES grid has its template's columns. Every other tier written carries
    every registry channel of (family, tier, n_layers) except the optional
    ones (`_optional_channels`, plus the row's own `row_optional`), and
    nothing beyond them and `_NOT_PARSED`: a native `v_...` left un-renamed
    means the deck and the registry disagree.
    """
    family, tiers = entry['family'], entry['tiers']
    assert 'window' in tiers, (entry['rel_path'], sorted(tiers))
    if family in PES_FAMILIES:
        assert tiers['window']['columns'] == _PES_COLUMNS[family], entry['rel_path']
        return
    optional = _optional_channels(entry['n_layers']) | row_optional
    for tier, frame in tiers.items():
        channels = set(rename_map(family, tier, entry['n_layers']).values())
        columns = set(frame['columns'])
        assert not columns - channels - _NOT_PARSED, (
            entry['rel_path'], tier, sorted(columns - channels - _NOT_PARSED))
        assert not channels - optional - columns, (
            entry['rel_path'], tier, sorted(channels - optional - columns))


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
