"""The shape of ``examples/``: one folder per ``run`` subcommand, every file runnable.

These pin what a new user relies on when they copy an example -- the file sits
in the folder of the command that runs it, its header gives that exact command,
and every list it names ships next to it -- and what the test suite relies on
when it discovers the examples instead of listing them.
"""
from __future__ import annotations

import re

import pytest

from tests.helpers.examples import EXAMPLES, MODES, REPO_ROOT, discover

_EXAMPLES = discover()
_MATERIALS_LIST_RE = re.compile(r'^\s*materials_list\s*=\s*([^#\s]+)', re.M)


def test_no_ini_outside_a_mode_folder():
    assert not list(EXAMPLES.glob('*.ini')), "an .ini at the examples/ top level"
    folders = {p.parent.relative_to(EXAMPLES).as_posix()
               for p in EXAMPLES.rglob('*.ini')}
    assert folders <= set(MODES), sorted(folders - set(MODES))
    top = {p.name for p in EXAMPLES.iterdir() if p.is_dir()}
    assert top <= set(MODES) | {'lists', 'settings'}, sorted(top)


def test_every_mode_has_an_example():
    assert {e.mode for e in _EXAMPLES} == set(MODES)


@pytest.mark.parametrize('example', _EXAMPLES, ids=lambda e: e.rel)
def test_every_example_header_runs_itself(example):
    text = example.path.read_text(encoding='utf-8')
    run_re = re.compile(
        rf'^#\s+FrictionSim2D run {re.escape(example.mode)} '
        rf'examples/{re.escape(example.rel)}(\s.*)?$', re.M)
    match = run_re.search(text)
    assert match, f"{example.rel}: no '#   FrictionSim2D run {example.mode} examples/{example.rel}' line"
    if example.settings is not None:
        wanted = f"--settings-file {example.settings.relative_to(REPO_ROOT).as_posix()}"
        assert wanted in match.group(0), f"{example.rel}: run line lacks {wanted}"


@pytest.mark.parametrize('example', _EXAMPLES, ids=lambda e: e.rel)
def test_every_materials_list_is_in_lists(example):
    text = example.path.read_text(encoding='utf-8')
    for value in _MATERIALS_LIST_RE.findall(text):
        path = REPO_ROOT / value
        assert path.parent == EXAMPLES / 'lists', f"{example.rel}: {value} is not in examples/lists/"
        assert path.is_file(), f"{example.rel}: {value} does not exist"


def test_a_missing_settings_header_file_raises(tmp_path):
    (tmp_path / 'afm').mkdir()
    (tmp_path / 'afm' / 'x.ini').write_text(
        "# settings: examples/settings/nope.yaml\n[2D]\nmat = a\n", encoding='utf-8')
    with pytest.raises(ValueError, match='x.ini'):
        discover(tmp_path)


_EXAMPLE_PATH_RE = re.compile(r'(?<![\w/])examples/[\w./-]+?\.(?:ini|txt|yaml)\b')
#: Where an ``examples/...`` path is shown to a person. ``core/utils/paths.py``
#: is excluded: its ``examples/`` is the legacy materials lookup, not this folder.
_DOC_FILES = sorted(
    [REPO_ROOT / 'README.md', REPO_ROOT / 'CLAUDE.md', EXAMPLES / 'README.md',
     *(REPO_ROOT / 'documentation').glob('*.md'),
     *(p for p in (REPO_ROOT / 'frictionsim2d').rglob('*')
       if p.suffix in ('.py', '.md') and p.name != 'paths.py')])


def test_readme_lists_every_example():
    readme = (EXAMPLES / 'README.md').read_text(encoding='utf-8')
    missing = [e.rel for e in _EXAMPLES if e.rel not in readme]
    assert not missing, f"examples/README.md does not list {missing}"


def test_no_stale_example_paths():
    stale = []
    for doc in _DOC_FILES:
        if not doc.is_file():
            continue
        for ref in _EXAMPLE_PATH_RE.findall(doc.read_text(encoding='utf-8')):
            if not (REPO_ROOT / ref).exists():
                stale.append(f"{doc.relative_to(REPO_ROOT)}: {ref}")
    assert not stale, "\n".join(stale)


def test_a_sheetonsheet_flake_needs_lammps(tmp_path):
    """A sheet-on-sheet flake build relaxes the flake with lmp_serial
    (sheetonsheet/flake_slide.py, _assemble_flake_on_sheet) even with one material."""
    for mode in ('sheetonsheet', 'afm'):
        (tmp_path / mode).mkdir()
        (tmp_path / mode / 'x.ini').write_text("[2D]\nmat = a\n[Flake]\nshape = hexagon\n",
                                               encoding='utf-8')
    needs = {e.mode: e.needs_lammps for e in discover(tmp_path)}
    assert needs == {'sheetonsheet': True, 'afm': False}


def test_readme_marks_every_lammps_example():
    readme = (EXAMPLES / 'README.md').read_text(encoding='utf-8').splitlines()
    for example in _EXAMPLES:
        row = next(line for line in readme if f"`{example.rel}`" in line)
        assert row.rstrip(' |').endswith('lammps') == example.needs_lammps, row
