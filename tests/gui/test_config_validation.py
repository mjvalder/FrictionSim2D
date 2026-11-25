"""Every way a config can fail must arrive as a located issue, not a traceback.

The GUI decorates the offending field from these reports, so an uncaught
exception here would surface as a crash in the Render handler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frictionsim2d.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from frictionsim2d.gui.config_validation import preflight_paths, validate_config
from tests.helpers.config_issues import issues_for

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

# Which root model each example config belongs to.
AFM_EXAMPLES = {
    'afm_config.ini', 'afm_flake_config.ini', 'indent_config.ini',
    'pes_tip_config.ini', 'afm_freestanding_pes_config.ini',
    'indent_campaign.ini', 'afm_freestanding_pes_campaign.ini',
}

#: Configs that construct cleanly from the repo root.
#:
#: `afm_config.ini` and `sheet_config.ini` moved here from UNRESOLVABLE_EXAMPLES
#: in `8129508`: both used to point their cif_path/pot_path outside the repo
#: (`run/cif/{mat}.cif`, `../tribo_2D/...`) and neither could build at all, which
#: is why the GUI listed them as unresolvable. They are now self-contained.
BUILDABLE_EXAMPLES = [
    'afm_config.ini', 'afm_flake_config.ini', 'afm_freestanding_pes_config.ini',
    'indent_config.ini', 'pes_sheet_config.ini', 'pes_tip_config.ini',
    'sheet_config.ini', 'sheet_config_interlayer.ini',
]

#: Configs that cannot construct: a missing file, or an unexpanded {mat}.
UNRESOLVABLE_EXAMPLES = [
    'indent_campaign.ini', 'afm_freestanding_pes_campaign.ini',
    # Pair sweep: {mat1}/{mat2} must report as unexpanded_template, NOT as
    # missing_file -- the latter is a hard error that would block the GUI's
    # Render/Generate for every pair config.
    'hetero_pair_sweep.ini',
]


def _model_for(name: str):
    return AFMSimulationConfig if name in AFM_EXAMPLES else SheetOnSheetSimulationConfig


def _raw(name: str):
    raw = parse_config(EXAMPLES / name)
    raw['settings'] = load_settings().model_dump()
    return raw


#: Not friction configs: `run quench` reads a [quench] section (the GUI
#: "Amorphous block" family validates it, Task 7b).
QUENCH_EXAMPLES = ['quench_config.ini']


def test_the_two_example_lists_cover_every_config():
    on_disk = {p.name for p in EXAMPLES.glob('*.ini')}
    assert (set(BUILDABLE_EXAMPLES) | set(UNRESOLVABLE_EXAMPLES)
            | set(QUENCH_EXAMPLES)) == on_disk


@pytest.mark.parametrize('name', BUILDABLE_EXAMPLES)
def test_buildable_examples_validate(name, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    # A sweep example (interlayer_style = [lj, lj_g, ilp]) is validated member
    # by member, as the builder sees it; a plain config expands to itself.
    from frictionsim2d.run.expand import expand_config_sweeps
    members = expand_config_sweeps(_raw(name))
    assert members
    for member in members:
        result = validate_config(member, _model_for(name))
        assert result.ok, result.render()
        assert result.config is not None


@pytest.mark.parametrize('name', UNRESOLVABLE_EXAMPLES)
def test_unresolvable_examples_report_issues_instead_of_raising(name, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    result = validate_config(_raw(name), _model_for(name))
    assert not result.ok
    assert result.config is None
    # Every issue must be locatable, otherwise the GUI has nothing to decorate.
    assert all(i.section for i in result.issues), result.render()
    assert all(i.kind in ('missing_file', 'unexpanded_template') for i in result.issues), \
        result.render()


def test_missing_potential_is_located_on_the_right_field(monkeypatch):
    """The issue must name the [tip] pot_path field, not just the file.

    Built by breaking one field of a config that otherwise validates, rather
    than by reading a shipped example that happens to be broken -- this used to
    use `afm_config.ini`, whose potential paths pointed outside the repo until
    `8129508` made it self-contained.
    """
    monkeypatch.chdir(REPO_ROOT)
    raw = _raw('afm_flake_config.ini')
    raw['tip']['pot_path'] = 'frictionsim2d/data/potentials/sw/not_a_real_potential.sw'

    result = validate_config(raw, AFMSimulationConfig)
    assert not result.ok
    tip_issues = issues_for(result, 'tip', 'pot_path')
    assert tip_issues, result.render()
    assert 'not_a_real_potential.sw' in tip_issues[0].message


def test_unexpanded_template_is_reported_as_its_own_kind(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    result = validate_config(_raw('indent_campaign.ini'), AFMSimulationConfig)
    kinds = {i.kind for i in result.issues}
    assert 'unexpanded_template' in kinds, result.render()


def test_pair_template_is_unexpanded_not_missing_file():
    """A {mat1}/{mat2} heterostructure-pair template must classify the same
    way a plain {mat} does. Getting this wrong is the most dangerous failure
    mode in the pair-sweep feature: 'missing_file' is a hard error that
    permanently blocks Render and Generate for every pair config, since the
    file that is 'missing' can never be found -- it does not exist until the
    sweep expands the template."""
    issues = preflight_paths({
        '2D-1': {'cif_path': 'cif/{mat1}.cif', 'pot_path': 'pot/{mat1}.sw'},
        '2D-2': {'cif_path': 'cif/{mat2}.cif', 'pot_path': 'pot/{mat2}.sw'},
    })
    assert issues, "expected issues for unresolved templated paths"
    kinds = {i.kind for i in issues}
    assert kinds == {'unexpanded_template'}
    assert 'missing_file' not in kinds


def test_a_second_sheet_error_keeps_its_field_name(monkeypatch):
    """A list-indexed loc -- ('sheets', 1, 'x') -- must report the field, not
    the index. Keeping only the first two elements makes every second-sheet
    failure arrive as key='1', which names no editor, so the GUI decorates
    nothing however well it maps the section."""
    monkeypatch.chdir(REPO_ROOT)
    raw = _raw('pes_sheet_config.ini')
    raw['2D-1'] = raw.pop('2D')
    raw['2D-2'] = dict(raw['2D-1'], x='not-a-number')
    result = validate_config(raw, SheetOnSheetSimulationConfig)

    indexed = [i for i in result.issues if i.section == 'sheets']
    assert indexed, result.render()
    # One report per union member (float | list[float]); all of them located.
    assert {(i.index, i.key) for i in indexed} == {(1, 'x')}


def test_a_plain_nested_loc_is_unchanged(monkeypatch):
    """Only list indices are pulled out: a nested model's ('settings', 'x')
    style loc must keep reporting its own second element as the key."""
    monkeypatch.chdir(REPO_ROOT)
    raw = _raw('indent_config.ini')
    raw['tip']['r'] = 'not-a-number'
    issue = issues_for(validate_config(raw, AFMSimulationConfig), 'tip', 'r')[0]
    assert issue.index is None


def test_type_errors_are_reported_with_a_location(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    raw = _raw('indent_config.ini')
    raw['tip']['r'] = 'not-a-number'
    result = validate_config(raw, AFMSimulationConfig)
    assert not result.ok
    assert issues_for(result, 'tip', 'r'), result.render()


def _repo_relative(raw):
    """``raw`` with every library name spelled as a repo-relative path
    (``frictionsim2d/data/...``): a path the library lookup cannot rescue, so
    only the working directory or ``base_dir`` can resolve it."""
    for section in raw.values():
        if not isinstance(section, dict):
            continue
        for key in ('cif_path', 'pot_path'):
            value = section.get(key)
            if isinstance(value, str) and value.startswith(('materials/', 'potentials/')):
                section[key] = f'frictionsim2d/data/{value}'
    return raw


class TestBaseDirResolution:
    """Config-relative paths must resolve from any working directory."""

    def test_relative_paths_fail_from_an_unrelated_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(_repo_relative(_raw('indent_config.ini')),
                                 AFMSimulationConfig)
        assert not result.ok
        assert any(i.kind == 'missing_file' for i in result.issues)

    def test_library_names_resolve_from_any_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(_raw('indent_config.ini'), AFMSimulationConfig)
        assert result.ok, result.render()

    def test_base_dir_makes_them_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(
            _repo_relative(_raw('indent_config.ini')), AFMSimulationConfig, base_dir=REPO_ROOT
        )
        assert result.ok, result.render()

    def test_preflight_alone_reports_the_same(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _repo_relative(parse_config(EXAMPLES / 'indent_config.ini'))
        assert preflight_paths(raw) != []
        assert preflight_paths(raw, base_dir=REPO_ROOT) == []

    def test_a_missing_file_names_the_directory_it_was_searched_from(self, tmp_path):
        raw = parse_config(EXAMPLES / 'indent_config.ini')
        raw['tip']['pot_path'] = 'no/such/potential.sw'
        issues = [i for i in preflight_paths(raw, base_dir=tmp_path)
                  if i.kind == 'missing_file' and i.key == 'pot_path']
        assert issues
        assert f'searched relative to {tmp_path}' in issues[0].message
        bare = [i for i in preflight_paths(raw) if i.kind == 'missing_file' and i.key == 'pot_path']
        assert 'searched relative to' not in bare[0].message
