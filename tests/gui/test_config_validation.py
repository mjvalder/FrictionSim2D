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
from tests.helpers.examples import FROZEN_EXAMPLES, friction_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Tests that name one config read the FROZEN copy (exact contents pinned);
#: the parametrized tests below read every LIVE example, discovered.
EXAMPLES = FROZEN_EXAMPLES
LIVE_INDENT = REPO_ROOT / 'examples' / 'indent' / 'basic.ini'

#: ``run`` models whose root config is AFMSimulationConfig.
_AFM_MODELS = {'afm', 'indent', 'pes_tip'}

#: Live friction examples, MLIP ones excluded: their multi-GB model files are
#: not in the repo (tests/integration/test_examples_mlip.py stubs them).
_LIVE = [e for e in friction_examples() if not e.mlip]


def _is_template(example) -> bool:
    """A ``{mat}``/``{mat1}`` placeholder in a value (comments do not count):
    resolvable only after list expansion."""
    lines = example.path.read_text(encoding='utf-8').splitlines()
    return any('{mat' in line.split('#', 1)[0] for line in lines)


BUILDABLE_EXAMPLES = [e for e in _LIVE if not _is_template(e)]
#: Configs that cannot construct as written: an unexpanded {mat}. The pair
#: sweep's {mat1}/{mat2} must report as unexpanded_template, NOT as
#: missing_file -- the latter is a hard error that would block the GUI's
#: Render/Generate for every pair config.
UNRESOLVABLE_EXAMPLES = [e for e in _LIVE if _is_template(e)]


def _model_for(model: str):
    return AFMSimulationConfig if model in _AFM_MODELS else SheetOnSheetSimulationConfig


def _raw(name: str):
    raw = parse_config(EXAMPLES / name)
    raw['settings'] = load_settings().model_dump()
    return raw


def _raw_example(example):
    raw = parse_config(example.path)
    raw['settings'] = load_settings(example.settings).model_dump()
    return raw


def test_both_example_lists_are_populated():
    assert BUILDABLE_EXAMPLES and UNRESOLVABLE_EXAMPLES


@pytest.mark.parametrize('example', BUILDABLE_EXAMPLES, ids=lambda e: e.rel)
def test_buildable_examples_validate(example, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    # A sweep example (interlayer_style = [lj, lj_g, ilp]) is validated member
    # by member, as the builder sees it; a plain config expands to itself.
    from frictionsim2d.run.expand import expand_config_sweeps
    members = expand_config_sweeps(_raw_example(example))
    assert members
    for member in members:
        result = validate_config(member, _model_for(example.model))
        assert result.ok, result.render()
        assert result.config is not None


@pytest.mark.parametrize('example', UNRESOLVABLE_EXAMPLES, ids=lambda e: e.rel)
def test_unresolvable_examples_report_issues_instead_of_raising(example, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    result = validate_config(_raw_example(example), _model_for(example.model))
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
    """Config-relative paths must resolve from any working directory.

    Reads the LIVE indent example: its portable ``materials/...`` names are the
    subject (the frozen copy still spells the legacy ``examples/materials/...``)."""

    def test_relative_paths_fail_from_an_unrelated_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(_repo_relative(_raw(LIVE_INDENT)),
                                 AFMSimulationConfig)
        assert not result.ok
        assert any(i.kind == 'missing_file' for i in result.issues)

    def test_library_names_resolve_from_any_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(_raw(LIVE_INDENT), AFMSimulationConfig)
        assert result.ok, result.render()

    def test_base_dir_makes_them_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = validate_config(
            _repo_relative(_raw(LIVE_INDENT)), AFMSimulationConfig, base_dir=REPO_ROOT
        )
        assert result.ok, result.render()

    def test_preflight_alone_reports_the_same(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _repo_relative(parse_config(LIVE_INDENT))
        assert preflight_paths(raw) != []
        assert preflight_paths(raw, base_dir=REPO_ROOT) == []

    def test_a_missing_file_names_the_directory_it_was_searched_from(self, tmp_path):
        raw = parse_config(LIVE_INDENT)
        raw['tip']['pot_path'] = 'no/such/potential.sw'
        issues = [i for i in preflight_paths(raw, base_dir=tmp_path)
                  if i.kind == 'missing_file' and i.key == 'pot_path']
        assert issues
        assert f'searched relative to {tmp_path}' in issues[0].message
        bare = [i for i in preflight_paths(raw) if i.kind == 'missing_file' and i.key == 'pot_path']
        assert 'searched relative to' not in bare[0].message
