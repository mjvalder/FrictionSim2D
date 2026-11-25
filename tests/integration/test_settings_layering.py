"""Both config layers must reach the rendered LAMMPS script.

Spec Phase 3, item 3. ``tests/core/test_config.py`` already pins which settings
file ``load_settings`` picks; nothing checks that the value it picked survives
the builder and lands in a script. A settings key that loads correctly and is
then dropped between the config object and the template is invisible to every
existing test, and silent at run time -- LAMMPS just uses the default.

A correction to the spec's wording while writing these: the two layers do not
*override* each other. ``[general]`` (``GeneralConfig``) and ``settings.yaml``
(``GlobalSettings``) are disjoint namespaces -- there is no key an ``.ini`` can
write that displaces a ``settings.yaml`` one. They compose, and the tests below
assert that composition rather than a precedence that does not exist. Precedence
does exist *within* the settings layer, and it is tested: an explicit
``settings_file=`` beats ``$XDG_CONFIG_HOME/FrictionSim2D/settings.yaml``.

``indent_config.ini`` is the vehicle (1 s to build); the three settings values
below are the ones that reach its rendered scripts verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from tests.helpers.examples import FROZEN_EXAMPLES, build

pytestmark = pytest.mark.integration

CONFIG, MODEL = 'indent_config.ini', 'indent'

#: settings.yaml values that appear verbatim in a rendered script, and the
#: command each lands in.
XDG_SETTINGS: Dict[str, object] = {
    'simulation': {'thermo': 777, 'timestep': 0.0025},
    'output': {'results_frequency': 250},
}
EXPECTED_LINES = ('thermo          777', 'timestep        0.0025')


def _write_settings(directory: Path, data: Dict[str, object]) -> Path:
    path = directory / 'settings.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def _xdg(monkeypatch, tmp_path: Path, data: Dict[str, object]) -> Path:
    """Point ``$XDG_CONFIG_HOME`` at a settings.yaml holding ``data``."""
    home = tmp_path / 'xdg'
    directory = home / 'FrictionSim2D'
    directory.mkdir(parents=True)
    path = _write_settings(directory, data)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home))
    return path


def _rendered(result) -> List[str]:
    """Every rendered script of the first member, as text."""
    member = Path(result.created[0])
    scripts = sorted(member.rglob('lammps/*.in'))
    assert scripts, member
    return [s.read_text(encoding='utf-8') for s in scripts]


def _build(tmp_path: Path, **kwargs):
    result = build(CONFIG, MODEL, tmp_path / 'out', source=FROZEN_EXAMPLES, **kwargs)
    assert result.ok, result.render()
    return result


def test_an_xdg_settings_value_reaches_the_rendered_script(monkeypatch, tmp_path):
    _xdg(monkeypatch, tmp_path, XDG_SETTINGS)
    texts = _rendered(_build(tmp_path))
    for line in EXPECTED_LINES:
        assert any(line in text for text in texts), (line, texts[0][:400])


def test_an_xdg_settings_value_reaches_a_fix_argument_not_just_a_bare_command(
        monkeypatch, tmp_path):
    """`results_frequency` is an argument in the middle of a `fix ave/time`
    line, not a command of its own -- a different substitution path from
    `thermo`, and the one postprocessing depends on."""
    _xdg(monkeypatch, tmp_path, XDG_SETTINGS)
    texts = _rendered(_build(tmp_path))
    assert any('ave/time 1 1000 250 ' in text for text in texts)


def test_an_explicit_settings_file_beats_the_xdg_one(monkeypatch, tmp_path):
    """`_settings_paths_in_precedence_order` puts an explicit file first. Proven
    at the rendered script, not at the loader: both files are valid and differ
    only in the value, so only the winner's number can appear."""
    _xdg(monkeypatch, tmp_path, XDG_SETTINGS)
    explicit = _write_settings(tmp_path, {'simulation': {'thermo': 12345}})
    texts = _rendered(_build(tmp_path, settings_file=explicit))
    assert any('thermo          12345' in text for text in texts)
    assert not any('thermo          777' in text for text in texts)


def test_an_ini_general_value_reaches_the_rendered_script(monkeypatch, tmp_path):
    """The other layer. `[general] temp = 300` in indent_config.ini becomes the
    thermostat target."""
    _xdg(monkeypatch, tmp_path, XDG_SETTINGS)
    texts = _rendered(_build(tmp_path))
    assert any('300.0 300.0' in text for text in texts), texts[0][:400]


def test_the_two_layers_compose_rather_than_replace_each_other(monkeypatch,
                                                               tmp_path):
    """The `.ini` names none of the settings.yaml keys, and the settings file
    names none of the `.ini` ones; both must survive into the same script.
    A builder that rebuilt GlobalSettings from the `.ini` alone would drop the
    settings half silently."""
    _xdg(monkeypatch, tmp_path, XDG_SETTINGS)
    texts = _rendered(_build(tmp_path))
    both = [text for text in texts
            if 'thermo          777' in text and '300.0 300.0' in text]
    assert both, "no single script carries a value from each layer"


def test_no_settings_file_anywhere_still_builds_on_the_defaults(monkeypatch,
                                                               tmp_path):
    """`load_settings` falls back to the hardcoded GlobalSettings defaults, so a
    checkout with no settings.yaml at all is a working configuration."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    monkeypatch.setenv('XDG_CONFIG_HOME', str(empty))
    texts = _rendered(_build(tmp_path))
    assert any('thermo          100000' in text for text in texts)
    assert not any('thermo          777' in text for text in texts)
