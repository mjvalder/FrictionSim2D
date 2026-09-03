"""Round-trip contract for the config writer.

The GUI authors configs that the CLI must be able to consume unchanged, so
``parse_config(f) == parse_config(write(parse_config(f)))`` has to hold for
every config in the repo. Equality is semantic (the parsed mapping), not
byte-wise: inline comments are not preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import parse_config
from src.core.config import format_ini_value, load_settings, read_config, write_config

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob('*.ini'))


def test_examples_directory_is_not_empty():
    """Guard against the parametrised tests below silently collecting nothing."""
    assert EXAMPLE_CONFIGS, f"no .ini configs found under {EXAMPLES_DIR}"


@pytest.mark.parametrize('config_path', EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_ini_round_trip_is_semantically_identical(config_path, tmp_path):
    original = parse_config(config_path)
    out = write_config(original, tmp_path / config_path.name)
    assert parse_config(out) == original


@pytest.mark.parametrize('config_path', EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_ini_round_trip_is_stable_on_a_second_pass(config_path, tmp_path):
    """Writing an already-written config must not drift further."""
    original = parse_config(config_path)
    first = parse_config(write_config(original, tmp_path / 'first.ini'))
    second = parse_config(write_config(first, tmp_path / 'second.ini'))
    assert second == first


@pytest.mark.parametrize('value,expected', [
    (None, ''),
    (5, '5'),
    (-5, '-5'),
    (5.0, '5.0'),
    (2.5, '2.5'),
    (1e-5, '1e-05'),
    ('sw', 'sw'),
    (True, 'True'),
    (False, 'False'),
    ([1, 2, 3], '[1, 2, 3]'),
    ([], '[]'),
    ([2.0, 5.5], '[2.0, 5.5]'),
    (['a', 'b'], '["a", "b"]'),
])
def test_format_ini_value(value, expected):
    assert format_ini_value(value) == expected


def test_none_is_written_as_an_empty_value_not_the_string_none(tmp_path):
    out = write_config({'2D': {'lat_c': None, 'mat': 'MoS2'}}, tmp_path / 'c.ini')
    assert 'lat_c =' in out.read_text(encoding='utf-8')
    assert 'None' not in out.read_text(encoding='utf-8')
    assert parse_config(out)['2D']['lat_c'] is None


def test_lists_survive_as_lists(tmp_path):
    original = {'general': {'force': [2, 5, 10], 'scan_angle': [0.0, 30.0]}}
    assert parse_config(write_config(original, tmp_path / 'c.ini')) == original


def test_floats_do_not_degrade_to_ints(tmp_path):
    original = {'general': {'temp': 300.0}}
    result = parse_config(write_config(original, tmp_path / 'c.ini'))
    assert isinstance(result['general']['temp'], float)


def test_percent_needs_no_escaping(tmp_path):
    """The writer must not use configparser interpolation."""
    out = write_config({'general': {'note': '50%'}}, tmp_path / 'c.ini')
    assert parse_config(out)['general']['note'] == '50%'


def test_hash_in_a_value_is_rejected_rather_than_silently_truncated():
    with pytest.raises(ValueError, match='#'):
        format_ini_value('a # b')


def test_inline_comments_are_emitted_and_ignored_on_read(tmp_path):
    out = write_config(
        {'2D': {'x': 100.0}},
        tmp_path / 'c.ini',
        comments={'2D': {'x': 'length of sheet in Angstrom'}},
    )
    assert '# length of sheet in Angstrom' in out.read_text(encoding='utf-8')
    assert parse_config(out) == {'2D': {'x': 100.0}}


def test_header_is_written_as_comments(tmp_path):
    out = write_config({'2D': {'x': 1.0}}, tmp_path / 'c.ini', header='written by test')
    assert out.read_text(encoding='utf-8').startswith('# written by test')


def test_yaml_and_json_round_trip(tmp_path):
    original = {'2D': {'x': 100.0, 'layers': [1, 2]}, 'general': {'temp': 300.0}}
    for name in ('c.yaml', 'c.json'):
        assert parse_config(write_config(original, tmp_path / name)) == original


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        write_config({}, tmp_path / 'c.toml')


def test_parse_config_rejects_an_unknown_extension(tmp_path):
    f = tmp_path / 'c.toml'; f.write_text('x = 1')
    with pytest.raises(ValueError, match='Unsupported configuration file format'):
        parse_config(f)


def test_parse_config_rejects_a_non_path_non_dict_source():
    with pytest.raises(TypeError, match='Unsupported configuration source type'):
        parse_config(42)


def test_parse_config_reads_json(tmp_path):
    f = tmp_path / 'c.json'; f.write_text(json.dumps({'general': {'temp': 300}}))
    assert parse_config(f)['general']['temp'] == 300


def test_read_config_parses_bracketed_lists_of_ints_and_floats(tmp_path):
    """read_config's own docstring: "a trailing `]` is what triggers list
    detection" -- an unbracketed comma list (the brief's original
    `layers = 1, 2, 3`) stays a plain string, confirmed by running it; adapted
    to the bracketed form the reader (and format_ini_value, its exact inverse)
    actually requires."""
    f = tmp_path / 'c.ini'; f.write_text('[2D]\nlayers = [1, 2, 3]\nx = [1.5, 2.5]\nmat = h-MoS2\n')
    cfg = read_config(f)
    assert cfg['2D']['layers'] == [1, 2, 3]
    assert cfg['2D']['x'] == [1.5, 2.5]
    assert cfg['2D']['mat'] == 'h-MoS2'


class _Float(float):
    """A float subclass whose repr lacks the '.0' — the guard at format_ini_value."""
    def __repr__(self):
        return '3'


def test_format_ini_value_keeps_a_float_subclass_a_float():
    assert format_ini_value(_Float(3.0)) == '3.0'


def test_load_settings_malformed_yaml_warns_and_falls_back_to_defaults(tmp_path, caplog):
    bad = tmp_path / 'settings.yaml'
    bad.write_text('thermostat: [unclosed')
    with caplog.at_level('WARNING'):
        result = load_settings(settings_file=bad)
    assert result.simulation.timestep == load_settings().simulation.timestep
    assert str(bad) in caplog.text
