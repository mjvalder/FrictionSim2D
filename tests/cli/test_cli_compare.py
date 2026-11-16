"""`FrictionSim2D compare` on archive sidecar directories (no aiida-core needed)."""
import json

from click.testing import CliRunner

from src.cli import cli


def _sidecar(root, temp):
    deck = root / 'afm' / 'slide.json'
    deck.parent.mkdir(parents=True)
    deck.write_text(json.dumps({
        'thermostat': {'kind': 'thermostat', 'block_schema': '1.0.0',
                       'fields': {'general.temp': temp}},
        'derived': {'geometry.lat_c': 3.19}}), encoding='utf-8')
    (root / 'schema-1.0.0.json').write_text(json.dumps({'fields': {
        'general.temp': {'kind': 'thermostat', 'unit': 'K', 'default': 300.0}}}),
        encoding='utf-8')
    return root


def test_compare_two_sidecars(tmp_path):
    a = _sidecar(tmp_path / 'a.params', 300.0)
    b = _sidecar(tmp_path / 'b.params', 400.0)
    result = CliRunner().invoke(cli, ['compare', str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert 'thermostat' in result.output and 'temp' in result.output
    assert 'K' in result.output and '300.0' in result.output and '400.0' in result.output


def test_compare_same_sidecar(tmp_path):
    a = _sidecar(tmp_path / 'a.params', 300.0)
    result = CliRunner().invoke(cli, ['compare', str(a), str(a)])
    assert 'no differences' in result.output
