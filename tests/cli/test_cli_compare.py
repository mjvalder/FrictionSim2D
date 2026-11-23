"""`FrictionSim2D compare` on archive sidecar directories (no aiida-core needed)."""
import json

from click.testing import CliRunner

from frictionsim2d.cli import cli


def _sidecar(root, temp, lat_c=3.19, unit='K', version='1.0.0'):
    deck = root / 'afm' / 'slide.json'
    deck.parent.mkdir(parents=True)
    deck.write_text(json.dumps({
        'thermostat': {'kind': 'thermostat', 'block_schema': version,
                       'fields': {'general.temp': temp}},
        'derived': {'geometry.lat_c': lat_c}}), encoding='utf-8')
    (root / f'schema-{version}.json').write_text(json.dumps({'fields': {
        'general.temp': {'kind': 'thermostat', 'unit': unit, 'default': 300.0},
        'derived.geometry.lat_c': {'kind': 'derived', 'unit': 'angstrom',
                                   'default': None}}}), encoding='utf-8')
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


def test_derived_values_print_with_units(tmp_path):
    a = _sidecar(tmp_path / 'a.params', 300.0, lat_c=3.19)
    b = _sidecar(tmp_path / 'b.params', 300.0, lat_c=3.21)
    out = CliRunner().invoke(cli, ['compare', str(a), str(b)]).output
    assert 'geometry.lat_c' in out and 'angstrom' in out
    loose = CliRunner().invoke(cli, ['compare', str(a), str(b), '--rtol', '0.1']).output
    assert 'geometry.lat_c' not in loose
    tight = CliRunner().invoke(cli, ['compare', str(a), str(b), '--atol', '1']).output
    assert 'geometry.lat_c' not in tight


def test_units_differing_between_sides_are_not_compared(tmp_path):
    a = _sidecar(tmp_path / 'a.params', 300.0)
    b = _sidecar(tmp_path / 'b.params', 0.3, unit='kK')
    out = CliRunner().invoke(cli, ['compare', str(a), str(b)]).output
    assert 'unit differs: A=K B=kK, not compared' in out
    assert '0.3' not in out


def test_sidecar_reads_each_deck_against_its_own_schema_version(tmp_path):
    root = tmp_path / 'm.params'
    _sidecar(root, 300.0, version='1.0.0')
    deck = root / 'afm' / 'slide.json'
    new = json.loads(deck.read_text(encoding='utf-8'))
    new['thermostat']['block_schema'] = '1.1.0'
    (root / 'afm2').mkdir()
    (root / 'afm2' / 'slide.json').write_text(json.dumps(new), encoding='utf-8')
    (root / 'schema-1.1.0.json').write_text(json.dumps({'fields': {
        'general.temp': {'kind': 'thermostat', 'unit': 'mK', 'default': 300.0}}}),
        encoding='utf-8')
    from frictionsim2d.aiida.compare import load_blocks_from_sidecar
    import pytest
    with pytest.raises(ValueError, match='schema versions'):
        load_blocks_from_sidecar(root)
    single = load_blocks_from_sidecar(root / 'afm2' / 'slide.json')
    assert single.units['general.temp'] == 'mK'   # its own 1.1.0, not the other
    assert load_blocks_from_sidecar(deck).units['general.temp'] == 'K'


# --- a store reference is never shadowed by a local directory -----------------

def _patched_store(monkeypatch, sides):
    """`load_blocks` from the store, recording each reference it is given."""
    import frictionsim2d.aiida as aiida_pkg
    import frictionsim2d.aiida.compare as compare_mod
    from frictionsim2d.aiida.compare import load_blocks_from_sidecar

    seen = []
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)

    def _store(ref, **kwargs):
        seen.append(ref)
        return load_blocks_from_sidecar(sides[ref], **kwargs)

    monkeypatch.setattr(compare_mod, 'load_blocks', _store)
    return seen


def test_a_plain_directory_named_like_a_group_reads_the_store(tmp_path, monkeypatch):
    a = _sidecar(tmp_path / 'store-a.params', 300.0)
    b = _sidecar(tmp_path / 'store-b.params', 400.0)
    seen = _patched_store(monkeypatch, {'camp': a, 'other': b})
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'camp').mkdir()                  # a folder, not an export sidecar
    (tmp_path / 'camp' / 'notes.json').write_text('{}', encoding='utf-8')

    result = CliRunner().invoke(cli, ['compare', 'camp', 'other'])

    assert result.exit_code == 0, result.output
    assert seen == ['camp', 'other']
    assert '300.0' in result.output and '400.0' in result.output


def test_a_directory_holding_an_export_sidecar_is_read_as_one(tmp_path, monkeypatch):
    seen = _patched_store(monkeypatch, {})
    export = tmp_path / 'transfer'
    _sidecar(export / 'camp.aiida.params', 300.0)
    b = _sidecar(tmp_path / 'b.params', 400.0)

    result = CliRunner().invoke(cli, ['compare', str(export), str(b)])

    assert result.exit_code == 0, result.output
    assert seen == []
    assert '300.0' in result.output and '400.0' in result.output


def test_the_sidecar_prefix_names_a_sidecar_explicitly(tmp_path, monkeypatch):
    seen = _patched_store(monkeypatch, {})
    a = _sidecar(tmp_path / 'a.params', 300.0)
    deck = _sidecar(tmp_path / 'b.params', 400.0) / 'afm' / 'slide.json'

    result = CliRunner().invoke(cli, ['compare', f'sidecar:{a}', f'sidecar:{deck}'])

    assert result.exit_code == 0, result.output
    assert seen == []
    assert '400.0' in result.output

    plain = tmp_path / 'plain'
    plain.mkdir()
    bad = CliRunner().invoke(cli, ['compare', f'sidecar:{plain}', f'sidecar:{a}'])
    assert bad.exit_code != 0
    assert 'schema-' in bad.output
