"""The block schema (units, defaults, the spring-unit cross-check).

No aiida import and no `aiida` marker: this file runs in the fast suite.
"""
import json

import pytest

from src.aiida.block_schema import (
    BLOCK_SCHEMA_VERSION, FIELD_MAP, LEGACY_DEFAULTS, UNITS, RunBlocks, UnitMismatch,
    check_units, defaults_for, missing_units, numeric_paths, schema_document,
)
from src.data.models import NM_TO_EV_A2

_ALLOWED = {'angstrom', 'eV', 'eV/angstrom^2', 'N/m', 'nN', 'GPa', 'K', 'ps', 'm/s',
            'deg', 'meV', 'steps', '1',
            # beyond the brief's list, each justified at its UNITS entry
            'K/s', 'nN*s/m', 'eV*ps/angstrom^2', 'tip=angstrom;sub=1'}


def test_version_and_legacy():
    assert BLOCK_SCHEMA_VERSION == '1.0.0'
    assert LEGACY_DEFAULTS == {}


def test_every_numeric_field_has_a_unit():                    # user ruling
    assert missing_units() == []


def test_missing_unit_is_reported(monkeypatch):
    monkeypatch.delitem(UNITS, 'settings.simulation.timestep')
    assert missing_units() == ['settings.simulation.timestep']


def test_numeric_paths_cover_blocks_derived_and_points():
    paths = set(numeric_paths())
    for path in ('general.temp', 'sheets.x', 'settings.simulation.timestep',
                 'settings.output.dump_frequency', 'derived.drive_k',
                 'derived.geometry.lat_c', 'point.load.force', 'point.load.pressure',
                 'point.angle', 'point.speed', 'seed_velocity'):
        assert path in paths, path
    for path in ('settings.simulation.drive_method', 'tip.amorph', 'sheets.mat',
                 'settings.output.dump', 'derived.cutoff_source'):
        assert path not in paths, path


def test_units_are_ascii_and_known():
    for path, unit in UNITS.items():
        assert unit.isascii() and unit in _ALLOWED, (path, unit)
    assert not set(UNITS) - set(numeric_paths()), sorted(set(UNITS) - set(numeric_paths()))


def test_spring_units():
    assert UNITS['derived.drive_k'] == 'eV/angstrom^2'
    assert UNITS['general.driving_spring'] == 'N/m'


def _run(*, family='afm', drive_k, **drive_fields):
    keys = {'driving_spring': 'general.driving_spring',
            'drive_method': 'settings.simulation.drive_method'}
    fields = {keys[k]: v for k, v in drive_fields.items()}
    return RunBlocks(
        blocks={'drive': {'kind': 'drive', 'block_schema': BLOCK_SCHEMA_VERSION,
                          'fields': fields}},
        derived={'drive_k': drive_k}, family=family)


def test_drive_k_unit_and_cross_check():                      # I7
    check_units(_run(driving_spring=16.02, drive_method='virtual_atom', drive_k=1.0))
    with pytest.raises(UnitMismatch):
        check_units(_run(driving_spring=16.02, drive_method='virtual_atom',
                         drive_k=16.02))


def test_cross_check_uses_the_family_conversion():
    """Sheet builders divide by NM_TO_EV_A2, not 16.02 (sheetonsheet/base.py:803)."""
    check_units(_run(family='sheetonsheet', driving_spring=50.0, drive_method='smd',
                     drive_k=50.0 / NM_TO_EV_A2))
    with pytest.raises(UnitMismatch):
        check_units(_run(family='sheetonsheet', driving_spring=50.0, drive_method='smd',
                         drive_k=50.0 / 16.02))


def test_cross_check_reads_elided_defaults_and_fallbacks():
    # Both fields elided: virtual_atom and 50 N/m are the defaults.
    check_units(_run(drive_k=50.0 / 16.02))
    # A blanked spring takes the builder's fallback (afm/base.py:864: or 8.0).
    check_units(_run(driving_spring=None, drive_k=8.0 / 16.02))
    with pytest.raises(UnitMismatch):
        check_units(_run(drive_k=50.0))


def test_cross_check_skips_runs_without_a_spring():
    check_units(_run(drive_method='fix_move', drive_k=123.0))
    check_units(_run(driving_spring=16.02, drive_k=None))


def test_defaults_for_family():
    afm = defaults_for('afm')
    assert afm['settings.simulation.timestep'] == 0.001
    assert afm['general.driving_spring'] == 50.0               # int default, float field
    assert isinstance(afm['general.driving_spring'], float)
    assert afm['tip.dspring'] == 0.0
    assert 'tip.r' not in afm                                  # required
    sheet = defaults_for('hetero')
    assert 'tip.dspring' not in sheet
    assert sheet['settings.simulation.timestep'] == 0.001
    assert defaults_for('pes_tip') == afm
    with pytest.raises(ValueError):
        defaults_for('nonsense')


def test_schema_document_is_json_safe():
    doc = schema_document('afm')
    assert doc['block_schema'] == BLOCK_SCHEMA_VERSION and doc['family'] == 'afm'
    json.dumps(doc)
    entry = doc['fields']['settings.simulation.timestep']
    assert entry == {'kind': 'integration', 'unit': 'ps', 'default': 0.001}
    assert doc['fields']['derived.drive_k']['unit'] == 'eV/angstrom^2'
    assert doc['fields']['point.load.force']['unit'] == 'nN'
    assert 'tip.r' in doc['fields'] and 'tip.r' not in schema_document('sheetonsheet')['fields']
    assert {p for p in FIELD_MAP if p.startswith('general.')} <= set(doc['fields'])
