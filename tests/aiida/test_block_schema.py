"""The block schema (units, defaults, the spring-unit cross-check).

No aiida import and no `aiida` marker: this file runs in the fast suite.
"""
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.aiida.block_schema import (
    AFM_FAMILIES, BLOCK_SCHEMA_VERSION, FIELD_MAP, LEGACY_DEFAULTS, SHEET_FAMILIES,
    UNITS, RunBlocks, UnitMismatch, check_units, defaults_for, missing_units,
    numeric_paths, schema_document,
)
from src.core.config import load_settings
from src.data.models import NM_TO_EV_A2

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / 'tests' / 'fixtures' / 'block_defaults.json'
FAMILIES = sorted(AFM_FAMILIES | SHEET_FAMILIES)

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


# --- the default snapshot -------------------------------------------------------

def _snapshot_drift(current_version, current, golden):
    """Why ``current`` defaults disagree with the ``golden`` snapshot, or None."""
    if golden['block_schema'] != current_version:
        return (f"BLOCK_SCHEMA_VERSION is {current_version} but "
                f"{SNAPSHOT.name} was taken at {golden['block_schema']}: "
                "regenerate the snapshot for the new version.")
    lines = []
    for family in sorted(set(current) | set(golden['families'])):
        now, then = current.get(family, {}), golden['families'].get(family, {})
        for path in sorted(set(now) | set(then)):
            if now.get(path, '<absent>') != then.get(path, '<absent>'):
                lines.append(f"  {family}: {path}: {then.get(path, '<absent>')!r} -> "
                             f"{now.get(path, '<absent>')!r}")
    if not lines:
        return None
    return ("A model default changed without a schema bump, so default elision would "
            "now hash different physics alike:\n" + "\n".join(lines) + "\n"
            "Bump BLOCK_SCHEMA_VERSION, record each old value in "
            "LEGACY_DEFAULTS[<old version>], then regenerate "
            f"tests/fixtures/{SNAPSHOT.name}.")


def _current_defaults():
    return {family: defaults_for(family) for family in FAMILIES}


def test_defaults_match_the_snapshot():
    golden = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    drift = _snapshot_drift(BLOCK_SCHEMA_VERSION, _current_defaults(), golden)
    assert drift is None, drift


def test_snapshot_drift_names_the_fix():
    golden = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    changed = _current_defaults()
    changed['afm'] = {**changed['afm'], 'settings.simulation.timestep': 0.002}
    drift = _snapshot_drift(BLOCK_SCHEMA_VERSION, changed, golden)
    assert 'afm: settings.simulation.timestep: 0.001 -> 0.002' in drift
    assert 'Bump BLOCK_SCHEMA_VERSION' in drift and 'LEGACY_DEFAULTS' in drift
    assert 'regenerate' in _snapshot_drift('9.9.9', _current_defaults(), golden)


def test_defaults_are_environment_free():
    """Only block fields: settings.database.* reads the environment, and is excluded."""
    for family in FAMILIES:
        assert set(defaults_for(family)) <= set(FIELD_MAP), family


# --- quench_rate: label vs builder ----------------------------------------------

def _quench_nsteps(tmp_path, rate):
    """The quench run length the builder writes for ``rate`` (no LAMMPS run)."""
    from src.builders.construction import components  # pylint: disable=import-outside-toplevel
    settings = load_settings().model_copy(deep=True)
    settings.quench.run_local = False                    # write the deck, then stop
    settings.quench.quench_rate = rate
    out = tmp_path / f'r{rate:g}'
    out.mkdir()
    mat = REPO / 'examples' / 'materials' / 'Si.cif'
    pot = REPO / 'examples' / 'potentials' / 'sw' / 'Si.sw'
    with patch.object(components, 'create_orthogonal_slab'), \
            pytest.raises(RuntimeError, match='manual LAMMPS run'):
        components.make_amorphous('zz_quench_rate_probe', mat, 8.0, 8.0, 8.0, pot,
                                  'sw', out, settings)
    deck = (out / 'lmp_input_amorphous_zz_quench_rate_probe.in').read_text(encoding='utf-8')
    return int(re.search(r'^fix\s+quench .*\n^run\s+(\d+)', deck, re.M).group(1))


@pytest.mark.xfail(strict=True, reason=(
    "quench_rate is labelled K/s but construction/components.py:1765-1766 makes the "
    "quench step count proportional to it; the builder fix awaits the user."))
def test_doubling_quench_rate_halves_the_quench(tmp_path):
    assert _quench_nsteps(tmp_path, 2e12) * 2 == _quench_nsteps(tmp_path, 1e12)
