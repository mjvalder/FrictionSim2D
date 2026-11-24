"""The block schema (units, defaults, the spring-unit cross-check).

No aiida import and no `aiida` marker: this file runs in the fast suite.
"""
import json
import re
from pathlib import Path

import pytest

from frictionsim2d.aiida.block_schema import (
    AFM_FAMILIES, BLOCK_SCHEMA_VERSION, FIELD_MAP, LEGACY_DEFAULTS, SHEET_FAMILIES,
    UNITS, RunBlocks, UnitMismatch, check_units, defaults_for, missing_units,
    numeric_paths, schema_document,
)
from frictionsim2d.core.config import load_settings
from frictionsim2d.data.models import NM_TO_EV_A2

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / 'tests' / 'fixtures' / 'block_defaults.json'
FAMILIES = sorted(AFM_FAMILIES | SHEET_FAMILIES)

_ALLOWED = {'angstrom', 'eV', 'eV/angstrom^2', 'N/m', 'nN', 'GPa', 'K', 'ps', 'm/s',
            'deg', 'meV', 'steps', '1',
            # beyond the brief's list, each justified at its UNITS entry
            'K/s', 'nN*s/m', 'eV*ps/angstrom^2', 'tip=angstrom;sub=1'}


def test_version_and_legacy():
    assert BLOCK_SCHEMA_VERSION == '1.1.0'
    assert LEGACY_DEFAULTS == {'1.0.0': {'settings.quench.melt_steps': 50000,
                                         'settings.quench.equilibrate_steps': 20000}}


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
    check_units(_run(driving_spring=NM_TO_EV_A2, drive_method='virtual_atom', drive_k=1.0))
    with pytest.raises(UnitMismatch):
        check_units(_run(driving_spring=NM_TO_EV_A2, drive_method='virtual_atom',
                         drive_k=NM_TO_EV_A2))


@pytest.mark.parametrize('family', ['afm', 'sheetonsheet'])
def test_cross_check_uses_the_one_exact_constant(family):
    """Every builder divides by NM_TO_EV_A2 (afm/base.py, sheetonsheet/base.py,
    hetero.py, flake_slide.py); the old AFM 16.02 no longer passes."""
    check_units(_run(family=family, driving_spring=50.0, drive_method='smd',
                     drive_k=50.0 / NM_TO_EV_A2))
    with pytest.raises(UnitMismatch):
        check_units(_run(family=family, driving_spring=50.0, drive_method='smd',
                         drive_k=50.0 / 16.02))


@pytest.mark.parametrize('family', ['afm', 'sheetonsheet'])
@pytest.mark.parametrize('spring', [50.0, 801.0])
def test_cross_check_accepts_a_drive_k_built_with_the_codata2010_factor(family, spring):
    """Stored runs converted N/m with 16.02176565 before the constant moved."""
    check_units(_run(family=family, driving_spring=spring, drive_method='smd',
                     drive_k=spring / 16.02176565))
    with pytest.raises(UnitMismatch):
        check_units(_run(family=family, driving_spring=spring, drive_method='smd',
                         drive_k=spring / 16.03))


def test_cross_check_reads_elided_defaults_and_fallbacks():
    # Both fields elided: virtual_atom and 50 N/m are the defaults.
    check_units(_run(drive_k=50.0 / NM_TO_EV_A2))
    # A blanked spring takes the family's builder fallback: AFM 8.0, sheet 50.0.
    check_units(_run(driving_spring=None, drive_k=8.0 / NM_TO_EV_A2))
    check_units(_run(family='sheetonsheet', driving_spring=None,
                     drive_k=50.0 / NM_TO_EV_A2))
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
    changed, added = [], []
    for family in sorted(set(current) | set(golden['families'])):
        now, then = current.get(family, {}), golden['families'].get(family, {})
        for path in sorted(set(now) | set(then)):
            if now.get(path, '<absent>') == then.get(path, '<absent>'):
                continue
            line = (f"  {family}: {path}: {then.get(path, '<absent>')!r} -> "
                    f"{now.get(path, '<absent>')!r}")
            (added if path not in then else changed).append(line)
    if not changed and not added:
        return None
    text = ''
    if changed:
        text += ("A model default changed or was removed without a schema bump, so default "
                 "elision would now hash different physics alike:\n" + "\n".join(changed)
                 + "\nBump BLOCK_SCHEMA_VERSION, record each old value in "
                 "LEGACY_DEFAULTS[<old version>], then regenerate the snapshot.\n")
    if added:
        text += ("A model field was added (an elided default changes no existing hash, so no "
                 "bump is needed):\n" + "\n".join(added) + "\n")
    return text + (f"Regenerate tests/fixtures/{SNAPSHOT.name}: "
                   "{'block_schema': VERSION, 'families': {family: defaults_for(family)}} "
                   "as json.dumps(indent=1, sort_keys=True) plus a newline.")


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


def test_snapshot_drift_tells_an_added_field_from_a_changed_default():
    golden = {'block_schema': BLOCK_SCHEMA_VERSION,
              'families': {'afm': {'a': 1, 'b': 2}}}
    added = _snapshot_drift(BLOCK_SCHEMA_VERSION, {'afm': {'a': 1, 'b': 2, 'c': 3}}, golden)
    assert "afm: c: '<absent>' -> 3" in added and 'no bump is needed' in added
    assert 'Bump' not in added
    changed = _snapshot_drift(BLOCK_SCHEMA_VERSION, {'afm': {'a': 9, 'b': 2}}, golden)
    assert 'afm: a: 1 -> 9' in changed and 'Bump BLOCK_SCHEMA_VERSION' in changed
    assert _snapshot_drift(BLOCK_SCHEMA_VERSION, golden['families'], golden) is None


def test_defaults_are_environment_free():
    """Only block fields are listed."""
    for family in FAMILIES:
        assert set(defaults_for(family)) <= set(FIELD_MAP), family


# --- quench_rate: label vs builder ----------------------------------------------

def _quench_nsteps(tmp_path, rate):
    """The quench run length ``run quench`` writes for ``rate`` (no LAMMPS run)."""
    # pylint: disable-next=import-outside-toplevel
    from frictionsim2d.builders.construction.amorphous import QuenchMember, write_quench_root
    settings = load_settings().model_copy(deep=True)
    settings.quench.quench_rate = rate
    mat = REPO / 'examples' / 'materials' / 'Si.cif'
    pot = REPO / 'examples' / 'potentials' / 'sw' / 'Si.sw'
    member = QuenchMember.make('Si', pot, 'sw', mat, (8.0, 8.0, 8.0), settings.quench)
    root = write_quench_root([member], tmp_path, settings, root_name=f'r{rate:g}')
    deck = next(root.rglob('quench.in')).read_text(encoding='utf-8')
    return int(re.search(r'^fix\s+quench .*\n^run\s+(\d+)', deck, re.M).group(1))


def test_doubling_quench_rate_halves_the_quench(tmp_path):
    assert _quench_nsteps(tmp_path, 2e12) * 2 == _quench_nsteps(tmp_path, 1e12)


def test_quench_step_count_is_dT_over_rate_dt(tmp_path):
    """Default (1e12 K/s, 0.002 ps, 2200 K) is exactly 1.1e6 steps; 2e12 gives half."""
    assert _quench_nsteps(tmp_path, 1e12) == 1_100_000
    assert _quench_nsteps(tmp_path, 2e12) == 550_000


def test_nonpositive_quench_rate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='quench_rate'):
        _quench_nsteps(tmp_path, 0.0)


# --- point units (Task 5b) ------------------------------------------------------

def _ingest_load_kinds():
    """Every string literal `ingest` can give a LoadPoint's `load_kind`: the
    `load_kind=` keywords and the first element of `_match_point`'s returns."""
    import ast
    tree = ast.parse((REPO / 'frictionsim2d' / 'aiida' / 'ingest.py').read_text(encoding='utf-8'))

    def strings(node):
        return {n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}

    kinds = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'load_kind':
            kinds |= strings(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == '_match_point':
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple):
                    kinds |= strings(ret.value.elts[0])
    return kinds - {'GPa'}       # the comparand of `'pressure' if unit == 'GPa'`


def test_point_units_cover_every_ingest_load_kind():
    from frictionsim2d.aiida.block_schema import POINT_UNITS
    kinds = _ingest_load_kinds()
    assert kinds >= {'force', 'pressure', 'none'}, kinds
    assert set(POINT_UNITS) >= kinds


def test_point_units_match_the_point_paths():
    from frictionsim2d.aiida.block_schema import POINT_UNITS
    assert POINT_UNITS['force'] == {'load': 'nN', 'angle': 'deg', 'speed': 'm/s'}
    assert POINT_UNITS['pressure']['load'] == 'GPa'
    assert POINT_UNITS['none'] == {}


# --- legacy AFM spring factor (fix round 1) -------------------------------------

def test_a_legacy_afm_record_may_carry_the_old_16_02_factor():
    from frictionsim2d.aiida.block_schema import LEGACY_AFM_SPRING_FACTOR
    assert LEGACY_AFM_SPRING_FACTOR == 16.02
    run = _run(driving_spring=50.0, drive_k=50.0 / 16.02)
    check_units(run, record={'schema_version': '2.1.0'})
    check_units(run, record={'schema_version': '2.2.0',
                             'run_uuid_stamped_at': '2026-11-16'})
    check_units(_run(driving_spring=50.0, drive_k=50.0 / NM_TO_EV_A2),
                record={'schema_version': '2.1.0'})       # exact still passes
    with pytest.raises(UnitMismatch):
        check_units(_run(driving_spring=50.0, drive_k=50.0),
                    record={'schema_version': '2.1.0'})


def test_a_current_afm_record_with_16_02_raises():
    run = _run(driving_spring=50.0, drive_k=50.0 / 16.02)
    with pytest.raises(UnitMismatch):
        check_units(run, record={'schema_version': '2.2.0'})
    with pytest.raises(UnitMismatch):
        check_units(run)                                   # no record: current


def test_a_legacy_sheet_record_with_16_02_still_raises():
    with pytest.raises(UnitMismatch):
        check_units(_run(family='sheetonsheet', driving_spring=50.0,
                         drive_method='smd', drive_k=50.0 / 16.02),
                    record={'schema_version': '2.1.0', 'run_uuid_stamped_at': 'x'})
