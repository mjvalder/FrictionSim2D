"""compare against the store: pk, point uuid and group references."""
import json

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from frictionsim2d.aiida.calcjob import import_run  # noqa: E402
from frictionsim2d.aiida.compare import load_blocks  # noqa: E402
from frictionsim2d.aiida.store import machine_computer  # noqa: E402
from frictionsim2d.core.compare import VARIES, compare_blocks  # noqa: E402
from frictionsim2d.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import seal_config_hash  # noqa: E402
from tests.aiida.test_calcjob_import import (  # noqa: E402
    RUN_UUID, _afm_run, _points, _set_record,
)

IDENTICAL_BLOCKS = ['drive', 'geometry', 'integration', 'interaction', 'material__sheet_0',
                    'output', 'potential__sheet_0', 'protocol', 'realization']
OTHER_UUID = '7a1d9c3e-2f44-4b58-8d3b-0c9e5f1a6b22'


@pytest.fixture
def computer(aiida_profile_clean):
    return machine_computer('localhost', {})


def _import(root, run_uuid, temp, computer):
    loc = _afm_run(root)
    _set_record(loc, run_uuid=run_uuid)
    config = loc.run_dir.parent / 'provenance' / 'config.json'
    if not config.is_file():
        config = loc.run_dir / 'provenance' / 'config.json'
    data = json.loads(config.read_text(encoding='utf-8'))
    data.setdefault('general', {})['temp'] = temp
    config.write_text(json.dumps(data), encoding='utf-8')
    seal_config_hash(loc.run_dir, config.parent)
    (loc,) = [r for r in discover_runs(root) if r.run_dir == loc.run_dir]
    return import_run(loc, computer=computer)


def _diff(left, right):
    return compare_blocks(left.blocks, right.blocks,
                          units=left.units, units_b=right.units,
                          derived_a=left.derived, derived_b=right.derived,
                          defaults=left.defaults, defaults_b=right.defaults,
                          kinds=left.kinds)


def test_two_runs_differ_in_temp_with_unit(computer, tmp_path):
    _, slide_a = _import(tmp_path / 'a', RUN_UUID, 300, computer)
    _, slide_b = _import(tmp_path / 'b', OTHER_UUID, 400, computer)

    report = _diff(load_blocks(str(slide_a.pk)), load_blocks(slide_b.uuid))
    (diff,) = report.differs['thermostat']
    assert diff == ('general.temp', 300.0, 400.0, 'K')
    assert sorted(report.identical) == IDENTICAL_BLOCKS
    assert report.only_a == [] and report.only_b == [] and report.derived_differs == []
    assert 'K' in report.render()


def test_point_uuid_resolves_to_its_calcjob(computer, tmp_path):
    _, slide = _import(tmp_path / 'a', RUN_UUID, 300, computer)
    puuid = _points(slide)['force10_a0_s2'].base.attributes.get('point_uuid')
    assert load_blocks(puuid).blocks == load_blocks(str(slide.pk)).blocks


def test_group_reports_varying_fields(computer, tmp_path):
    _, slide_a = _import(tmp_path / 'a', RUN_UUID, 300, computer)
    _, slide_b = _import(tmp_path / 'b', OTHER_UUID, 400, computer)
    group = orm.Group(label='friction2d/campaign').store()
    group.add_nodes([slide_a, slide_b])

    for ref in ('campaign', 'friction2d/campaign'):
        merged = load_blocks(ref)
        assert merged.blocks['thermostat']['general.temp'] == VARIES


def test_unknown_reference_is_a_lookup_error(aiida_profile_clean):
    with pytest.raises(LookupError):
        load_blocks('no-such-thing')


def test_group_excludes_superseded_calcjobs(computer, tmp_path):
    _, slide_a = _import(tmp_path / 'a', RUN_UUID, 300, computer)
    _, slide_b = _import(tmp_path / 'b', OTHER_UUID, 400, computer)
    group = orm.Group(label='friction2d/campaign').store()
    group.add_nodes([slide_a, slide_b])
    slide_b.base.extras.set('superseded', True)
    merged = load_blocks('campaign')
    assert merged.blocks['thermostat']['general.temp'] == 300.0
    slide_a.base.extras.set('superseded', True)
    with pytest.raises(LookupError):
        load_blocks('campaign')


def test_point_uuid_takes_the_current_node_and_rejects_ambiguity(computer, tmp_path):
    import shutil
    first = _import(tmp_path / 'a', RUN_UUID, 300, computer)[1]
    shutil.copytree(tmp_path / 'a', tmp_path / 'moved')
    (moved,) = discover_runs(tmp_path / 'moved')
    second = import_run(moved, computer=computer)[1]
    point_a, point_b = (_points(n)['force10_a0_s2'] for n in (first, second))
    puuid = point_a.base.attributes.get('point_uuid')
    assert puuid == point_b.base.attributes.get('point_uuid')
    with pytest.raises(LookupError, match='current nodes'):
        load_blocks(puuid)
    point_a.base.extras.set('superseded', True)
    assert load_blocks(puuid).blocks == load_blocks(str(second.pk)).blocks


def test_reference_precedence_and_prefixes(computer, tmp_path):
    _, slide_a = _import(tmp_path / 'a', RUN_UUID, 350, computer)
    _, slide_b = _import(tmp_path / 'b', OTHER_UUID, 400, computer)
    group = orm.Group(label=str(slide_a.pk)).store()   # a group named like a pk
    group.add_nodes([slide_b])
    assert load_blocks(str(slide_a.pk)).blocks['thermostat']['general.temp'] == 350.0
    assert load_blocks(f'pk:{slide_a.pk}').blocks['thermostat']['general.temp'] == 350.0
    assert load_blocks(f'group:{slide_a.pk}').blocks['thermostat']['general.temp'] == 400.0
    assert load_blocks(f'uuid:{slide_b.uuid}').blocks['thermostat']['general.temp'] == 400.0
    with pytest.raises(LookupError):
        load_blocks(f'uuid:{slide_a.pk}')


def _array_mode(loc):
    (loc.lammps_dir / 'slide.in').unlink()
    for name in ('slide_f10N', 'slide_f30N'):
        (loc.lammps_dir / f'{name}.in').write_text('# f\n', encoding='utf-8')
    return loc


def test_array_and_combined_decks_of_one_physics_do_not_differ(computer, tmp_path):
    combined = _import(tmp_path / 'c', RUN_UUID, 300, computer)
    loc = _afm_run(tmp_path / 'arr')
    _set_record(loc, run_uuid=OTHER_UUID)
    array = import_run(_array_mode(loc), computer=computer)
    assert len(array) == 3 and len(combined) == 2
    gc, ga = (orm.Group(label=f'friction2d/{n}').store() for n in ('c', 'arr'))
    gc.add_nodes(combined)
    ga.add_nodes(array)
    report = _diff(load_blocks('c'), load_blocks('arr'))
    assert report.same, report.render()
    assert VARIES not in str(load_blocks('arr').blocks)


# --- merging and sidecar schema selection (no store needed) --------------------

def test_merge_refuses_blocks_stored_in_different_units():
    from frictionsim2d.aiida.compare import LoadedBlocks, _merge

    a = LoadedBlocks({'drive': {'x': 1.0}}, {}, {'drive.x': 'nm'}, {}, {'drive.x': 'drive'})
    b = LoadedBlocks({'drive': {'x': 1.0}}, {}, {'drive.x': 'angstrom'}, {}, {'drive.x': 'drive'})
    with pytest.raises(ValueError, match='different units for drive.x'):
        _merge([a, b])


def test_a_sidecar_is_read_against_its_own_schema_else_the_newest(tmp_path):
    from frictionsim2d.aiida.compare import _schema_for

    for version in ('1.0.0', '1.10.0', '1.2.0'):
        (tmp_path / f'schema-{version}.json').write_text('{}', encoding='utf-8')

    assert _schema_for(tmp_path, ['1.2.0']).name == 'schema-1.2.0.json'
    assert _schema_for(tmp_path, ['9.9.9']).name == 'schema-1.10.0.json'   # newest, by version
    assert _schema_for(tmp_path / 'nowhere', ['1.0.0']) is None
