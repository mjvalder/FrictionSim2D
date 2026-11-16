"""compare against the store: pk, point uuid and group references."""
import json

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.calcjob import import_run  # noqa: E402
from src.aiida.compare import load_blocks  # noqa: E402
from src.aiida.store import machine_computer  # noqa: E402
from src.core.compare import VARIES, compare_blocks  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.aiida.test_calcjob_import import (  # noqa: E402
    RUN_UUID, _afm_run, _points, _set_record,
)

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
    (loc,) = [r for r in discover_runs(root) if r.run_dir == loc.run_dir]
    return import_run(loc, computer=computer)


def _diff(left, right):
    return compare_blocks(left.blocks, right.blocks,
                          units={**left.units, **right.units},
                          derived_a=left.derived, derived_b=right.derived,
                          defaults={**left.defaults, **right.defaults})


def test_two_runs_differ_in_temp_with_unit(computer, tmp_path):
    _, slide_a = _import(tmp_path / 'a', RUN_UUID, 300, computer)
    _, slide_b = _import(tmp_path / 'b', OTHER_UUID, 400, computer)

    report = _diff(load_blocks(str(slide_a.pk)), load_blocks(slide_b.uuid))
    (diff,) = report.differs['thermostat']
    assert diff == ('general.temp', 300.0, 400.0, 'K')
    assert 'potential__sheet_0' in report.identical or 'material__sheet_0' in report.identical
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
