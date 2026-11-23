"""A store written before the `src` -> `frictionsim2d` rename still loads.

`tests/fixtures/pre_rename_store.aiida` is an archive of one imported legacy
tree (rebomos). Node types are stored by entry point, so it must round-trip
through the renamed package; `pre_rename_store_rows.json` is its point table
as recorded before the rename.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402
from aiida.tools.archive import import_archive  # noqa: E402

from frictionsim2d.aiida.query import Friction2DDB  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'


def test_old_nodes_load(aiida_profile_clean):
    import_archive(FIXTURES / 'pre_rename_store.aiida')
    table = Friction2DDB.point_table()
    rows = json.loads(table.to_json(orient='records', default_handler=str))
    expected = json.loads((FIXTURES / 'pre_rename_store_rows.json').read_text(encoding='utf-8'))
    assert rows == expected
    calc = orm.load_node(rows[0]['calc_uuid'])
    assert calc.is_finished_ok
    assert 'points' in calc.outputs
    assert [p.base.attributes.get('load') for p in calc.outputs.points.values()] == [1.0]
