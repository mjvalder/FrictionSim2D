"""Materials Project and 2DMatPedia tools, over a fake client (no network)."""
import pytest

from frictionsim2d.agent.changes import FsChangeSet, apply_fs_changeset, describe_change
from frictionsim2d.agent.materials_db import NO_KEY, MaterialsDBError
from frictionsim2d.agent.tools.materials_db import (fetch_2dmatpedia_structure, query_2dmatpedia,
                                          query_materials_project)
from tests.helpers.agent import CIF_TEXT, make_deps, materials_factory
from tests.helpers.agent import FakeMaterialsClient as FakeClient


def _mp_row(i):
    return {'material_id': f'mp-{i}', 'formula': 'MoS2', 'e_above_hull': 0.01 * i,
            'formation_energy': -0.9, 'band_gap': 1.2, 'is_metal': False,
            'density': 5.0, 'bulk_modulus': 50.0, 'shear_modulus': None,
            'spacegroup': 'P6_3/mmc'}


def _dm_row(i, formula='MoS2'):
    return {'dm_id': f'2dm-{i}', 'mp_id': 'mp-2815', 'formula': formula,
            'structure_id': f'sid{i}', 'process': 'top-down exfoliation',
            'E': -7.1, 'Ed': 0.07, 'Ex': 0.02, 'dE': 0.01, 'Evdw': -0.3, 'mu': 0.0}


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


def _with(deps, client):
    deps.materials_db = materials_factory(client)
    return deps


def test_mp_table_is_compact_and_capped(deps):
    text = query_materials_project(_with(deps, FakeClient(mp_rows=[_mp_row(i)
                                                                   for i in range(25)])), 'MoS2')
    assert 'mp-0' in text and 'mp-9' in text and 'mp-10' not in text
    assert '25 entries' in text and len(text) <= 1500
    assert 'P6_3/mmc' in text


def test_mp_no_match(deps):
    assert 'no Materials Project entry' in query_materials_project(
        _with(deps, FakeClient()), 'XeF9')


def test_2dmatpedia_table(deps):
    text = query_2dmatpedia(_with(deps, FakeClient(dm_rows=[_dm_row(i) for i in range(12)])),
                            'MoS2')
    assert '2dm-0' in text and '2dm-10' not in text and 'mp-2815' in text
    assert 'fetch_2dmatpedia_structure' in text and len(text) <= 1500


def test_no_key_is_reported_as_text(deps):
    def factory(should_stop=None):  # pylint: disable=unused-argument
        raise MaterialsDBError(NO_KEY)
    deps.materials_db = factory
    assert query_materials_project(deps, 'MoS2') == NO_KEY
    assert NO_KEY == ('No Materials Project key: set MP_API_KEY or put it in '
                      '~/.mp_api_key')


def test_client_errors_become_text(deps):
    client = FakeClient(error=MaterialsDBError('Materials Project key rejected'))
    assert query_2dmatpedia(_with(deps, client), 'MoS2') == 'Materials Project key rejected'


def test_no_factory_is_reported(deps):
    deps.materials_db = None
    assert 'not available' in query_materials_project(deps, 'MoS2')


def test_fetch_proposes_write_and_applies(deps):
    client = FakeClient(dm_rows=[_dm_row(5)])
    handle = fetch_2dmatpedia_structure(_with(deps, client), '2dm-5')
    assert client.structures == ['sid5']
    change = deps.pending[-1]
    assert isinstance(change, FsChangeSet) and change.ok, change.issues
    assert change.id in handle and 'do not say it is done' in handle
    assert [op[0] for op in change.ops] == ['mkdir', 'write_text']
    assert change.shown[-1] == ('write_text', 'workspace:/materials/MoS2_2dm-5.cif')
    ws = deps.roots.workspace
    assert apply_fs_changeset(change, ws).applied
    assert (ws / 'materials' / 'MoS2_2dm-5.cif').read_text() == CIF_TEXT


def test_fetch_unknown_id(deps):
    text = fetch_2dmatpedia_structure(_with(deps, FakeClient(dm_rows=[_dm_row(5)])), '2dm-6')
    assert 'no 2DMatPedia entry 2dm-6' in text and not deps.pending


def test_fetch_never_overwrites(deps):
    ws = deps.roots.workspace
    (ws / 'materials').mkdir()
    (ws / 'materials' / 'MoS2_2dm-5.cif').write_text('mine')
    client = FakeClient(dm_rows=[_dm_row(5)])
    fetch_2dmatpedia_structure(_with(deps, client), '2dm-5')
    assert client.structures == []  # refused before any structure request
    change = deps.pending[-1]
    assert not change.ok and 'already exists' in change.issues[0]
    assert not apply_fs_changeset(change, ws).applied
    assert (ws / 'materials' / 'MoS2_2dm-5.cif').read_text() == 'mine'


@pytest.mark.parametrize('dest', ['../outside', '.preview', 'examples:/materials'])
def test_fetch_confined_to_workspace(deps, dest):
    client = FakeClient(dm_rows=[_dm_row(5)])
    fetch_2dmatpedia_structure(_with(deps, client), '2dm-5', dest)
    assert not deps.pending[-1].ok and client.structures == []


def test_tools_pass_the_turn_cancellation(deps):
    client = FakeClient()
    deps.cancelled = lambda: True
    query_2dmatpedia(_with(deps, client), 'MoS2')
    assert client.should_stop is deps.cancelled


def test_first_download_is_mentioned(deps):
    client = FakeClient(dm_rows=[_dm_row(1)])
    client.downloaded_listing = True
    text = query_2dmatpedia(_with(deps, client), 'MoS2')
    assert text.startswith('Downloaded the 2DMatPedia listing (first use; cached for 7 days)')
    client.downloaded_listing = False
    assert not query_2dmatpedia(deps, 'MoS2').startswith('Downloaded')


def test_fetch_card_describes_the_structure_not_the_cif(deps):
    pytest.importorskip('pymatgen.core')
    from pymatgen.core import Lattice, Structure  # pylint: disable=import-outside-toplevel
    from pymatgen.io.cif import CifWriter  # pylint: disable=import-outside-toplevel
    cif = str(CifWriter(Structure(Lattice.hexagonal(3.19, 20.0), ['Mo', 'S', 'S'],
                                  [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.42],
                                   [2 / 3, 1 / 3, 0.58]])))
    fetch_2dmatpedia_structure(_with(deps, FakeClient(dm_rows=[_dm_row(5)], cif=cif)), '2dm-5')
    change = deps.pending[-1]
    text = describe_change(change)
    assert 'Structure: MoS2, 3 sites, a=3.190 b=3.190 c=20.000 Å' in text
    assert '_cell_length_a' not in text and '_atom_site' not in text
    assert change.notes == ['Structure: MoS2, 3 sites, a=3.190 b=3.190 c=20.000 Å']
