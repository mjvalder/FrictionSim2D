"""Searching structures and potentials: workspace first, then bundled examples."""
from pathlib import Path

import pytest

from src.agent.changes import ToolInputError
from src.agent.tools.materials import (build_index, elements_of, material_ops,
                                       potential_type_of, resolve_material,
                                       search_potentials, search_structures)
from tests.helpers.agent import EXAMPLES, make_deps


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


@pytest.mark.parametrize('name, expected', [
    ('h-MoS2', {'Mo', 'S'}), ('MoS2_wen', {'Mo', 'S'}), ('MoS', {'Mo', 'S'}),
    ('ffield_Wang.reax.MoS', {'Mo', 'S'}), ('graphene', set()), ('CH', {'C', 'H'}),
])
def test_elements_of(name, expected):
    assert elements_of(name) == frozenset(expected)


@pytest.mark.parametrize('rel, expected', [
    ('potentials/sw/MoS2_wen.sw', 'sw'), ('potentials/rebo/MoS.rebomos', 'rebomos'),
    ('potentials/reaxff/ffield_Wang.reax.MoS', 'reaxff'), ('materials/h-MoS2.cif', None),
])
def test_potential_type_of(rel, expected):
    assert potential_type_of(EXAMPLES / rel) == expected


def test_index_finds_examples(deps):
    index = build_index(deps.roots)
    assert any(f.name == 'h-MoS2' for f in index.structures)
    assert any(f.pot_type == 'sw' and f.name == 'h-MoS2' for f in index.potentials)


def test_workspace_tier_listed_first(deps):
    mine = deps.roots.workspace / 'cifs'
    mine.mkdir()
    (mine / 'h-MoS2.cif').write_text('x')
    text = search_structures(deps, 'MoS2')
    assert text.splitlines()[0].endswith('workspace:/cifs/h-MoS2.cif')


def test_search_structures_reports_tier(deps):
    assert 'examples:/materials/h-WS2.cif' in search_structures(deps, 'WS2')


def test_search_structures_no_match(deps):
    assert search_structures(deps, 'unobtainium').startswith('no structure')


def test_search_potentials_by_type(deps):
    text = search_potentials(deps, 'h-MoS2', 'sw')
    assert 'sw' in text and 'h-MoS2.sw' in text
    assert 'rebomos' not in text


def test_resolve_material_prefers_exact_stem(deps):
    structure, potential = resolve_material(deps, 'h-MoS2', 'sw')
    assert structure.path == EXAMPLES / 'materials' / 'h-MoS2.cif'
    assert potential.path.name == 'h-MoS2.sw'


def test_resolve_unknown_material(deps):
    with pytest.raises(ToolInputError, match='search_structures'):
        resolve_material(deps, 'unobtainium', 'sw')


def test_single_material_ops_set_paths(deps):
    ops, notes = material_ops(deps, ['h-MoS2'], 'sw')
    by_key = {op.key: op.value for op in ops if op.kind == 'set'}
    assert by_key['mat'] == 'h-MoS2' and by_key['pot_type'] == 'sw'
    assert Path(by_key['cif_path']).name == 'h-MoS2.cif'
    assert notes and 'examples:/' in notes[0]


def test_sweep_ops_use_material_directories(deps):
    ops, _ = material_ops(deps, ['h-MoS2', 'h-WS2'], 'sw')
    kinds = [op.kind for op in ops]
    assert kinds[:2] == ['material_dirs', 'materials']
    assert ops[1].value == ['h-MoS2', 'h-WS2']
    assert ops[0].value[2] == '.sw'


def test_sweep_needs_one_folder_each(deps):
    other = deps.roots.workspace / 'cifs'
    other.mkdir()
    (other / 'h-MoSe2.cif').write_text('x')
    with pytest.raises(ToolInputError, match='one structure folder'):
        material_ops(deps, ['h-MoS2', 'h-MoSe2'], 'sw')


def test_reaxff_found_by_material(deps):
    assert 'ffield_Wang.reax.MoS' in search_potentials(deps, 'h-MoS2', 'reaxff')
    assert resolve_material(deps, 'h-MoS2', 'reaxff')[1].pot_type == 'reaxff'


def test_cap_counts_only_candidates_and_per_tier(deps, monkeypatch):
    for i in range(800):
        (deps.roots.workspace / f'out{i}.log').write_text('x')
    (deps.roots.workspace / 'zz').mkdir()
    (deps.roots.workspace / 'zz' / 'h-MoS2.cif').write_text('x')
    deps.material_index = build_index(deps.roots, max_files=400)
    text = search_structures(deps, 'MoS2')
    assert 'workspace:/zz/h-MoS2.cif' in text and 'examples:/materials/h-MoS2.cif' in text


def test_cap_truncation_reported_and_examples_walked(deps, monkeypatch):
    for i in range(410):
        (deps.roots.workspace / f'a{i}.cif').write_text('x')
    deps.material_index = build_index(deps.roots, max_files=400)
    text = search_structures(deps, 'WS2')
    assert 'examples:/materials/h-WS2.cif' in text
    assert 'workspace search stopped after 400 files' in text
