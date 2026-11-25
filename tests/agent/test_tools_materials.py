"""Searching structures and potentials: workspace first, then bundled examples."""
from pathlib import Path

import pytest

from frictionsim2d.agent.changes import ToolInputError
from frictionsim2d.agent.tools.materials import (build_index, elements_of, folder_material_ops, lattice_mismatch,
                                       material_ops, potential_type_of, resolve_material,
                                       search_potentials, search_structures)
from frictionsim2d.agent.tools.setup import AFMSetupRequest, setup_afm
from frictionsim2d.core.config import LatticeMatchConfig
from frictionsim2d.core.utils.structures import cifread
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


def test_cap_counts_only_candidates_and_per_tier(deps):
    for i in range(800):
        (deps.roots.workspace / f'out{i}.log').write_text('x')
    (deps.roots.workspace / 'zz').mkdir()
    (deps.roots.workspace / 'zz' / 'h-MoS2.cif').write_text('x')
    deps.material_index = build_index(deps.roots, max_files=400)
    text = search_structures(deps, 'MoS2')
    assert 'workspace:/zz/h-MoS2.cif' in text and 'examples:/materials/h-MoS2.cif' in text


def test_cap_truncation_reported_and_examples_walked(deps):
    for i in range(410):
        (deps.roots.workspace / f'a{i}.cif').write_text('x')
    deps.material_index = build_index(deps.roots, max_files=400)
    text = search_structures(deps, 'WS2')
    assert 'examples:/materials/h-WS2.cif' in text
    assert 'workspace search stopped after 400 files' in text


# ---- lattice_mismatch ------------------------------------------------------------

def _cell(name):
    return cifread(EXAMPLES / 'materials' / f'{name}.cif')


def test_lattice_mismatch_reports_both_cells_and_the_mismatch(deps):
    text = lattice_mismatch(deps, 'h-MoS2', 'h-WS2')
    mos2, ws2 = _cell('h-MoS2'), _cell('h-WS2')
    expected = abs(mos2['lat_a'] - ws2['lat_a']) / ((mos2['lat_a'] + ws2['lat_a']) / 2) * 100
    first = text.splitlines()[0]
    assert 'h-MoS2' in first and 'h-WS2' in first and f'{expected:.2f} %' in first
    assert f"a {mos2['lat_a']:.3f}" in text and f"a {ws2['lat_a']:.3f}" in text
    assert 'coincidence cell' in text and '[[1, 0], [0, 1]]' in text
    assert '/home' not in text and len(text) <= 1500


def test_lattice_mismatch_is_relative_to_the_mean_cell(deps):
    """A mismatch large enough to tell mean-relative (4.17 %) from relative
    to either cell (4.26 / 4.09 %); a literal, not re-derived."""
    first = lattice_mismatch(deps, 'h-MoS2', 'h-MoSe2').splitlines()[0]
    assert '4.17 %' in first


def test_lattice_mismatch_numbers_fit_in_the_first_200_chars(deps):
    """The eval's grounding check sees tool results cut to 200 characters."""
    text = lattice_mismatch(deps, 'h-MoS2', 'h-WS2')
    mos2 = _cell('h-MoS2')
    assert f"{mos2['lat_a']:.3f}" in text[:200] and ' %' in text[:200]


def test_lattice_mismatch_one_material_reports_its_cell(deps):
    text = lattice_mismatch(deps, 'h-WS2')
    ws2 = _cell('h-WS2')
    assert f"a {ws2['lat_a']:.3f}" in text and 'mismatch' not in text


def test_lattice_mismatch_different_lattices_say_so(deps):
    text = lattice_mismatch(deps, 'h-MoS2', 'black_phosphorus')
    assert 'gamma' in text and 'different in-plane lattice' in text


def test_lattice_mismatch_unknown_material_is_input_error(deps):
    with pytest.raises(ToolInputError, match='search_structures'):
        lattice_mismatch(deps, 'h-MoS2', 'unobtainium')


def test_lattice_mismatch_searches_as_the_hetero_builder_does(deps):
    """Same bounds as LatticeMatchConfig, so the tool never reports 'no cell' for a
    pair the builder would match."""
    defaults = LatticeMatchConfig()
    text = lattice_mismatch(deps, 'h-MoS2', 'black_phosphorus')
    assert f'<= {defaults.max_supercell} repeats' in text
    assert f'strain <= {defaults.strain_tol * 100:g} %' in text


def test_a_folder_sweeps_every_structure_with_a_matching_potential(deps):
    ops, notes = folder_material_ops(deps, 'examples:/materials', 'sw')
    kinds = [op.kind for op in ops]
    assert kinds[:2] == ['material_dirs', 'materials']
    names = ops[1].value
    potentials = {p.stem for p in (deps.roots.examples / 'potentials/sw/sw_lammps').glob('*.sw')}
    assert set(names) == {p.stem for p in (deps.roots.examples / 'materials').glob('*.cif')} & potentials
    assert len(names) > 100 and ops[0].value[2] == '.sw'
    assert f'{len(names)} materials from examples:/materials' in notes[0]


def test_a_folder_names_what_it_skipped(deps):
    folder = deps.roots.workspace / 'mats'
    folder.mkdir()
    for name in ('h-MoS2', 'h-WS2', 'x-Unknown'):
        (folder / f'{name}.cif').write_text((deps.roots.examples / 'materials/h-MoS2.cif').read_text())
    ops, notes = folder_material_ops(deps, 'workspace:/mats', 'sw')
    assert ops[1].value == ['h-MoS2', 'h-WS2']
    assert 'x-Unknown' in notes[-1]


@pytest.mark.parametrize('folder, message', [('workspace:/', 'no structure'),
                                             ('/etc', '.'), ('workspace:/nothere', '.')])
def test_a_folder_that_cannot_be_swept_is_refused(deps, folder, message):
    with pytest.raises(ToolInputError, match=message):
        folder_material_ops(deps, folder, 'sw')


def test_setup_takes_materials_or_a_folder_not_both():
    with pytest.raises(ValueError):
        AFMSetupRequest(materials=['h-MoS2'], materials_folder='workspace:/m')
    with pytest.raises(ValueError):
        AFMSetupRequest()
    assert AFMSetupRequest(materials_folder='workspace:/m').describe() == \
        'every material in workspace:/m'


def test_setup_afm_over_a_folder_proposes_the_sweep(deps):
    text = setup_afm(deps, AFMSetupRequest(materials_folder='examples:/materials',
                                           potential_type='sw'))
    assert 'proposed' in text and 'every material in examples:/materials' in text


def test_a_potential_folder_is_named_as_such_and_says_to_ask(deps):
    folder = deps.roots.workspace / 'pots'
    folder.mkdir()
    (folder / 'h-MoS2.sw').write_text('x')
    with pytest.raises(ToolInputError, match='holds potential files.*ask the user'):
        folder_material_ops(deps, 'workspace:/pots', 'sw')


def test_a_missing_structure_says_to_ask_and_explains_amorphous(deps):
    from frictionsim2d.agent.tools.materials import resolve_material
    with pytest.raises(ToolInputError, match='built amorphous.*ask the user'):
        resolve_material(deps, 'amorphous Si', 'sw')
