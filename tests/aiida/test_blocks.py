"""Parameter blocks (spec §5.2): the aiida-free split and its content hash.

No aiida import and no `aiida` marker: this file runs in the fast suite on a
machine without aiida-core.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from src.aiida.blocks import (
    BLOCK_KINDS, EXCLUDED, FIELD_MAP, PROVENANCE_MAP, assigned, block_hash,
    block_kind, leaf_field_paths, split_blocks,
)
from src.core.config import (
    AFMSimulationConfig, GlobalSettings, SheetOnSheetSimulationConfig,
)
from src.data.provenance import (
    BuiltGeometry, ProvenanceRecord, material_provenance,
)

TOP_LEVEL_MODELS = (AFMSimulationConfig, SheetOnSheetSimulationConfig)


def _files(tmp_path):
    pot = tmp_path / 'dummy.sw'
    cif = tmp_path / 'dummy.cif'
    pot.write_text('# dummy potential', encoding='utf-8')
    cif.write_text('# dummy cif', encoding='utf-8')
    return str(pot), str(cif)


def _component(pot, cif, mat):
    return {'mat': mat, 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif}


def _provenance(config_obj):
    """A provenance record built the way the builders build it."""
    sheets = list(config_obj.sheets) or [config_obj.sheet]
    materials = [material_provenance(s, 'sheet') for s in sheets]
    materials += [material_provenance(fs, 'flake') for fs in config_obj.flake_sheets]
    flake = config_obj.flake
    if flake is not None and flake.mat:
        materials.append(material_provenance(flake, 'flake'))
    if getattr(config_obj, 'tip', None) is not None:
        materials.append(material_provenance(config_obj.tip, 'tip'))
    if getattr(config_obj, 'sub', None) is not None:
        materials.append(material_provenance(config_obj.sub, 'substrate'))
    return ProvenanceRecord(
        family='afm', model='afm', materials=materials,
        seed_thermal_tip=1, seed_thermal_sub=2, seed_velocity=3,
        substrate_realization_id=None, damping_ratio=0.1, damp_ev=6.25,
        git_commit='deadbeef', git_dirty=False, config_hash='ff' * 32,
        layer_type_map={'layer_1': [1]}, contact_cutoff=4.8,
        cutoff_source='test', thermostat_scheme='langevin_system',
        load_requested=10.0, drive_k=0.3,
        geometry=BuiltGeometry(lat_c=6.1, cell_a=3.16, cell_b=5.47),
    ).model_dump()


@pytest.fixture
def afm(tmp_path, mock_settings):
    """A full AFM config: every optional section present."""
    pot, cif = _files(tmp_path)
    obj = AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {**_component(pot, cif, 'Si'), 'r': 20.0, 'dspring': 0.1},
        'sub': {**_component(pot, cif, 'Si'), 'thickness': 10.0, 'amorph': 'a'},
        '2D': {**_component(pot, cif, 'h-MoS2'), 'x': 90.0, 'y': 90.0, 'layers': [1]},
        'flake': {**_component(pot, cif, 'h-WS2'), 'edge_length': 10.0},
        'Flake-1': {**_component(pot, cif, 'h-MoS2'), 'x': 30.0, 'y': 30.0,
                    'layers': [1]},
        'pes': {}, 'indent': {}, 'lattice_match': {},
        'lj_override': {},
        'settings': mock_settings.model_dump(),
    })
    return obj.model_dump(), mock_settings.model_dump(), _provenance(obj)


@pytest.fixture
def hetero(tmp_path, mock_settings):
    """A two-material sheet-on-sheet stack, [2D-1] + [2D-2]."""
    pot, cif = _files(tmp_path)
    other_cif = tmp_path / 'other.cif'
    other_cif.write_text('# another cif', encoding='utf-8')
    obj = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D-1': {**_component(pot, cif, 'h-MoS2'), 'x': 100.0, 'y': 100.0,
                 'layers': [2]},
        '2D-2': {**_component(pot, str(other_cif), 'h-WS2'), 'x': 100.0,
                 'y': 100.0, 'layers': [2]},
        'settings': mock_settings.model_dump(),
    })
    return obj.model_dump(), mock_settings.model_dump(), _provenance(obj)


def _diff(a, b):
    assert set(a) == set(b)
    return {k for k in a if block_hash(k, a[k]) != block_hash(k, b[k])}


# --- the split ----------------------------------------------------------------

def test_temperature_only_changes_thermostat(afm):
    cfg, st, prov = afm
    cfg_t400 = copy.deepcopy(cfg)
    cfg_t400['general']['temp'] = 400.0
    assert _diff(split_blocks(cfg, st, prov),
                 split_blocks(cfg_t400, st, prov)) == {'thermostat'}


def test_block_names(afm):
    cfg, st, prov = afm
    names = set(split_blocks(cfg, st, prov))
    assert names == {
        'material__sheet_0', 'material__flake_0', 'material__flake_1',
        'material__tip_0', 'material__substrate_0',
        'potential__sheet_0', 'potential__flake_0', 'potential__flake_1',
        'potential__tip_0', 'potential__substrate_0',
        'thermostat', 'geometry', 'drive', 'protocol', 'output'}
    assert {block_kind(n) for n in names} == set(BLOCK_KINDS)


def test_hetero_has_two_material_blocks(hetero):
    cfg, st, prov = hetero
    blocks = split_blocks(cfg, st, prov)
    mats = sorted(k for k in blocks if block_kind(k) == 'material')
    assert mats == ['material__sheet_0', 'material__sheet_1']
    assert blocks['material__sheet_0']['name'] == 'h-MoS2'
    assert blocks['material__sheet_1']['name'] == 'h-WS2'
    assert (block_hash('material', blocks['material__sheet_0'])
            != block_hash('material', blocks['material__sheet_1']))


def test_material_block_carries_structure_hash_not_path(afm, tmp_path):
    cfg, st, prov = afm
    sheet = split_blocks(cfg, st, prov)['material__sheet_0']
    assert sheet['cif_sha256'] is not None and len(sheet['cif_sha256']) == 64
    assert str(tmp_path) not in repr(sheet)
    assert sheet['built'] == {'lat_c': 6.1, 'cell_a': 3.16, 'cell_b': 5.47}


def test_potential_block_carries_file_hash_not_path(afm, tmp_path):
    cfg, st, prov = afm
    pot = split_blocks(cfg, st, prov)['potential__tip_0']
    assert pot['pot_file_sha256'] == prov['materials'][-2]['pot_file_sha256']
    assert pot['pot_type'] == 'sw'
    assert str(tmp_path) not in repr(pot)


def test_same_material_in_two_roles_hashes_alike(afm):
    """Tip and substrate are both Si from one file: one shared material node."""
    cfg, st, prov = afm
    blocks = split_blocks(cfg, st, prov)
    assert (block_hash('material__tip_0', blocks['material__tip_0'])
            == block_hash('material__substrate_0', blocks['material__substrate_0']))


def test_moved_files_keep_their_hash(afm, tmp_path):
    """A tree read on another machine: the path is gone, the copy in provenance/ is not."""
    cfg, st, prov = afm
    original = split_blocks(cfg, st, prov)
    run_dir = tmp_path / 'elsewhere'
    (run_dir / 'provenance' / 'cif').mkdir(parents=True)
    moved = copy.deepcopy(cfg)
    for key in ('tip', 'sub', 'sheet', 'flake'):
        moved[key]['cif_path'] = '/gone/dummy.cif'
    for item in moved['sheets'] + moved['flake_sheets']:
        item['cif_path'] = '/gone/dummy.cif'
    (run_dir / 'provenance' / 'cif' / 'dummy.cif').write_text(
        '# dummy cif', encoding='utf-8')
    assert _diff(original, split_blocks(moved, st, prov, run_dir=run_dir)) == set()


# --- the hash -----------------------------------------------------------------

def test_block_hash_ignores_key_order_and_float_repr():
    a = {'timestep': 0.001, 'nested': {'x': 1.0, 'y': [2.5, 3]}, 'flag': True}
    b = {'flag': True, 'nested': {'y': [np.float64(2.5), 3], 'x': 1.0},
         'timestep': 1e-3}
    assert block_hash('drive', a) == block_hash('drive', b)
    assert block_hash('drive', a) != block_hash('thermostat', a)
    assert block_hash('drive', a) != block_hash('drive', {**a, 'timestep': 0.002})


def test_block_hash_keeps_ints_and_bools_distinct_from_floats():
    assert block_hash('drive', {'v': 1}) != block_hash('drive', {'v': 1.0})
    assert block_hash('drive', {'v': True}) != block_hash('drive', {'v': 1})


def test_block_hash_kind_is_the_block_kind():
    assert block_hash('material__sheet_0', {'a': 1}) == block_hash('material', {'a': 1})
    with pytest.raises(ValueError):
        block_hash('nonsense', {'a': 1})


# --- coverage (Review Focus 5) ------------------------------------------------

def _all_paths():
    paths = set()
    for model in TOP_LEVEL_MODELS:
        paths |= set(leaf_field_paths(model))
    paths |= {f'settings.{p}' for p in leaf_field_paths(GlobalSettings)}
    return paths


def test_every_config_field_is_assigned():
    for model in TOP_LEVEL_MODELS:
        for path in leaf_field_paths(model):
            assert assigned(path) or path in EXCLUDED, path
    for path in leaf_field_paths(GlobalSettings):
        path = f'settings.{path}'
        assert assigned(path) or path in EXCLUDED, path


def test_leaf_field_paths_reaches_nested_and_list_models():
    paths = set(leaf_field_paths(AFMSimulationConfig))
    for path in ('general.temp', 'tip.r', 'sub.thickness', 'sheets.layers',
                 'flake.edge_length', 'flake_sheets.x', 'pes.grid_n',
                 'indent.z_step', 'lattice_match.strain_tol', 'lj_override',
                 'settings.thermostat.langevin_boundaries',
                 'settings.integrity.hole_bonds'):
        assert path in paths, path
    assert 'settings.thermostat' not in paths      # a model, not a leaf


def test_a_new_field_is_caught():
    from src.core.config import ThermostatSettings  # pylint: disable=import-outside-toplevel

    class Grown(AFMSimulationConfig):
        brand_new: float = 1.0

    class GrownThermostat(ThermostatSettings):
        brand_new: float = 1.0

    class GrownSettings(GlobalSettings):
        thermostat: GrownThermostat

    unassigned = [p for p in leaf_field_paths(Grown)
                  if not assigned(p) and p not in EXCLUDED]
    assert unassigned == ['brand_new']
    unassigned = [p for p in leaf_field_paths(GrownSettings, 'settings.')
                  if not assigned(p) and p not in EXCLUDED]
    assert unassigned == ['settings.thermostat.brand_new']


def test_map_and_exclusions_name_only_real_fields():
    real = _all_paths()
    assert set(FIELD_MAP) <= real, sorted(set(FIELD_MAP) - real)
    assert set(EXCLUDED) <= real, sorted(set(EXCLUDED) - real)
    assert not set(FIELD_MAP) & set(EXCLUDED)
    assert set(FIELD_MAP.values()) <= set(BLOCK_KINDS)
    assert set(PROVENANCE_MAP.values()) <= set(BLOCK_KINDS)


def test_exclusions_are_named_not_physics():
    for section in ('hpc', 'aiida', 'database', 'integrity'):
        assert any(p.startswith(f'settings.{section}.') for p in EXCLUDED), section
    assert not any(p.startswith('settings.thermostat.') for p in EXCLUDED)


# --- the split honours the map, field by field ---------------------------------

def _perturb(value, tmp_path, name):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, dict):
        return {**value, 'perturbed': 1}
    if isinstance(value, list):
        return value + [value[-1] if value else 1]
    if name.endswith('_path') or name == 'interlayer_file':
        other = tmp_path / f'perturbed_{name}'
        other.write_text(f'different {name}', encoding='utf-8')
        return str(other)
    if isinstance(value, str):
        return value + '_perturbed'
    return 1.0                                       # was None


def _set(doc, path, tmp_path):
    head, *rest = path.split('.')
    target = doc[head]
    if isinstance(target, list):
        target = target[0]
    if not rest:
        doc[head] = _perturb(target, tmp_path, head)
        return
    for key in rest[:-1]:
        target = target[key]
    target[rest[-1]] = _perturb(target[rest[-1]], tmp_path, rest[-1])


# Component potential fields the block reads from provenance (pot_type,
# pot_file_sha256) rather than from config.json, by design: provenance is
# what the build resolved, config.json is what was asked for.
_FROM_PROVENANCE = ('pot_path', 'pot_type')
# `sheet` mirrors `sheets[0]`; the split reads `sheets`.
_MIRROR = 'sheet.'


@pytest.mark.parametrize('path', sorted(FIELD_MAP))
def test_each_mapped_field_changes_only_its_block(afm, tmp_path, path):
    cfg, st, prov = afm
    if path.startswith(_MIRROR) or path.rsplit('.', 1)[-1] in _FROM_PROVENANCE:
        pytest.skip('read from sheets / provenance')
    kind = FIELD_MAP[path]
    cfg2, st2 = copy.deepcopy(cfg), copy.deepcopy(st)
    if path.startswith('settings.'):
        _set(st2, path[len('settings.'):], tmp_path)
    else:
        _set(cfg2, path, tmp_path)
    changed = _diff(split_blocks(cfg, st, prov), split_blocks(cfg2, st2, prov))
    assert changed, path
    assert {block_kind(n) for n in changed} == {kind}, (path, changed)


@pytest.mark.parametrize('path', sorted(p for p in EXCLUDED if p.startswith('settings.')))
def test_excluded_settings_change_nothing(afm, tmp_path, path):
    cfg, st, prov = afm
    st2 = copy.deepcopy(st)
    _set(st2, path[len('settings.'):], tmp_path)
    assert _diff(split_blocks(cfg, st, prov), split_blocks(cfg, st2, prov)) == set()


@pytest.mark.parametrize('field', sorted(PROVENANCE_MAP))
def test_each_provenance_field_changes_only_its_block(afm, field):
    cfg, st, prov = afm
    prov2 = copy.deepcopy(prov)
    if field.startswith('geometry.'):
        prov2['geometry'][field.split('.', 1)[1]] += 1.0
    elif field.startswith('materials.'):
        for entry in prov2['materials']:
            entry[field.split('.', 1)[1]] = 'perturbed'
    elif field == 'thermostat_scheme':
        prov2[field] += '_perturbed'
    else:
        prov2[field] += 1.0
    changed = _diff(split_blocks(cfg, st, prov), split_blocks(cfg, st, prov2))
    assert changed and {block_kind(n) for n in changed} == {PROVENANCE_MAP[field]}


def test_settings_default_to_the_config_copy(afm):
    cfg, st, prov = afm
    assert _diff(split_blocks(cfg, st, prov), split_blocks(cfg, {}, prov)) == set()


def test_split_is_json_plain(afm):
    import json  # pylint: disable=import-outside-toplevel
    cfg, st, prov = afm
    json.dumps(split_blocks(cfg, st, prov))         # Dict-node storable


def test_blocks_module_does_not_import_aiida():
    import ast  # pylint: disable=import-outside-toplevel
    path = Path(__file__).resolve().parents[2] / 'src' / 'aiida' / 'blocks.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names]
    names += [node.module or '' for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.level == 0]
    assert not [n for n in names if n.split('.')[0] == 'aiida'], names
