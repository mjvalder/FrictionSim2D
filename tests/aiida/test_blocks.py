"""Parameter blocks (spec §5.2): the aiida-free split and its content hash.

No aiida import and no `aiida` marker: this file runs in the fast suite on a
machine without aiida-core.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from frictionsim2d.aiida import block_schema
from frictionsim2d.aiida.block_schema import (
    BLOCK_KINDS, EXCLUDED, FIELD_MAP, PROVENANCE_EXCLUDED, PROVENANCE_MAP,
    PROVENANCE_SOURCED, SWEPT, RunBlocks, defaults_for, leaf_field_paths,
)
from frictionsim2d.aiida.blocks import (
    UNRESOLVED, block_hash, block_kind, split_blocks, upgrade_block,
)
from frictionsim2d.core.config import (
    AFMSimulationConfig, GlobalSettings, SheetOnSheetSimulationConfig,
)
from frictionsim2d.data.provenance import (
    BuiltGeometry, ProvenanceRecord, material_provenance,
)

TOP_LEVEL_MODELS = (AFMSimulationConfig, SheetOnSheetSimulationConfig)


def assigned(path):
    """Is the field at ``path`` assigned a block, a swept axis, or provenance?"""
    return path in FIELD_MAP or path in PROVENANCE_SOURCED or path in SWEPT


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


def _split(cfg, st, prov, **kwargs):
    kwargs.setdefault('family', 'afm')
    return split_blocks(cfg, st, prov, **kwargs)


def _hashes(run):
    return {name: block_hash(content) for name, content in run.blocks.items()}


def _diff(a, b):
    """Names of the blocks whose hash differs between two splits."""
    ha, hb = _hashes(a), _hashes(b)
    assert set(ha) == set(hb)
    return {k for k in ha if ha[k] != hb[k]}


def _with(doc, path, value):
    """A deep copy of ``doc`` with the dotted ``path`` set to ``value``."""
    doc = copy.deepcopy(doc)
    *parents, last = path.split('.')
    target = doc
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[last] = value
    return doc


# --- the split ----------------------------------------------------------------

def test_temperature_only_changes_thermostat(afm):
    cfg, st, prov = afm
    assert _diff(_split(cfg, st, prov),
                 _split(_with(cfg, 'general.temp', 400.0), st, prov)) == {'thermostat'}


def test_block_names(afm):
    cfg, st, prov = afm
    run = _split(cfg, st, prov)
    assert isinstance(run, RunBlocks) and run.family == 'afm'
    assert set(run.blocks) == {
        'material__sheet_0', 'material__flake_0', 'material__flake_1',
        'material__tip_0', 'material__substrate_0',
        'potential__sheet_0', 'potential__flake_0', 'potential__flake_1',
        'potential__tip_0', 'potential__substrate_0',
        'interaction', 'thermostat', 'integration', 'drive', 'geometry',
        'protocol', 'output', 'realization'}
    assert {block_kind(n) for n in run.blocks} == set(BLOCK_KINDS)


def test_kind_and_schema_inside_content(afm):                 # C2
    cfg, st, prov = afm
    for name, content in _split(cfg, st, prov).blocks.items():
        assert set(content) == {'kind', 'block_schema', 'fields'}, name
        assert content['kind'] == block_kind(name)
        assert content['block_schema'] == '1.0.0'


def test_load_sweep_not_in_blocks(afm):                       # C1a
    cfg, st, prov = afm
    a = _split(_with(cfg, 'general.force', [10, 20, 30]), st, prov, n_layers=1)
    b = _split(_with(cfg, 'general.force', [10, 20, 30, 40]), st, prov, n_layers=1)
    assert _hashes(a) == _hashes(b)


@pytest.mark.parametrize('path, value', [
    ('general.force', [5.0, 50.0]), ('general.pressure', [0.1, 0.2]),
    ('general.scan_angle', [0.0, 30.0, 60.0]), ('general.scan_speed', [1.0, 4.0]),
    ('general.outer_loop', 'force'), ('general.scan_angle_force', [10.0]),
])
def test_swept_axes_and_deck_shape_are_not_identity(afm, path, value):
    cfg, st, prov = afm
    assert _diff(_split(cfg, st, prov), _split(_with(cfg, path, value), st, prov)) == set()


def test_outer_loop_is_not_identity(afm):                     # array vs combined
    cfg, st, prov = afm
    array = _with(_with(cfg, 'general.outer_loop', 'force'), 'general.force', [10, 30])
    combined = _with(cfg, 'general.force', [10, 30])
    assert _hashes(_split(array, st, prov)) == _hashes(_split(combined, st, prov))


def test_timestep_only_changes_integration(afm):              # C1b
    cfg, st, prov = afm
    assert _diff(_split(cfg, st, prov),
                 _split(cfg, _with(st, 'simulation.timestep', 0.0005), prov)) \
        == {'integration'}


def test_thermo_and_quench_placement(afm):
    cfg, st, prov = afm
    base = _split(cfg, st, prov)
    assert _diff(base, _split(cfg, _with(st, 'simulation.thermo', 7), prov)) == {'output'}
    assert _diff(base, _split(cfg, _with(st, 'quench.timestep', 0.003), prov)) \
        == {'geometry'}
    assert _diff(base, _split(_with(cfg, 'tip.dspring', 0.2), st, prov)) == {'thermostat'}
    assert _diff(base, _split(_with(cfg, 'sheets.0.interlayer_g', 4.0), st, prov)) \
        == {'interaction'}


def test_potential_swap_changes_only_potential(afm):          # C1c
    cfg, st, prov = afm
    prov_a = _with(_with(prov, 'materials.0.pot_file_sha256', 'a' * 64),
                   'geometry.lat_c', 3.19)
    prov_a = _with(prov_a, 'geometry.lat_c_source', 'minimised')
    prov_b = _with(_with(prov_a, 'materials.0.pot_file_sha256', 'b' * 64),
                   'geometry.lat_c', 3.21)
    a, b = _split(cfg, st, prov_a), _split(cfg, st, prov_b)
    assert _diff(a, b) == {'potential__sheet_0'}
    assert a.derived['geometry.lat_c'] == 3.19 and b.derived['geometry.lat_c'] == 3.21


def test_interlayer_change_leaves_material_potentials_alone(afm):   # M6
    cfg, st, prov = afm
    changed = _diff(_split(cfg, st, prov),
                    _split(_with(cfg, 'general.interlayer_style', 'kc'), st, prov))
    assert changed == {'interaction'}


def test_provenance_interlayer_sha_wins(afm):
    cfg, st, prov = afm
    prov2 = {**prov, 'interlayer': {'style': 'ilp', 'sha256': 'c' * 64, 'file': 'x.ILP'}}
    run = _split(cfg, st, prov2)
    assert run.blocks['interaction']['fields']['general.interlayer_file_sha256'] == 'c' * 64
    assert run.derived['interlayer.style'] == 'ilp'
    assert 'interlayer.sha256' not in run.derived


def test_seed_change_only_realization(afm):
    cfg, st, prov = afm
    assert _diff(_split(cfg, st, prov), _split(cfg, st, {**prov, 'seed_velocity': 7})) \
        == {'realization'}
    assert _diff(_split(cfg, st, prov), _split(_with(cfg, 'indent.seed', 5), st, prov)) \
        == {'realization'}


def test_absent_field_equals_default(afm):                    # C1e
    cfg, st, prov = afm
    default = defaults_for('afm')['settings.simulation.ramp_max_iterations']
    without = copy.deepcopy(st)
    del without['simulation']['ramp_max_iterations']
    at_default = _with(st, 'simulation.ramp_max_iterations', default)
    assert _hashes(_split(cfg, without, prov)) == _hashes(_split(cfg, at_default, prov))
    fields = _split(cfg, at_default, prov).blocks['integration']['fields']
    assert 'settings.simulation.ramp_max_iterations' not in fields


def test_a_missing_section_equals_its_defaults(afm):
    """An old config.json without [indent]/[pes] is the same request as one at defaults."""
    cfg, st, prov = afm
    old = {k: v for k, v in cfg.items() if k not in ('pes', 'indent', 'lattice_match')}
    assert _diff(_split(cfg, st, prov), _split(old, st, prov)) == set()


def test_int_and_float_coerce_alike(afm):
    cfg, st, prov = afm
    a = _split(_with(cfg, 'general.temp', 300), st, prov)
    b = _split(_with(cfg, 'general.temp', 300.0), st, prov)
    assert _hashes(a) == _hashes(b)
    c = _split(_with(cfg, 'general.temp', 301), st, prov)
    assert c.blocks['thermostat']['fields']['general.temp'] == 301.0
    assert isinstance(c.blocks['thermostat']['fields']['general.temp'], float)


def test_layer_list_resolved_to_run(afm):
    """content names sheets.<i>: `sheet` mirrors sheets[0] (see _sheets)."""
    cfg, st, prov = afm
    cfg = _with(cfg, 'sheets.0.layers', [1, 2])
    fields = _split(cfg, st, prov, n_layers=2).blocks['geometry']['fields']
    assert fields['sheets.0.layers'] == 2
    assert not [k for k in fields if k.startswith('sheet.')]
    with pytest.raises(ValueError, match='sheets.0.layers'):
        _split(cfg, st, prov)
    with pytest.raises(ValueError, match='sheets.0.layers'):
        _split(cfg, st, prov, n_layers=3)


def test_one_layer_count_needs_no_n_layers(afm):
    cfg, st, prov = afm
    assert _split(cfg, st, prov).blocks['geometry']['fields']['sheets.0.layers'] == 1


def test_stack_layers_are_per_sheet(hetero):
    cfg, st, prov = hetero
    fields = _split(cfg, st, prov, family='hetero', n_layers=4).blocks['geometry']['fields']
    assert fields['sheets.0.layers'] == 2 and fields['sheets.1.layers'] == 2
    # A stack builder reads layers[0] of each sheet (hetero_stack.py:1439).
    longer = _with(cfg, 'sheets.1.layers', [2, 3])
    fields = _split(longer, st, prov, family='hetero').blocks['geometry']['fields']
    assert fields['sheets.1.layers'] == 2


def test_list_left_under_non_swept_field_raises(afm):
    cfg, st, prov = afm
    with pytest.raises(ValueError, match='sheets.0.x'):
        _split(_with(cfg, 'sheets.0.x', [90.0, 100.0]), st, prov)
    one = _split(_with(cfg, 'sheets.0.x', [90.0]), st, prov)
    assert _hashes(one) == _hashes(_split(cfg, st, prov))


def test_sheet_only_config_equals_sheets(afm):
    cfg, st, prov = afm
    old = {**copy.deepcopy(cfg), 'sheets': []}
    assert _diff(_split(cfg, st, prov), _split(old, st, prov)) == set()


def test_unknown_family_raises(afm):
    cfg, st, prov = afm
    with pytest.raises(ValueError, match='family'):
        _split(cfg, st, prov, family='nonsense')


def test_hetero_has_two_material_blocks(hetero):
    cfg, st, prov = hetero
    blocks = _split(cfg, st, prov, family='hetero', n_layers=4).blocks
    mats = sorted(k for k in blocks if block_kind(k) == 'material')
    assert mats == ['material__sheet_0', 'material__sheet_1']
    assert blocks['material__sheet_0']['fields']['name'] == 'h-MoS2'
    assert blocks['material__sheet_1']['fields']['name'] == 'h-WS2'
    assert (block_hash(blocks['material__sheet_0'])
            != block_hash(blocks['material__sheet_1']))


def test_material_block_is_name_and_structure_hash(afm, tmp_path):
    cfg, st, prov = afm
    sheet = _split(cfg, st, prov).blocks['material__sheet_0']['fields']
    assert set(sheet) == {'name', 'cif_sha256'}
    assert len(sheet['cif_sha256']) == 64
    assert str(tmp_path) not in repr(sheet)


def test_potential_block_carries_file_hash_not_path(afm, tmp_path):
    cfg, st, prov = afm
    pot = _split(cfg, st, prov).blocks['potential__tip_0']['fields']
    assert pot == {'pot_type': 'sw',
                   'pot_file_sha256': prov['materials'][-2]['pot_file_sha256']}
    assert str(tmp_path) not in repr(pot)


def test_same_material_in_two_roles_hashes_alike(afm):
    """Tip and substrate are both Si from one file: one shared material node."""
    cfg, st, prov = afm
    blocks = _split(cfg, st, prov).blocks
    assert block_hash(blocks['material__tip_0']) == block_hash(blocks['material__substrate_0'])


def test_derived_is_flat_and_unhashed(afm):
    cfg, st, prov = afm
    run = _split(cfg, st, prov)
    assert run.derived['drive_k'] == 0.3 and run.derived['geometry.cell_a'] == 3.16
    assert run.derived['damp_ev'] == 6.25
    prov2 = {**prov, 'drive_k': 0.4, 'damp_ev': 1.0, 'contact_cutoff': 5.0}
    assert _diff(run, _split(cfg, st, prov2)) == set()


def test_moved_files_keep_their_hash(afm, tmp_path):
    """A tree read on another machine: the path is gone, the copy in provenance/ is not."""
    cfg, st, prov = afm
    original = _split(cfg, st, prov)
    run_dir = tmp_path / 'elsewhere'
    (run_dir / 'provenance' / 'cif').mkdir(parents=True)
    moved = copy.deepcopy(cfg)
    for key in ('tip', 'sub', 'sheet', 'flake'):
        moved[key]['cif_path'] = '/gone/dummy.cif'
    for item in moved['sheets'] + moved['flake_sheets']:
        item['cif_path'] = '/gone/dummy.cif'
    (run_dir / 'provenance' / 'cif' / 'dummy.cif').write_text(
        '# dummy cif', encoding='utf-8')
    assert _diff(original, _split(moved, st, prov, run_dir=run_dir)) == set()


def _afm_tree(tmp_path, cfg):
    """An AFM-shaped run: provenance/ at the run level, provenance.json in L1/."""
    run = tmp_path / 'afm' / 'h-MoS2' / 'K300'
    (run / 'provenance' / 'cif').mkdir(parents=True)
    (run / 'L1').mkdir()
    name = Path(cfg['sheet']['cif_path']).name
    copy_path = run / 'provenance' / 'cif' / name
    return run, copy_path


def test_afm_layout_finds_provenance_one_level_up(afm, tmp_path):
    cfg, st, prov = afm
    run, copy_path = _afm_tree(tmp_path, cfg)
    copy_path.write_text('# dummy cif', encoding='utf-8')
    moved = copy.deepcopy(cfg)
    for item in [moved[k] for k in ('tip', 'sub', 'sheet', 'flake')] + \
            moved['sheets'] + moved['flake_sheets']:
        item['cif_path'] = '/gone/dummy.cif'
    assert _diff(_split(cfg, st, prov),
                 _split(moved, st, prov, run_dir=run / 'L1')) == set()
    assert _diff(_split(cfg, st, prov),
                 _split(moved, st, prov, provenance_dir=run / 'provenance')) == set()


def test_provenance_copy_wins_over_config_path(afm, tmp_path):
    """The build-time copy is what the run read; the config path may have changed since."""
    from frictionsim2d.data.provenance import sha256_file  # pylint: disable=import-outside-toplevel
    cfg, st, prov = afm
    run, copy_path = _afm_tree(tmp_path, cfg)
    copy_path.write_text('# the cif the build used', encoding='utf-8')
    live = Path(cfg['sheet']['cif_path'])
    assert live.is_file() and live.read_bytes() != copy_path.read_bytes()
    sheet = _split(cfg, st, prov, run_dir=run / 'L1').blocks['material__sheet_0']
    assert sheet['fields']['cif_sha256'] == sha256_file(copy_path)
    assert _split(cfg, st, prov).blocks['material__sheet_0']['fields']['cif_sha256'] == \
        sha256_file(live)


def test_unresolved_file_is_recorded_and_logged(afm, caplog):
    cfg, st, prov = afm
    moved = copy.deepcopy(cfg)
    moved['sheets'][0]['cif_path'] = '/gone/a.cif'
    moved['tip']['cif_path'] = '/gone/b.cif'
    with caplog.at_level('WARNING', logger='frictionsim2d.aiida.blocks'):
        blocks = _split(moved, st, prov).blocks
    assert blocks['material__sheet_0']['fields']['cif_sha256'] == f'{UNRESOLVED}a.cif'
    assert blocks['material__tip_0']['fields']['cif_sha256'] == f'{UNRESOLVED}b.cif'
    assert '/gone/a.cif' in caplog.text and '/gone/b.cif' in caplog.text


def test_sheet_family_flake_without_material_is_filled_from_2d(tmp_path, mock_settings):
    """FlakeSlideSimulation._filled_flake_config fills [Flake] from [2D] field by field."""
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
    from frictionsim2d.builders.sheetonsheet.flake_slide import (  # pylint: disable=import-outside-toplevel
        FlakeSlideSimulation)
    pot, cif = _files(tmp_path)
    obj = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': {**_component(pot, cif, 'h-MoS2'), 'x': 100.0, 'y': 100.0,
               'layers': [1]},
        'Flake': {'edge_length': 10.0},
        'settings': mock_settings.model_dump(),
    })
    filled = FlakeSlideSimulation._filled_flake_config(  # pylint: disable=protected-access
        SimpleNamespace(config=obj))
    record = _provenance(obj.model_copy(update={'flake': filled}))
    assert [m['role'] for m in record['materials']] == ['sheet', 'flake']
    blocks = split_blocks(obj.model_dump(), mock_settings.model_dump(), record,
                          family='flake').blocks
    flake, sheet = blocks['material__flake_0']['fields'], blocks['material__sheet_0']['fields']
    assert flake['name'] == 'h-MoS2'
    assert flake['cif_sha256'] == sheet['cif_sha256']

    # Partial: a flake naming its own mat but no structure takes [2D]'s cif.
    partial = obj.model_copy(update={'flake': obj.flake.model_copy(update={'mat': 'h-WS2'})})
    filled = FlakeSlideSimulation._filled_flake_config(  # pylint: disable=protected-access
        SimpleNamespace(config=partial))
    record = _provenance(partial.model_copy(update={'flake': filled}))
    blocks = split_blocks(partial.model_dump(), mock_settings.model_dump(), record,
                          family='flake').blocks
    flake, sheet = blocks['material__flake_0']['fields'], blocks['material__sheet_0']['fields']
    assert flake['name'] == 'h-WS2'
    assert flake['cif_sha256'] == sheet['cif_sha256']


def test_provenance_sourced_fields_are_not_read_from_config(afm, tmp_path):
    """pot_type/pot_path enter the potential block from provenance, by design."""
    cfg, st, prov = afm
    assert set(PROVENANCE_SOURCED) == {
        f'{s}.{f}' for s in ('sheet', 'sheets', 'flake_sheets', 'tip', 'sub', 'flake')
        for f in ('pot_type', 'pot_path')}
    for path in PROVENANCE_SOURCED:
        cfg2 = copy.deepcopy(cfg)
        _set(cfg2, path, tmp_path)
        assert _diff(_split(cfg, st, prov), _split(cfg2, st, prov)) == set()


# --- default elision across schema versions ---------------------------------------

@pytest.fixture
def bumped(monkeypatch):
    """Schema 1.1.0, whose bump changed general.temp's default from 250 K to 300 K."""
    monkeypatch.setattr(block_schema, 'BLOCK_SCHEMA_VERSION', '1.1.0')
    monkeypatch.setattr(block_schema, 'LEGACY_DEFAULTS',
                        {'1.0.0': {'general.temp': 250.0}})


def _old(content, version='1.0.0', **fields):
    return {**content, 'block_schema': version, 'fields': {**content['fields'], **fields}}


def test_legacy_value_is_not_an_elision_value(afm, bumped):
    """A new run at the new default and one asking for the old value stay apart."""
    cfg, st, prov = afm
    absent = _split({**cfg, 'general': {k: v for k, v in cfg['general'].items()
                                         if k != 'temp'}}, st, prov)
    old_value = _split(_with(cfg, 'general.temp', 250.0), st, prov)
    assert 'general.temp' not in absent.blocks['thermostat']['fields']
    assert old_value.blocks['thermostat']['fields']['general.temp'] == 250.0
    assert _diff(absent, old_value) == {'thermostat'}
    assert absent.blocks['thermostat']['block_schema'] == '1.1.0'


def test_old_block_refills_the_old_default(afm, bumped):
    """A 1.0.0 block that elided temp meant 250 K: it joins the explicit-250 run,
    not the run left at the new 300 K default."""
    cfg, st, prov = afm
    current = _split(cfg, st, prov).blocks['thermostat']        # temp 300 -> elided
    explicit_250 = _split(_with(cfg, 'general.temp', 250.0), st, prov).blocks['thermostat']
    old_elided = _old(current)                                  # 1.0.0, temp absent
    upgraded = upgrade_block(old_elided, family='afm')
    assert upgraded['block_schema'] == '1.1.0'
    assert upgraded['fields']['general.temp'] == 250.0
    assert block_hash(upgraded) == block_hash(explicit_250)
    assert block_hash(upgraded) != block_hash(current)
    # An old block that asked for 300 K explicitly is the new default: elided.
    old_300 = _old(current, **{'general.temp': 300.0})
    assert block_hash(upgrade_block(old_300, family='afm')) == block_hash(current)


def test_upgrade_leaves_current_and_unrelated_blocks_alone(afm, bumped):
    cfg, st, prov = afm
    blocks = _split(cfg, st, prov).blocks
    for name in ('thermostat', 'geometry', 'material__sheet_0'):
        assert upgrade_block(blocks[name], family='afm') == blocks[name]
    # temp belongs to thermostat; an old geometry block gains nothing.
    old_geometry = _old(blocks['geometry'])
    assert upgrade_block(old_geometry, family='afm') == blocks['geometry']
    with pytest.raises(ValueError, match='newer'):
        upgrade_block(_old(blocks['thermostat'], version='2.0.0'), family='afm')


def test_upgrade_takes_the_earliest_recorded_change(afm, monkeypatch):
    cfg, st, prov = afm
    monkeypatch.setattr(block_schema, 'BLOCK_SCHEMA_VERSION', '1.2.0')
    monkeypatch.setattr(block_schema, 'LEGACY_DEFAULTS', {
        '1.0.0': {'general.temp': 250.0}, '1.1.0': {'general.temp': 280.0},
        '1.0.5': {'sheets.stack_type': 'AB'}})
    blocks = _split(cfg, st, prov).blocks
    assert upgrade_block(_old(blocks['thermostat']), family='afm')['fields'][
        'general.temp'] == 250.0
    assert upgrade_block(_old(blocks['thermostat'], version='1.1.0'), family='afm')[
        'fields']['general.temp'] == 280.0
    # A list-section field is refilled for each index the block names.
    geometry = upgrade_block(_old(blocks['geometry']), family='afm')['fields']
    assert geometry['sheets.0.stack_type'] == 'AB'
    assert geometry['flake_sheets.0.x'] == blocks['geometry']['fields']['flake_sheets.0.x']
    assert 'sheets.0.stack_type' not in upgrade_block(
        _old(blocks['geometry'], version='1.1.0'), family='afm')['fields']


# --- the hash -----------------------------------------------------------------

def _block(kind, **fields):
    return {'kind': kind, 'block_schema': '1.0.0', 'fields': fields}


def test_block_hash_ignores_key_order_and_float_repr():
    a = _block('integration', **{'settings.simulation.timestep': 0.001,
                                 'x': {'a': 1.0, 'b': [2.5, 3]}, 'flag': True})
    b = {'fields': {'flag': True, 'x': {'b': [np.float64(2.5), 3], 'a': 1.0},
                    'settings.simulation.timestep': 1e-3},
         'block_schema': '1.0.0', 'kind': 'integration'}
    assert block_hash(a) == block_hash(b)
    assert block_hash(a) != block_hash({**a, 'kind': 'thermostat'})
    assert block_hash(a) != block_hash({**a, 'block_schema': '1.1.0'})
    assert block_hash(a) != block_hash(
        _block('integration', **{**a['fields'], 'settings.simulation.timestep': 0.002}))


def test_split_coerces_so_one_and_one_point_oh_hash_alike(afm):
    """`1` vs `1.0` is decided by the split (field type), not the hash."""
    cfg, st, prov = afm
    a = _split(_with(cfg, 'tip.r', 20), st, prov).blocks['geometry']
    b = _split(_with(cfg, 'tip.r', 20.0), st, prov).blocks['geometry']
    assert block_hash(a) == block_hash(b)


def test_block_hash_keeps_ints_and_bools_distinct_from_floats():
    assert block_hash(_block('drive', v=1)) != block_hash(_block('drive', v=1.0))
    assert block_hash(_block('drive', v=True)) != block_hash(_block('drive', v=1))


def test_block_hash_reads_the_kind_from_the_content():
    with pytest.raises(ValueError):
        block_hash(_block('nonsense', a=1))
    with pytest.raises(ValueError):
        block_hash({'fields': {}})
    assert block_kind('material__sheet_0') == 'material'


# --- coverage (Review Focus 4/5) ----------------------------------------------

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


def test_every_provenance_field_is_assigned():                # I3
    for path in leaf_field_paths(ProvenanceRecord):
        assert path in PROVENANCE_MAP or path in EXCLUDED, path


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
    from frictionsim2d.core.config import ThermostatSettings  # pylint: disable=import-outside-toplevel

    class Grown(AFMSimulationConfig):
        brand_new: float = 1.0

    class GrownThermostat(ThermostatSettings):
        brand_new: float = 1.0

    class GrownSettings(GlobalSettings):
        thermostat: GrownThermostat

    class GrownRecord(ProvenanceRecord):
        brand_new: float = 1.0

    unassigned = [p for p in leaf_field_paths(Grown)
                  if not assigned(p) and p not in EXCLUDED]
    assert unassigned == ['brand_new']
    unassigned = [p for p in leaf_field_paths(GrownSettings, 'settings.')
                  if not assigned(p) and p not in EXCLUDED]
    assert unassigned == ['settings.thermostat.brand_new']
    unassigned = [p for p in leaf_field_paths(GrownRecord)
                  if p not in PROVENANCE_MAP and p not in EXCLUDED]
    assert unassigned == ['brand_new']


def test_map_and_exclusions_name_only_real_fields():
    real = _all_paths()
    provenance = set(leaf_field_paths(ProvenanceRecord))
    assert set(FIELD_MAP) <= real, sorted(set(FIELD_MAP) - real)
    assert set(SWEPT) <= real and set(PROVENANCE_SOURCED) <= real
    assert not set(FIELD_MAP) & set(PROVENANCE_SOURCED)
    assert not set(FIELD_MAP) & set(SWEPT)
    assert not set(PROVENANCE_SOURCED) & set(EXCLUDED)
    # run_uuid arrives with Task 5b; it is excluded ahead of it.
    assert set(EXCLUDED) <= real | provenance | {'run_uuid'}, \
        sorted(set(EXCLUDED) - real - provenance)
    assert not (set(FIELD_MAP) | set(SWEPT)) & set(EXCLUDED)
    assert set(FIELD_MAP.values()) <= set(BLOCK_KINDS)
    assert set(PROVENANCE_MAP) <= provenance | {'interlayer.sha256'}
    assert not set(PROVENANCE_MAP) & set(EXCLUDED)
    assert set(PROVENANCE_MAP.values()) <= set(BLOCK_KINDS) | {'derived', 'point',
                                                               'run_info'}
    assert set(PROVENANCE_EXCLUDED) <= set(EXCLUDED)


def test_exclusions_are_named_not_physics():
    for section in ('hpc', 'aiida', 'integrity'):
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
    if name == 'layers':
        return [value[0] + 1]                        # one count: no n_layers needed
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
        if target.get(key) is None:      # an optional nested model left unset (hpc.gpu)
            target[key] = {}
        target = target[key]
    target[rest[-1]] = _perturb(target.get(rest[-1]), tmp_path, rest[-1])


@pytest.mark.parametrize('path', sorted(FIELD_MAP))
def test_each_mapped_field_changes_only_its_block(afm, tmp_path, path):
    cfg, st, prov = afm
    if path.startswith('sheet.'):
        # `sheet` mirrors `sheets[0]`; the split reads it when `sheets` is empty.
        cfg = {**copy.deepcopy(cfg), 'sheets': []}
    kind = FIELD_MAP[path]
    cfg2, st2 = copy.deepcopy(cfg), copy.deepcopy(st)
    if path.startswith('settings.'):
        _set(st2, path[len('settings.'):], tmp_path)
    else:
        _set(cfg2, path, tmp_path)
    changed = _diff(_split(cfg, st, prov), _split(cfg2, st2, prov))
    assert changed, path
    assert {block_kind(n) for n in changed} == {kind}, (path, changed)


@pytest.mark.parametrize('path', sorted(p for p in EXCLUDED if p.startswith('settings.')))
def test_excluded_settings_change_nothing(afm, tmp_path, path):
    cfg, st, prov = afm
    st2 = copy.deepcopy(st)
    _set(st2, path[len('settings.'):], tmp_path)
    assert _diff(_split(cfg, st, prov), _split(cfg, st2, prov)) == set()


def _perturb_provenance(prov, field):
    prov2 = copy.deepcopy(prov)
    if field == 'interlayer.sha256':
        prov2['interlayer'] = {'sha256': 'e' * 64}
        return prov2
    if field == 'interlayer':
        prov2['interlayer'] = {'style': 'kc', 'g_meV': {'h-MoS2': {'value': 1.0}}}
        return prov2
    head, _, rest = field.partition('.')
    if head == 'materials':
        for entry in prov2['materials']:
            entry[rest] = 'perturbed'
        return prov2
    target, name = (prov2['geometry'], rest) if head == 'geometry' else (prov2, head)
    value = target.get(name)
    if isinstance(value, bool):
        target[name] = not value
    elif isinstance(value, (int, float)):
        target[name] = value + 1
    elif isinstance(value, str):
        target[name] = value + '_perturbed'
    elif isinstance(value, dict):
        target[name] = {**value, 'perturbed': 1}
    else:
        target[name] = 7
    return prov2


@pytest.mark.parametrize('field', sorted(PROVENANCE_MAP))
def test_each_provenance_field_lands_where_mapped(afm, field):
    cfg, st, prov = afm
    destination = PROVENANCE_MAP[field]
    if field == 'materials.name':
        # The entry's name is the fallback for a component config without one.
        cfg = copy.deepcopy(cfg)
        for item in [cfg[k] for k in ('tip', 'sub', 'sheet', 'flake')] + \
                cfg['sheets'] + cfg['flake_sheets']:
            item['mat'] = None
    a, b = _split(cfg, st, prov), _split(cfg, st, _perturb_provenance(prov, field))
    changed = _diff(a, b)
    if destination in BLOCK_KINDS:
        assert changed and {block_kind(n) for n in changed} == {destination}, changed
        return
    assert changed == set(), (field, changed)
    if destination == 'derived':
        assert a.derived != b.derived, field
    else:                                            # point / run_info: elsewhere
        assert a.derived == b.derived, field


def test_settings_default_to_the_config_copy(afm):
    cfg, st, prov = afm
    assert _diff(_split(cfg, st, prov), _split(cfg, {}, prov)) == set()


def test_split_is_json_plain(afm):
    import json  # pylint: disable=import-outside-toplevel
    cfg, st, prov = afm
    run = _split(cfg, st, prov)
    json.dumps(run.blocks)                           # Dict-node storable
    json.dumps(run.derived)


@pytest.mark.parametrize('module', ['blocks.py', 'block_schema.py'])
def test_module_does_not_import_aiida(module):
    import ast  # pylint: disable=import-outside-toplevel
    path = Path(__file__).resolve().parents[2] / 'frictionsim2d' / 'aiida' / module
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names]
    names += [node.module or '' for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.level == 0]
    assert not [n for n in names if n.split('.')[0] == 'aiida'], names
