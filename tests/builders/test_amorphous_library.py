"""The amorphous block library: conditions-keyed reuse, the quench root, import.

A block is reused only when its material, potential bytes and the six quench
conditions match and it is large enough (spec 2026-11-24-amorphous-quench-design
§4). No path here may start a melt-quench, and nothing may write into the
packaged materials directory.
"""
from __future__ import annotations

import json
import subprocess
from importlib import resources
from pathlib import Path
from typing import Tuple

import pytest

from frictionsim2d.builders.construction import amorphous
from frictionsim2d.builders.construction.amorphous import (
    AmorphousMissing,
    QuenchKey,
    QuenchMember,
    amorphous_library_dir,
    find_block,
    import_block,
    import_quench_root,
    nearest_blocks,
    quench_members_for,
    write_quench_root,
)
from frictionsim2d.core.config import GlobalSettings, QuenchSettings

REPO = Path(__file__).resolve().parents[2]
SI_CIF = REPO / 'examples' / 'materials' / 'Si.cif'
SI_POT = REPO / 'examples' / 'potentials' / 'sw' / 'Si.sw'
PACKAGED = Path(str(resources.files('frictionsim2d.data'))) / 'materials'
SI_SHA = 'c6d3a7d26db28ee8e3feaf2f8059607de9a4accb45d9719fcabe59488ecf1cad'


@pytest.fixture(autouse=True)
def _data_home(tmp_path, monkeypatch):
    """Every test gets its own empty user library and materials pack."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg_data'))


def _key(material='Si', **overrides) -> QuenchKey:
    return QuenchKey.from_settings(material, SI_POT, QuenchSettings(**overrides))


def _write_block(directory: Path, name: str, box: Tuple[float, float, float],
                 meta: dict = None, natoms: int = 8) -> Path:
    """A tiny LAMMPS data file with ``box``, plus its sidecar when ``meta`` is given."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    lines = ['LAMMPS data file via write_data', '', f'{natoms} atoms', '1 atom types', '',
             f'0.0 {box[0]} xlo xhi', f'0.0 {box[1]} ylo yhi', f'0.0 {box[2]} zlo zhi',
             '', 'Masses', '', '1 28.085', '', 'Atoms # atomic', '']
    lines += [f'{i} 1 0.0 0.0 0.0' for i in range(1, natoms + 1)]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if meta is not None:
        Path(str(path) + '.json').write_text(json.dumps(meta), encoding='utf-8')
    return path


def _user_block(key: QuenchKey, box, **extra) -> Path:
    meta = {**key.as_dict(), 'key8': key.key8, 'box': list(box), **extra}
    return _write_block(amorphous_library_dir(), f'amor_{key.material}__{key.key8}.lmp',
                        box, meta)


def _packaged_mtimes():
    return {p.name: p.stat().st_mtime_ns for p in PACKAGED.iterdir()}


# --------------------------------------------------------------- key + lookup

def test_key_canonicalises_floats_to_six_digits():
    assert _key(quench_rate=1e12).key8 == _key(quench_rate=1000000000000.0).key8
    assert _key(quench_rate=1e12).key8 != _key(quench_rate=2e12).key8


def test_key_hashes_the_potential_bytes_not_its_name(tmp_path):
    renamed = tmp_path / 'copy_of_si.sw'
    renamed.write_bytes(SI_POT.read_bytes())
    assert QuenchKey.from_settings('Si', renamed, QuenchSettings()).potential_sha256 == SI_SHA


def test_conditions_mismatch_not_reused():
    _user_block(_key(melt_temp=2500), (300.0, 300.0, 60.0))
    assert find_block(_key(melt_temp=3000), (10.0, 10.0, 10.0)) is None


def test_large_enough_and_matching_reused():
    key = _key(melt_temp=3000)
    block = _user_block(key, (300.0, 300.0, 60.0))
    assert find_block(key, (250.0, 250.0, 50.0)) == block


def test_too_small_block_not_reused():
    key = _key(melt_temp=3000)
    _user_block(key, (100.0, 100.0, 60.0))
    assert find_block(key, (150.0, 50.0, 10.0)) is None


def test_smallest_fit_wins():
    key = _key(melt_temp=3000)
    big = _user_block(key, (400.0, 400.0, 60.0))
    small = _write_block(amorphous.materials_pack_dir(), f'amor_Si__{key.key8}.lmp',
                         (200.0, 200.0, 60.0),
                         {**key.as_dict(), 'key8': key.key8, 'box': [200.0, 200.0, 60.0]})
    assert find_block(key, (100.0, 100.0, 10.0)) == small
    assert find_block(key, (300.0, 300.0, 10.0)) == big


def test_legacy_unkeyed_block_still_found():
    """The shipped amor_Si carries the conditions recovered from git (spec §11.1);
    default settings key to it."""
    found = find_block(QuenchKey.from_settings('Si', SI_POT, QuenchSettings()),
                       (100.0, 100.0, 12.0))
    assert found is not None and found.parent == PACKAGED
    meta = json.loads(Path(str(found) + '.json').read_text(encoding='utf-8'))
    assert meta['source'] == 'legacy' and meta['melt_seed'] == 1234579


def test_sidecar_without_all_key_fields_is_never_matched():
    key = _key(melt_temp=3000)
    meta = {k: v for k, v in key.as_dict().items() if k != 'melt_steps'}
    _write_block(amorphous_library_dir(), 'amor_Si.lmp', (300.0, 300.0, 60.0), meta)
    assert find_block(key, (1.0, 1.0, 1.0)) is None
    near = [m for _p, m in nearest_blocks('Si', key=key) if m['root'] == 'user']
    assert near and 'melt_steps' in near[0]['unkeyed_reason']


def test_block_without_sidecar_is_invisible_to_find_block():
    key = _key(melt_temp=3000)
    _write_block(amorphous_library_dir(), f'amor_Si__{key.key8}.lmp', (300.0, 300.0, 60.0))
    assert find_block(key, (1.0, 1.0, 1.0)) is None


def test_nearest_blocks_name_what_differs():
    _user_block(_key(melt_temp=2500), (300.0, 300.0, 60.0))
    wanted = _key(melt_temp=3000)
    near = nearest_blocks('Si', key=wanted)
    assert any('melt_temp 2500 (wanted 3000)' in d for _p, m in near for d in m['differs'])


def test_nearest_blocks_rank_an_oversize_block_ahead_of_an_undersized_one():
    """Size shortfall counts only what is MISSING: a block larger than needed has
    none, so it must not be ranked behind a smaller one for being big."""
    key = _key(melt_temp=3000)
    meta = {**key.as_dict(), 'key8': key.key8}
    name = f'amor_Si__{key.key8}.lmp'
    small = _write_block(amorphous.materials_pack_dir(), name, (200.0, 200.0, 60.0),
                         {**meta, 'box': [200.0, 200.0, 60.0]})
    big = _write_block(amorphous_library_dir(), name, (600.0, 600.0, 60.0),
                       {**meta, 'box': [600.0, 600.0, 60.0]})
    ranked = [p for p, _m in nearest_blocks('Si', key=key, min_dims=(300.0, 300.0, 60.0))]
    assert ranked[:2] == [big, small]


def test_quench_run_length_is_at_least_one_step():
    """A quench rate so fast the run rounds to zero steps still runs one."""
    from frictionsim2d.builders.construction.amorphous import quench_nsteps
    assert quench_nsteps(_key(quench_rate=1e30)) == 1


# --------------------------------------------------------------- the miss

def _afm_config(tmp_path) -> Path:
    """examples/afm_config.ini with absolute paths (amorphous Si substrate)."""
    text = (REPO / 'examples' / 'afm_config.ini').read_text(encoding='utf-8')
    text = text.replace('examples/', f'{REPO}/examples/')
    path = tmp_path / 'afm.ini'
    path.write_text(text, encoding='utf-8')
    return path


def test_run_without_block_raises_with_command(tmp_path):
    settings = GlobalSettings(quench=QuenchSettings(melt_temp=3000))
    members = quench_members_for(_afm_config(tmp_path), settings)
    assert [m.material for m in members] == ['Si']
    exc = AmorphousMissing(members, config_file='examples/afm_config.ini',
                           settings_file='my.yaml')
    assert 'run quench --for examples/afm_config.ini' in exc.command
    assert '--settings-file my.yaml' in exc.command
    text = str(exc)
    assert 'No amorphous Si block matches this run.' in text
    assert 'melt 3000 K' in text and exc.command in text


def test_default_settings_need_no_quench(tmp_path):
    assert quench_members_for(_afm_config(tmp_path), GlobalSettings()) == []


def test_make_amorphous_miss_raises_not_generates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: calls.append(a))
    settings = GlobalSettings(quench=QuenchSettings(melt_temp=3000))
    before = _packaged_mtimes()
    with pytest.raises(AmorphousMissing) as info:
        amorphous.make_amorphous('Si', SI_CIF, 10.0, 10.0, 10.0, SI_POT, 'sw', settings)
    assert calls == []
    assert info.value.keys[0].melt_temp == 3000
    assert _packaged_mtimes() == before


# --------------------------------------------------------------- quench root

def _members(slab=(12.0, 12.0, 12.0), **conditions):
    settings = GlobalSettings(quench=QuenchSettings(**conditions))
    return [QuenchMember.make('Si', SI_POT, 'sw', SI_CIF, slab, settings.quench)], settings


def test_quench_root_layout(tmp_path):
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='quench_x')
    key8 = members[0].key.key8
    mdir = root / 'quench' / 'Si' / key8
    deck = (mdir / 'lammps' / 'quench.in').read_text(encoding='utf-8')
    assert f'write_data      quench/Si/{key8}/data/amor_Si__{key8}.lmp' in deck
    assert f'read_data       quench/Si/{key8}/build/Si_crystal_block.lmp' in deck
    assert f'quench/Si/{key8}/provenance/potentials/Si.sw' in deck
    prov = json.loads((mdir / 'provenance.json').read_text(encoding='utf-8'))
    assert prov['kind'] == 'quench' and prov['key8'] == key8
    assert prov['quench_nsteps'] == round((3000 - 300) / (1e12 * 0.002 * 1e-12))
    for sub in ('data', 'results', 'visuals', 'build'):
        assert (mdir / sub).is_dir()
    assert (mdir / 'provenance' / 'settings.yaml').is_file()


def test_quench_step_count_is_dT_over_rate_dt(tmp_path):
    members, settings = _members(quench_rate=2e12)
    root = write_quench_root(members, tmp_path, settings, root_name='q')
    prov = json.loads(next(root.rglob('provenance.json')).read_text(encoding='utf-8'))
    assert prov['quench_nsteps'] == 550_000


def test_nonpositive_quench_rate_is_rejected():
    with pytest.raises(ValueError, match='quench_rate'):
        _members(quench_rate=0.0)


def _finish(member_dir: Path, box=(12.5, 12.5, 12.5), log_tail='Total wall time: 0:00:01'):
    """Stand in for the HPC job: the block in data/ and a finished log."""
    key8 = member_dir.name
    mat = member_dir.parent.name
    _write_block(member_dir / 'data', f'amor_{mat}__{key8}.lmp', box)
    (member_dir / 'lammps' / 'run_quench.log').write_text(
        f'LAMMPS (29 Aug 2024)\nLoop time of 1.0\n{log_tail}\n', encoding='utf-8')


def test_hpc_tree_then_hpc_generate_then_import(tmp_path):
    from frictionsim2d.hpc import HPCConfig
    from frictionsim2d.run.hpc_dispatch import (
        collect_hpc_simulation_paths, generate_scripts_for_tree)

    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='quench_hpc')
    settings.hpc.modules = ['lammps']
    hpc_config = HPCConfig.from_settings(settings.hpc)
    paths = collect_hpc_simulation_paths(root, lammps_scripts=hpc_config.lammps_scripts)
    assert paths == [f'quench/Si/{members[0].key.key8}']
    generate_scripts_for_tree(root, hpc_config, root / 'hpc', paths, '$PBS_O_WORKDIR')
    manifest = (root / 'hpc' / 'manifest_1.txt').read_text(encoding='utf-8')
    assert manifest.strip().endswith('lammps/quench.in')
    assert (root / 'hpc' / 'run.pbs').is_file()

    mdir = root / paths[0]
    _finish(mdir)
    before = _packaged_mtimes()
    report = import_quench_root(root)
    assert report.incomplete == [] and len(report.imported) == 1
    found = find_block(members[0].key, (12.0, 12.0, 12.0))
    assert found is not None and found.parent == amorphous_library_dir()
    meta = json.loads(Path(str(found) + '.json').read_text(encoding='utf-8'))
    assert meta['schema'] == 'amorphous-1' and meta['source'] == 'hpc'
    assert meta['box'] == [12.5, 12.5, 12.5] and meta['natoms'] == 8
    assert _packaged_mtimes() == before


def test_import_rejects_incomplete_tree(tmp_path):
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='quench_bad')
    mdir = next((root / 'quench' / 'Si').iterdir())
    with pytest.raises(ValueError, match='data/amor_Si__'):
        import_block(mdir)
    report = import_quench_root(root)
    assert report.imported == [] and len(report.incomplete) == 1
    assert not amorphous_library_dir().exists() or not any(amorphous_library_dir().iterdir())


def test_import_rejects_a_log_without_wall_time(tmp_path):
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='quench_cut')
    mdir = next((root / 'quench' / 'Si').iterdir())
    _finish(mdir, log_tail='Step Temp')
    with pytest.raises(ValueError, match='Total wall time'):
        import_block(mdir)


def test_import_of_a_not_larger_block_is_refused_unless_replace(tmp_path):
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='q1')
    mdir = next((root / 'quench' / 'Si').iterdir())
    _finish(mdir, box=(12.5, 12.5, 12.5))
    import_block(mdir)
    _finish(mdir, box=(12.4, 13.0, 13.0))
    with pytest.raises(FileExistsError):
        import_block(mdir)
    import_block(mdir, replace=True)
    meta = json.loads(Path(str(find_block(members[0].key, (1, 1, 1))) + '.json').read_text())
    assert meta['box'] == [12.4, 13.0, 13.0]


def test_packaged_dir_never_written(tmp_path):
    before = _packaged_mtimes()
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='q')
    mdir = next((root / 'quench' / 'Si').iterdir())
    _finish(mdir)
    import_quench_root(root)
    assert _packaged_mtimes() == before


def test_aiida_import_refuses_a_quench_root(tmp_path):
    members, settings = _members(melt_temp=3000)
    root = write_quench_root(members, tmp_path, settings, root_name='q')
    assert amorphous.is_quench_root(root)
    assert not amorphous.is_quench_root(tmp_path)


# --------------------------------------------------------------- fix round 1

#: What the publication set's trees record (Publication-data/settings.yaml):
#: built on the shipped amor_Si by the old size-only lookup all the same.
LEGACY_RECORDED = {'timestep': 0.001, 'melt_steps': 50000, 'equilibrate_steps': 20000}


def test_legacy_rebuild_lookup_finds_the_shipped_block_for_recorded_conditions():
    settings = GlobalSettings(quench=QuenchSettings(**LEGACY_RECORDED))
    with pytest.raises(AmorphousMissing):
        amorphous.make_amorphous('Si', SI_CIF, 100.0, 100.0, 12.0, SI_POT, 'sw', settings)
    with amorphous.legacy_rebuild_lookup():
        found = amorphous.make_amorphous('Si', SI_CIF, 100.0, 100.0, 12.0, SI_POT, 'sw',
                                         settings)
    assert found.parent == PACKAGED and found.name.startswith('amor_Si__')
    # The mode ends with the block: a live lookup is strict again.
    with pytest.raises(AmorphousMissing):
        amorphous.make_amorphous('Si', SI_CIF, 100.0, 100.0, 12.0, SI_POT, 'sw', settings)


def test_legacy_rebuild_lookup_still_needs_the_same_potential_and_size(tmp_path):
    other = tmp_path / 'Si_edited.sw'
    other.write_text(SI_POT.read_text(encoding='utf-8') + '\n# edited\n', encoding='utf-8')
    settings = GlobalSettings(quench=QuenchSettings(**LEGACY_RECORDED))
    with amorphous.legacy_rebuild_lookup():
        with pytest.raises(AmorphousMissing):
            amorphous.make_amorphous('Si', SI_CIF, 100.0, 100.0, 12.0, other, 'sw', settings)
        with pytest.raises(AmorphousMissing):
            amorphous.make_amorphous('Si', SI_CIF, 300.0, 100.0, 12.0, SI_POT, 'sw',
                                     settings)


def _legacy_settings_file(tmp_path) -> Path:
    import yaml
    path = tmp_path / 'legacy_settings.yaml'
    path.write_text(yaml.safe_dump({'quench': {**LEGACY_RECORDED, 'run_local': True}}),
                    encoding='utf-8')
    return path


def test_legacy_tree_rebuilds_onto_the_shipped_block(tmp_path, lammps_replay, monkeypatch):
    """A tree recording dt 0.001 / 50000 / 20000 rebuilds with the very
    substrate the old size-only lookup gave it; a live run with the same
    settings refuses."""
    from frictionsim2d.run.artifacts import generate_package
    from tests.helpers.deck_snapshot import SNAPSHOT_DIR, _deterministic, normalise_deck

    monkeypatch.chdir(REPO)
    settings_file = _legacy_settings_file(tmp_path)
    config = REPO / 'examples' / 'afm_config.ini'
    with pytest.raises(AmorphousMissing):
        generate_package(config, 'afm', output_root=tmp_path / 'live',
                         settings_file=settings_file, simulation_root_name='sim')
    assert not (tmp_path / 'live' / 'sim').exists()

    # As the snapshot renders it (fixed seeds/uuids), so the replayed LAMMPS
    # calls are the recorded ones.
    with amorphous.legacy_rebuild_lookup(), _deterministic():
        # Straight under tmp_path: the replay key covers the normalised paths.
        result = generate_package(config, 'afm', output_root=tmp_path,
                                  settings_file=settings_file, simulation_root_name='sim')
    assert not result.failures and result.created
    built = next((tmp_path / 'sim').rglob('build/sub.lmp'))
    expected = next((SNAPSHOT_DIR / 'afm_config').rglob('build/sub.lmp'))
    text = built.read_text(encoding='utf-8').replace(str(tmp_path), '<BUILD>')
    assert normalise_deck(text) == expected.read_text(encoding='utf-8')


def test_preflight_sizes_the_periodic_substrate_as_built():
    """A 214.9 A request tiles to 67 whole h-MoS2 cells (213.3 A), which the
    214.8 A shipped block covers; the request alone would have refused."""
    run = {
        '2D': {'mat': 'MoS2', 'pot_type': 'sw', 'cif_path': 'examples/materials/h-MoS2.cif',
               'pot_path': 'examples/potentials/sw/MoS2_wen.sw', 'x': 214.9, 'y': 100.0,
               'layers': [1]},
        'sub': {'mat': 'Si', 'amorph': 'a', 'thickness': 12.0, 'pot_type': 'sw',
                'pot_path': str(SI_POT), 'cif_path': str(SI_CIF)},
        'general': {'temp': 300.0},
    }
    for name in ('cif_path', 'pot_path'):
        run['2D'][name] = str(REPO / run['2D'][name])
    needs = amorphous.amorphous_needs([run], GlobalSettings())
    assert len(needs) == 1
    assert 213.0 < needs[0].min_dims[0] < 214.79
    assert amorphous.missing_members(needs) == []


def test_local_run_continues_past_a_failed_member(tmp_path, monkeypatch):
    settings = GlobalSettings()
    members = [QuenchMember.make('Si', SI_POT, 'sw', SI_CIF, (12.0, 12.0, 12.0),
                                 settings.quench, melt_temp=t) for t in (2800, 3000)]
    root = write_quench_root(members, tmp_path, settings, root_name='q')
    ran = []

    def flaky(deck, log, cwd):
        ran.append(deck)
        if len(ran) == 1:
            raise subprocess.CalledProcessError(1, ['lmp'])

    monkeypatch.setattr(amorphous, 'run_lmp_serial', flaky)
    done, failed = amorphous.run_quench_root_local(root)
    assert len(ran) == 2 and len(done) == 1 and len(failed) == 1


def test_quench_import_report_lines_send_failures_to_stderr(tmp_path):
    report = amorphous.QuenchImportReport()
    assert report.ok and report.lines() == []
    report.imported.append(tmp_path / 'a.lmp')
    report.warnings.append('odd box')
    report.incomplete.append((tmp_path / 'm', 'no log'))
    assert not report.ok
    assert report.lines() == [(f"✅ imported {tmp_path / 'a.lmp'}", False),
                              ("⚠️  odd box", False),
                              (f"❌ incomplete {tmp_path / 'm'}: no log", True)]


def test_block_line_shows_box_key_and_the_set_conditions():
    row = {'file': 'amor_Si.lmp', 'root': 'user', 'key8': None, 'unkeyed': 'no sidecar',
           'box': [10.0, 20.04, 12.0], 'natoms': 500,
           **{name: None for name in amorphous.CONDITION_FIELDS}}
    row[amorphous.CONDITION_FIELDS[0]] = 4000
    assert amorphous.block_line(row) == (
        f"amor_Si.lmp  user  unkeyed (no sidecar)  10.0x20.0x12.0 A  natoms=500  "
        f"{amorphous.CONDITION_FIELDS[0]}=4000")


def test_block_rows_take_the_conditions_from_the_key_else_the_sidecar(monkeypatch, tmp_path):
    from types import SimpleNamespace
    keyed = SimpleNamespace(key8='abcd1234', **{n: 1 for n in amorphous.CONDITION_FIELDS})
    blocks = [SimpleNamespace(path=tmp_path / 'amor_Si.lmp', root='user', key=keyed,
                              unkeyed_reason=None, box=(1.0, 2.0, 3.0), meta={'natoms': 9}),
              SimpleNamespace(path=tmp_path / 'amor_C.lmp', root='pack', key=None,
                              unkeyed_reason='no sidecar', box=None,
                              meta={amorphous.CONDITION_FIELDS[0]: 7})]
    seen = []
    monkeypatch.setattr(amorphous, 'scan_blocks', lambda m: seen.append(m) or blocks)
    rows = amorphous.block_rows('Si')
    assert seen == ['Si']
    assert rows[0]['key8'] == 'abcd1234' and rows[0]['box'] == [1.0, 2.0, 3.0]
    assert rows[0]['natoms'] == 9 and rows[0][amorphous.CONDITION_FIELDS[-1]] == 1
    assert rows[1]['key8'] is None and rows[1]['unkeyed'] == 'no sidecar'
    assert rows[1][amorphous.CONDITION_FIELDS[0]] == 7 and rows[1]['box'] is None
