"""export-results is the inverse of ingest (spec §4.2)."""
import json
import re
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.export import export_results, rebuild_set  # noqa: E402
from src.aiida.ingest import ingest_root  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

#: An AFM run: `rel_path` ends in `/K300/L2`, so -- exactly like a real
#: AFMSimulation build -- `make_run` writes `provenance/` one level ABOVE the
#: `L2` directory (shared by every layer), not inside it. `provenance.json`
#: itself stays inside `L2`, alongside `lammps/` and `results/` (R18).
REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'

#: `langevin ... $(100.0*dt) <seed> zero yes tally yes` and
#: `velocity ... create <temp> <seed>` -- the two RNG seeds
#: `SimulationBase._run_seed` draws fresh with `random.randint` on every
#: build. Used only by the real (non-mocked) round-trip test below to mask
#: the one thing two builds of the same config are never expected to agree on.
_LANGEVIN_SEED_RE = re.compile(r'(langevin\s+[\d.]+\s+[\d.]+\s+\$\([\d.]+\*dt\)\s+)\d+')
_VELOCITY_SEED_RE = re.compile(r'(velocity\s+\S+\s+create\s+[\d.]+\s+)\d+')


def _mask_rng_seeds(text: str) -> str:
    text = _LANGEVIN_SEED_RE.sub(r'\1SEED', text)
    return _VELOCITY_SEED_RE.sub(r'\1SEED', text)


def test_round_trip_rebuilds_the_tree(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(10.0, 30.0), tiers=('window', 'raw'))
    ingest_root(root, label='rt')

    written = export_results('rt', tmp_path / 'out')

    out_run = tmp_path / 'out' / REL
    assert (out_run / 'provenance.json').is_file()
    # provenance/ is shared by every layer, so it lands one level up from L2
    # (at .../K300/provenance), matching src.run.tree.provenance_dir_for --
    # not inside L2 itself.
    assert (out_run.parent / 'provenance' / 'config.json').is_file()
    assert (out_run / 'lammps' / 'system.in').is_file()
    assert {p.name for p in (out_run / 'results').iterdir()} == {
        p.name for p in (root / REL / 'results').iterdir()}
    assert (tmp_path / 'out' / 'settings.yaml').is_file()
    assert written == sorted(written)


def test_exported_tree_is_discoverable_again(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    ingest_root(root, label='rt2')

    export_results('rt2', tmp_path / 'out')

    original, = discover_runs(root)
    exported, = discover_runs(tmp_path / 'out')
    assert exported.tokens == original.tokens


def test_exported_results_bytes_are_identical(aiida_profile_clean, tmp_path):
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='rt3')

    export_results('rt3', tmp_path / 'out')

    name = 'friction_f30_a0_s2_layer2.txt'
    assert (tmp_path / 'out' / REL / 'results' / name).read_bytes() == \
        (root / REL / 'results' / name).read_bytes()


def test_unknown_label_raises(aiida_profile_clean, tmp_path):
    with pytest.raises(LookupError, match='no simulation set'):
        export_results('nope', tmp_path / 'out')


def test_superseded_results_are_excluded_from_export(aiida_profile_clean, tmp_path):
    """A re-ingested load point supersedes its old node (Task 16); the two
    share the same run-relative file name, so export must only ever write
    back the CURRENT one, never the stale one it replaced.

    Asserts `_results_of` returns exactly one node directly (not just the
    end-to-end byte comparison below): a QueryBuilder with no `order_by`
    happens to return the new node last on this backend today, so the
    dump-order "last write wins" would pass the byte check by coincidence --
    the direct count is what actually pins the fix down.
    """
    from src.aiida.export import _results_of, resolve_set  # pylint: disable=protected-access
    from src.aiida.ingest import find_or_create_run
    from tests.helpers.synthetic_results import write_results

    root = tmp_path / 'simulation_super'
    make_run(root, REL, 'afm', loads=(30.0,))
    ingest_root(root, label='rt5')

    name = 'friction_f30_a0_s2_layer2.txt'
    old_bytes = (root / REL / 'results' / name).read_bytes()

    # Same load point (force=30, angle=0, speed=2, layer=2), different bytes
    # -- a re-run, not a duplicate.
    write_results(root / REL, 'afm', load_kind='force', loads=(30.0,),
                  layers=2, n=99)
    new_bytes = (root / REL / 'results' / name).read_bytes()
    assert new_bytes != old_bytes

    report = ingest_root(root, label='rt5')
    assert report.superseded_results

    set_node = resolve_set('rt5')
    loc, = discover_runs(root)
    run_node, created = find_or_create_run(set_node, loc)
    assert not created  # already exists from the ingest above

    results = _results_of(run_node)
    assert len(results) == 1
    assert not results[0].base.extras.get('superseded', False)

    export_results('rt5', tmp_path / 'out')

    assert (tmp_path / 'out' / REL / 'results' / name).read_bytes() == new_bytes


def test_oversize_run_files_are_logged_not_silently_dropped(
        aiida_profile_clean, tmp_path, caplog):
    """A file over `max_mb` is never stored (Task 16 R11c); export must say
    so rather than silently produce a tree missing it."""
    from src.aiida.data import FrictionSimulationData
    from src.aiida.ingest import find_or_create_set

    root = tmp_path / 'simulation_over'
    make_run(root, REL, 'afm', loads=(30.0,))
    loc, = discover_runs(root)
    set_node = find_or_create_set(root, 'rt-over')

    node = FrictionSimulationData()
    node.set_from_locator(loc, set_uuid=str(set_node.uuid), set_label=set_node.label)
    node.add_run_files(loc, max_mb=0)  # forces every candidate file "oversize"
    node.store()

    with caplog.at_level('WARNING', logger='src.aiida.export'):
        export_results('rt-over', tmp_path / 'out')

    assert not (tmp_path / 'out' / REL / 'provenance.json').exists()
    assert any('provenance.json' in message for message in caplog.messages)


def test_oversize_result_aux_files_are_logged_not_silently_dropped(
        aiida_profile_clean, tmp_path, caplog):
    """Same as above, for a `FrictionResultsData` aux file (tier files are
    always stored regardless of size -- only an aux file, like a hold/final
    artefact, is ever recorded as oversize)."""
    from src.aiida.data import FrictionResultsData
    from src.aiida.ingest import find_or_create_run, find_or_create_set

    root = tmp_path / 'simulation_over2'
    make_run(root, REL, 'afm', loads=(30.0,))
    (root / REL / 'data').mkdir(exist_ok=True)
    (root / REL / 'data' / 'final_f30.data').write_text('x' * 10, encoding='utf-8')
    loc, = discover_runs(root)
    set_node = find_or_create_set(root, 'rt-over2')
    run_node, _created = find_or_create_run(set_node, loc)

    tier_path = root / REL / 'results' / 'friction_f30_a0_s2_layer2.txt'
    aux_path = root / REL / 'data' / 'final_f30.data'
    node = FrictionResultsData()
    node.family = 'afm'
    node.load_kind = 'force'
    node.load = 30.0
    node.angle = 0.0
    node.speed = 2.0
    node.layers = 2
    node.run_uuid = str(run_node.uuid)
    node.set_uuid = str(run_node.set_uuid)
    node.add_result_files({'window': tier_path}, run_dir=loc.run_dir,
                          aux=(aux_path,), max_mb=0, content_sha256='deadbeef')
    node.store()

    with caplog.at_level('WARNING', logger='src.aiida.export'):
        export_results('rt-over2', tmp_path / 'out')

    assert not (tmp_path / 'out' / REL / 'data' / 'final_f30.data').exists()
    assert any('final_f30.data' in message for message in caplog.messages)


def test_reexport_overwrites_stale_files_and_writes_shared_provenance_once(
        aiida_profile_clean, tmp_path):
    """A re-export into a non-empty `dest` must overwrite whatever is already
    there -- from an earlier export or a hand-edit -- never trust it. The
    dedup that skips rewriting the shared AFM `provenance/**` subtree is
    about not rewriting a path THIS call already wrote, not about the
    filesystem; the multi-layer case (L1 and L2 share one `provenance/`)
    must still only write that shared subtree once per call.
    """
    root = tmp_path / 'simulation_multi'
    base = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300'
    make_run(root, f'{base}/L1', 'afm', loads=(30.0,))
    make_run(root, f'{base}/L2', 'afm', loads=(30.0,))
    ingest_root(root, label='rt-multi')

    out = tmp_path / 'out'
    written = export_results('rt-multi', out)
    shared = [p for p in written
             if p.as_posix().endswith(f'{base}/provenance/config.json')]
    assert len(shared) == 1

    provenance_config = out / base / 'provenance' / 'config.json'
    lammps_file = out / base / 'L1' / 'lammps' / 'system.in'
    original_config = provenance_config.read_bytes()
    original_lammps = lammps_file.read_bytes()

    provenance_config.write_bytes(b'corrupted')
    lammps_file.write_bytes(b'corrupted')

    export_results('rt-multi', out)

    assert provenance_config.read_bytes() == original_config
    assert lammps_file.read_bytes() == original_lammps


def test_sheet_like_provenance_stays_at_the_run_level(aiida_profile_clean, tmp_path):
    """A sheet-like run has no shared per-layer provenance -- unlike AFM,
    `provenance/` lives right next to `provenance.json`, inside the run
    directory itself (`src.run.tree.provenance_dir_for`'s "both in the same
    place" branch)."""
    root = tmp_path / 'simulation_y'
    rel = 'sheetonsheet/h-MoS2/100x_100y/K300'
    make_run(root, rel, 'sheetonsheet')
    ingest_root(root, label='rt4')

    export_results('rt4', tmp_path / 'out')

    out_run = tmp_path / 'out' / rel
    assert (out_run / 'provenance' / 'config.json').is_file()


class _FakeResult:
    """Stand-in for `GenerationResult` -- only the fields `rebuild_set` reads."""

    def __init__(self, created=None, simulation_root=None):
        self.created = created if created is not None else []
        self.simulation_root = simulation_root
        self.failures = []


def test_rebuild_calls_generate_package_once_per_run(aiida_profile_clean, tmp_path,
                                                     monkeypatch):
    """One call per DISTINCT stored config, not one per run and not one per
    model: a second `afm` run with a different material must still get its
    own call.

    `generate_package` is patched at `src.aiida.export.generate_package` (an
    allowed `src.aiida.` boundary), not `src.run.artifacts.generate_package`
    (refused by `tests/repo/test_mock_targets.py`'s ratchet) -- `rebuild_set`
    imports the name at module level specifically so this is the patch point
    that actually intercepts the call.
    """
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    # a SECOND afm run: rebuild must call the builder for it too, not once per model
    make_run(root, 'afm/h-WS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2', 'afm')
    make_run(root, 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'indent')
    ingest_root(root, label='rb')
    calls = []

    def _fake(config_file, model, **kwargs):
        calls.append((Path(config_file).name, model, kwargs))
        return _FakeResult(created=[tmp_path / 'built'],
                          simulation_root=tmp_path / 'built_root')

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    created, out_root = rebuild_set('rb', tmp_path / 'rebuilt')

    assert len(calls) == 3
    assert sorted(m for _c, m, _k in calls) == ['afm', 'afm', 'indent']
    assert all(name == 'config.json' for name, _m, _k in calls)
    assert all(k['simulation_root_name'] == '.' for _c, _m, k in calls)
    assert all(k['use_aiida'] is False for _c, _m, k in calls)
    assert all(k['allow_generate_amorphous'] is False for _c, _m, k in calls)
    assert out_root == tmp_path / 'rebuilt'
    assert len(created) == 3


def test_rebuild_uses_the_stored_settings_not_the_defaults(aiida_profile_clean,
                                                           tmp_path, monkeypatch):
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    ingest_root(root, label='rb2')
    seen = {}

    def _fake(config_file, model, **kwargs):
        # Checked from inside the call, not after `rebuild_set` returns: the
        # staging directory is a `tempfile.TemporaryDirectory()` (R17) and is
        # gone by the time the call comes back, so this is the only point at
        # which the exported settings.yaml can still be read.
        settings_file = kwargs['settings_file']
        seen['name'] = Path(settings_file).name
        seen['existed'] = Path(settings_file).is_file()
        return _FakeResult(simulation_root=tmp_path / 'r')

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    rebuild_set('rb2', tmp_path / 'rebuilt')

    assert seen['name'] == 'settings.yaml'
    assert seen['existed'] is True


def test_rebuild_refuses_an_unknown_label(aiida_profile_clean, tmp_path):
    with pytest.raises(LookupError, match='no simulation set'):
        rebuild_set('nope', tmp_path / 'rebuilt')


def test_rebuild_dedupes_shared_afm_config_across_layers(aiida_profile_clean,
                                                         tmp_path, monkeypatch):
    """`L1` and `L2` of one AFM layer sweep share ONE stored `config.json`
    (`run_simulations` already flattened the sweep into one member that built
    every layer) -- rebuilding must call `generate_package` once for the pair,
    not once per `L{n}` run node, or it would redo the whole layer sweep a
    second time.
    """
    root = tmp_path / 'simulation_multi'
    base = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300'
    make_run(root, f'{base}/L1', 'afm', loads=(30.0,))
    make_run(root, f'{base}/L2', 'afm', loads=(30.0,))
    ingest_root(root, label='rb-dedup')
    calls = []

    def _fake(config_file, model, **kwargs):
        calls.append(Path(config_file).resolve())
        return _FakeResult(created=[tmp_path / 'built'])

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    created, _out_root = rebuild_set('rb-dedup', tmp_path / 'rebuilt')

    assert len(calls) == 1
    assert len(created) == 1


def test_rebuild_stages_inputs_outside_dest(aiida_profile_clean, tmp_path,
                                            monkeypatch):
    """The staging export must live outside `dest`, in a real temp directory
    that is gone once `rebuild_set` returns -- never `dest/something`, which
    `JobManifest.from_simulation_directory`'s `rglob('lammps')` would also
    find once `generate_hpc=True` builds real `lammps/` directories into
    `dest`, duplicating every job (R17).
    """
    root = tmp_path / 'simulation_x'
    make_run(root, REL, 'afm')
    ingest_root(root, label='rb-stage')
    seen = {}

    def _fake(config_file, model, **kwargs):
        seen['config_dir'] = Path(config_file).resolve().parent
        seen['settings_dir'] = Path(kwargs['settings_file']).resolve().parent
        return _FakeResult()

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    dest = tmp_path / 'rebuilt'
    rebuild_set('rb-stage', dest)

    dest = dest.resolve()
    assert dest not in seen['config_dir'].parents and seen['config_dir'] != dest
    assert dest not in seen['settings_dir'].parents and seen['settings_dir'] != dest
    # the TemporaryDirectory is cleaned up on exit -- nothing was left behind
    assert not seen['config_dir'].exists()
    assert not seen['settings_dir'].exists()


def _minimal_sheet_flake_config(tmp_path):
    """A real, validating `SheetOnSheetSimulationConfig` with a `[Flake]`
    section -- `flake_slide.py`'s `FlakeSlideSimulation` route -- built the
    same way `tests/core/test_config.py`'s fixtures do (a dummy pot/cif file
    on disk, real content unneeded since nothing here builds a structure).

    `[Flake]` carries only `edge_length`: no `mat`/`pot_type`/`pot_path`/
    `cif_path` of its own -- the common single-material case, where the flake
    is carved from the `[2D]` sheet's own material rather than naming a
    second one. `ComponentConfig.check_pot_type`'s `None` guard (added
    alongside this fixture) is what makes this dump-and-reload safe at all;
    before it, an unset flake `pot_type` round-tripped through
    `model_dump_json()`'s explicit `null` into an `AttributeError` on reload,
    independent of the alias fix this fixture exists to exercise.
    """
    from src.core.config import SheetOnSheetSimulationConfig, load_settings

    dummy_pot = tmp_path / 'dummy.sw'
    dummy_cif = tmp_path / 'dummy.cif'
    dummy_pot.write_text('# dummy potential\n', encoding='utf-8')
    dummy_cif.write_text('# dummy cif\n', encoding='utf-8')

    return SheetOnSheetSimulationConfig(
        general={'temp': 300.0},
        **{'2D': {'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': str(dummy_pot),
                  'cif_path': str(dummy_cif), 'x': 50.0, 'y': 50.0,
                  'layers': [1], 'stack_type': 'AB'},
           'Flake': {'edge_length': 20.0}},
        settings=load_settings(),
    )


def test_normalize_config_aliases_lets_a_flake_slide_config_reload(tmp_path):
    """Unit-level check of `_normalize_config_aliases`, independent of the
    full `rebuild_set` flow covered below: a sheet-on-sheet config with a
    `[Flake]` section, dumped the way `run_simulations` stores it (field
    names, no `by_alias=True`) and normalized back, must reload with
    `flake` still populated -- `SheetOnSheetSimulationConfig.flake` has
    `alias='Flake'` and neither `populate_by_name` nor `validate_by_name`,
    so the un-normalized field-name key `'flake'` silently validates to
    `None` instead of raising.
    """
    from src.aiida.export import _normalize_config_aliases  # pylint: disable=import-outside-toplevel
    from src.core.config import SheetOnSheetSimulationConfig  # pylint: disable=import-outside-toplevel

    original = _minimal_sheet_flake_config(tmp_path)
    assert original.flake is not None  # sanity: the fixture really has a flake

    dumped = json.loads(original.model_dump_json())
    assert 'flake' in dumped and 'Flake' not in dumped  # field name, not the alias
    assert dumped['flake'] is not None

    # Isolate the flake defect from the '2D' one (also present, and covered by
    # the AFM-side tests/integration test): fix '2D' by hand first, so
    # reloading fails ONLY on the still-unfixed 'flake' key, silently
    # dropping it to its default rather than raising.
    isolated = dict(dumped)
    isolated['2D'] = isolated.pop('sheet')
    assert SheetOnSheetSimulationConfig(**isolated).flake is None

    _normalize_config_aliases(dumped, SheetOnSheetSimulationConfig)
    assert 'Flake' in dumped and 'flake' not in dumped

    reloaded = SheetOnSheetSimulationConfig(**dumped)
    assert reloaded.flake is not None
    assert reloaded.flake.edge_length == 20.0

    # Idempotent: normalizing an already-fixed dict changes nothing further.
    before = json.loads(json.dumps(dumped))
    _normalize_config_aliases(dumped, SheetOnSheetSimulationConfig)
    assert dumped == before


def test_rebuild_normalizes_flake_alias_so_the_config_routes_to_flake_slide(
        aiida_profile_clean, tmp_path, monkeypatch):
    """End-to-end (`generate_package` mocked at the allowed `src.aiida.`
    boundary): a stored sheet-on-sheet run whose `config.json` has a
    `[Flake]` section must reach `generate_package` with that section
    restored, or `run.simulate._select_sheet_builder_cls` -- the REAL
    selector, not a reimplementation of it -- would route it to the plain
    `SheetOnSheetSimulation` instead of `FlakeSlideSimulation`, silently
    building the wrong thing with no error at all.
    """
    from src.core.config import SheetOnSheetSimulationConfig  # pylint: disable=import-outside-toplevel
    from src.run.simulate import _select_sheet_builder_cls  # pylint: disable=import-outside-toplevel
    from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'simulation_flake'
    rel = 'sheetonsheet/h-MoS2/100x_100y/K300'
    make_run(root, rel, 'sheetonsheet')

    flake_cfg = _minimal_sheet_flake_config(tmp_path)
    assert flake_cfg.flake is not None
    (root / rel / 'provenance' / 'config.json').write_text(
        flake_cfg.model_dump_json(indent=2), encoding='utf-8')

    ingest_root(root, label='rb-flake')
    seen = {}

    def _fake(config_file, model, **kwargs):
        data = json.loads(Path(config_file).read_text(encoding='utf-8'))
        seen['config_obj'] = SheetOnSheetSimulationConfig(**data)
        return _FakeResult()

    monkeypatch.setattr('src.aiida.export.generate_package', _fake)

    rebuild_set('rb-flake', tmp_path / 'rebuilt')

    assert seen['config_obj'].flake is not None
    assert seen['config_obj'].flake.edge_length == 20.0
    assert _select_sheet_builder_cls('sheetonsheet',
                                     seen['config_obj']) is FlakeSlideSimulation


@pytest.mark.integration
def test_rebuild_round_trips_a_real_build_through_the_router(aiida_profile_clean,
                                                              tmp_path):
    """The one non-mocked check in this file: a stored `config.json` --
    `model_dump()` with `settings` embedded, as `run_simulations` writes it --
    actually round-trips through the real `generate_package` router, giving
    back the same run layout and the same LAMMPS inputs. Every other test
    here mocks `generate_package`, which only proves the plumbing calls it
    correctly, not that what it stores can be read back by the real builder.

    Compared after masking the `langevin`/`velocity create` RNG seeds: those
    are drawn fresh with `random.randint` on every build by design
    (`src.builders.simulation_base.SimulationBase._run_seed` -- "a fresh seed
    per run is mandatory", never reused across builds of the same config), so
    a second build of the identical config is expected to draw different
    seeds. Everything else in the script -- box, materials, groups, fixes,
    potentials, paths -- must still match exactly.
    """
    from tests.helpers.examples import build  # pylint: disable=import-outside-toplevel

    built_root = tmp_path / 'built'
    result = build('afm_config.ini', 'afm', built_root)
    if not result.created:
        pytest.skip("examples/afm_config.ini did not build in this environment "
                   f"(missing material/potential fixtures?): "
                   f"{[f.render() for f in result.failures]}")

    ingest_root(result.simulation_root, label='rb-real')

    rebuilt_dest = tmp_path / 'rebuilt'
    created, out_root = rebuild_set('rb-real', rebuilt_dest)

    assert created
    assert out_root == rebuilt_dest

    original = {loc.tokens.rel_path: loc for loc in discover_runs(result.simulation_root)}
    rebuilt = {loc.tokens.rel_path: loc for loc in discover_runs(rebuilt_dest)}
    assert original.keys() == rebuilt.keys()
    assert original  # discover_runs found at least one run in each tree

    for rel_path, orig_loc in original.items():
        rebuilt_loc = rebuilt[rel_path]
        orig_scripts = {p.name: _mask_rng_seeds(p.read_text(encoding='utf-8'))
                        for p in orig_loc.lammps_dir.glob('*.in')}
        rebuilt_scripts = {p.name: _mask_rng_seeds(p.read_text(encoding='utf-8'))
                           for p in rebuilt_loc.lammps_dir.glob('*.in')}
        assert orig_scripts, f"{orig_loc.lammps_dir} had no *.in scripts to compare"
        assert orig_scripts == rebuilt_scripts, f"{rel_path}: rebuilt *.in differs"
