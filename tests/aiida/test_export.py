"""export-results is the inverse of ingest (spec §4.2)."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from src.aiida.export import export_results  # noqa: E402
from src.aiida.ingest import ingest_root  # noqa: E402
from src.run.tree import discover_runs  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

#: An AFM run: `rel_path` ends in `/K300/L2`, so -- exactly like a real
#: AFMSimulation build -- `make_run` writes `provenance/` one level ABOVE the
#: `L2` directory (shared by every layer), not inside it. `provenance.json`
#: itself stays inside `L2`, alongside `lammps/` and `results/` (R18).
REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


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
