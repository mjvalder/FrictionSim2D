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
