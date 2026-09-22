"""The store, the reader and the central row report ONE COF (spec section 3 F5).

No LAMMPS: `summarise_run` consumes frames, so synthetic registry-conformant
results exercise the identical path. What is being pinned is that four
independent readers of the same bytes agree to the last float.

Filed under `tests/aiida/` rather than `tests/integration/` (task-32a-brief.md's
nominal location): `aiida_profile_clean` is only registered by
`tests/aiida/conftest.py`, which star-imports `aiida.tools.pytest_fixtures`
guarded behind `importorskip`.
"""
import pytest

pytest.importorskip('aiida')

from src.aiida.data import FrictionResultsData          # noqa: E402
from src.aiida.export import resolve_set                # noqa: E402
from src.aiida.ingest import ingest_root                 # noqa: E402
from src.aiida.query import Friction2DDB                 # noqa: E402
from src.data.models import compute_friction_stats, summarise_run  # noqa: E402
from src.postprocessing.read_data import DataReader, read_results_file  # noqa: E402
from src.run.tree import discover_runs                   # noqa: E402
from tests.helpers.synthetic_results import make_run      # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.aiida]

AFM = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'
SHEET = 'sheetonsheet/h-MoS2/100x_100y/K300'

#: The four readers agree, but not to the exact bit pattern: a value that
#: round-trips through an AiiDA node attribute (`stored_attr`, `reparsed` via
#: the node's own re-parse of its stored copy, `record.mean_cof` via
#: `to_record`'s `attrs.get('mean_cof')`) comes back rounded to ~14
#: significant digits by the storage backend -- observed directly:
#: `psql_dos` gave back `1.4142135623731` for a freshly computed
#: `1.4142135623730947`, a ~3.8e-14 relative difference. `direct.mean_cof`
#: (computed straight from the frame, no store round trip) matches `kernel`
#: exactly. A real F5 defect -- e.g. the dropped legacy correction the third
#: test below pins -- is a factor of `LF_SCHEMA1_CORRECTION` (~1.6x) off,
#: many orders of magnitude past this tolerance, so loosening the equality
#: this far cannot mask it.
COF_TOL = 1e-9


def _stored_result(label):
    found = Friction2DDB().query(set_label=label)
    assert found.total_count == 1, found.total_count
    return found.runs[0], found.results[0]


@pytest.mark.parametrize('rel, family, load_kind', [
    (AFM, 'afm', 'force'),
    (SHEET, 'sheetonsheet', 'pressure'),
])
def test_one_cof(aiida_profile_clean, tmp_path, rel, family, load_kind):
    root = tmp_path / 'simulation_cof'
    run_dir = make_run(root, rel, family, loads=(30.0,), tiers=('window',), n=60)
    locator, = discover_runs(root)
    window_file = next((run_dir / 'results').glob('friction_*.txt'))

    report = ingest_root(root, label='cof')
    assert report.ok(), report.render()
    run_node, result_node = _stored_result('cof')

    # 1. the kernel, straight off the file
    frame = read_results_file(window_file, family,
                              n_layers=locator.tokens.layers or 2)
    kernel = compute_friction_stats(frame['nf'].to_numpy(),
                                    frame['lfx'].to_numpy(),
                                    frame['lfy'].to_numpy(), 0.2)

    # 2. summarise_run over the same frame
    direct = summarise_run({'window': frame}, family)

    # 3. the stored node's attribute, and the node re-parsing its OWN copy
    stored_attr = result_node.base.attributes.get('mean_cof')
    reparsed = summarise_run(result_node.frames_by_tier(), family)

    # 4. the central row -- the real SET node (R26): `to_record`'s second
    # argument reads `.source_host`/`.label` off it, which only the
    # `FrictionSimulationSetData` `resolve_set` returns carries. A run node
    # from `Friction2DDB().query(...).runs` has neither and would AttributeError.
    set_node = resolve_set('cof')
    record = result_node.to_record(run_node, set_node,
                                   uploader='m', contact='m@example.org')

    assert kernel['mean_cof'] is not None
    assert direct.mean_cof == pytest.approx(kernel['mean_cof'], rel=COF_TOL)
    assert stored_attr == pytest.approx(kernel['mean_cof'], rel=COF_TOL)
    assert reparsed.mean_cof == pytest.approx(kernel['mean_cof'], rel=COF_TOL)
    assert record.mean_cof == pytest.approx(kernel['mean_cof'], rel=COF_TOL)
    assert record.time_series_hash == result_node.base.attributes.get(
        'time_series_hash')


@pytest.mark.parametrize('rel, family', [(AFM, 'afm'), (SHEET, 'sheetonsheet')])
def test_the_datareader_agrees_with_the_store(aiida_profile_clean, tmp_path,
                                              rel, family):
    """The fourth reader: `postprocess read` over the same tree."""
    root = tmp_path / 'simulation_cof2'
    make_run(root, rel, family, loads=(30.0,), tiers=('window',), n=60)
    ingest_root(root, label='cof2')
    _run_node, result_node = _stored_result('cof2')

    reader = DataReader(results_dir=str(root))
    frames = []

    def _walk(node):
        if hasattr(node, 'columns'):
            frames.append(node)
            return
        for child in node.values():
            _walk(child)

    _walk(reader.full_data_nested)
    assert len(frames) == 1

    reader_cof = summarise_run({'window': frames[0]}, family).mean_cof

    assert reader_cof == pytest.approx(
        result_node.base.attributes.get('mean_cof'), rel=COF_TOL)


def test_a_legacy_sheet_file_is_corrected_on_both_paths(aiida_profile_clean,
                                                        tmp_path):
    """The correction the store used to drop entirely.

    A pre-`schema=2` sheet file on the LJ branch carries lateral force in eV/A
    against a load already in nN. If `read_results_file` skips
    `correct_legacy_lf_schema1`, the stored COF is 1.602176565x too small and
    the reader's is right -- two numbers from one file.

    The header uses the sheet-native normal-load column `v_fz` (not the
    AFM-native `v_fz_tip`, which the sheet family's rename map never maps to
    `nf` -- see `src/core/observables.py`'s sheet `nf` entry, `rename_to=
    'v_fz'`, vs. the AFM/indent-only legacy `nf` entry, `rename_to=
    'v_fz_tip'`). The wrong column name would leave the `normal_load` role
    unresolved, `mean_cof` stuck at `None`, and the assertion below comparing
    `None` to a real float -- caught explicitly so the test cannot pass
    vacuously.
    """
    from src.postprocessing.read_data import LF_SCHEMA1_CORRECTION

    root = tmp_path / 'simulation_legacy'
    run_dir = make_run(root, SHEET, 'sheetonsheet', loads=(1.0,), tiers=())
    results = run_dir / 'results'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'friction_p1_a0_s2.txt').write_text(
        '# lf_units=eV/A family=sheetonsheet\n'
        '# TimeStep v_fz v_xfrict v_yfrict v_sx\n'
        + ''.join(f'{(i + 1) * 1000} 30.0 1.0 0.0 2.0\n' for i in range(40)),
        encoding='utf-8')

    report = ingest_root(root, label='legacy')
    assert report.ok(), report.render()
    _run_node, result_node = _stored_result('legacy')

    frame = result_node.frame('window')

    assert frame.attrs['lf_schema'] == 1
    assert frame.attrs['lf_corrected'] is True
    expected = 1.0 * LF_SCHEMA1_CORRECTION / 30.0
    mean_cof = result_node.base.attributes.get('mean_cof')
    assert mean_cof is not None
    assert mean_cof == pytest.approx(expected)


def test_an_indent_row_publishes_no_cof(aiida_profile_clean, tmp_path):
    """m3: an indent window is a static hold; its "COF" is noise and must not
    reach the central table. The normal force and retract features do."""
    root = tmp_path / 'simulation_indent'
    run_dir = make_run(root, 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1',
                       'indent', loads=(30.0,), n=60)
    (run_dir / 'results' / 'indent_f30N_l1.csv').write_text(
        '# family=indent\nstep,z_tip,fz_tip,fx_tip,fy_tip,pe\n'
        '0,0,5.0,0,0,-1\n1,1,-2.0,0,0,-1\n2,2,-1.0,0,0,-1\n', encoding='utf-8')
    assert ingest_root(root, label='indent-cof').ok()
    run_node, result_node = _stored_result('indent-cof')

    record = result_node.to_record(run_node, resolve_set('indent-cof'))

    for stat in ('mean_cof', 'std_cof', 'mean_lf', 'std_lf',
                 'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy'):
        assert getattr(record, stat) is None, stat
    assert record.mean_nf == result_node.base.attributes.get('mean_nf')
    assert record.force_nN == 30.0
    assert record.extras['pulloff_force'] == pytest.approx(2.0)
    assert 'work_of_adhesion_nnA' in record.extras
    # the local store keeps what summarise_run computed; only publishing drops it
    assert result_node.base.attributes.get('mean_cof') is not None
