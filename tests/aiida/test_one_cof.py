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

#: The readers agree, but not to the exact bit pattern: a value that
#: round-trips through an AiiDA node attribute (`stored_attr`, `reparsed` via
#: the node's own re-parse of its stored copy) comes back rounded to ~14
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


@pytest.mark.parametrize('rel, family', [(AFM, 'afm'), (SHEET, 'sheetonsheet')])
def test_the_datareader_agrees_with_the_store(aiida_profile_clean, tmp_path,
                                              rel, family):
    """The fourth reader: `postprocess read` over the same tree."""
    root = tmp_path / 'simulation_cof2'
    make_run(root, rel, family, loads=(30.0,), tiers=('window',), n=60)
    ingest_root(root, label='cof2')
    _run_node, result_node = _stored_result('cof2')

    reader = DataReader(results_dir=str(root))

    assert len(reader.runs) == 1

    reader_cof = summarise_run(reader.runs[0].frames_by_tier(), family).mean_cof

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
