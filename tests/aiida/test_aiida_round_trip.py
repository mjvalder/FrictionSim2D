"""Generate -> ingest -> export-results -> DataReader equality (AiiDA spec §7).

Lives under `tests/aiida/` (not `tests/integration/`) so `aiida_profile_clean`
-- registered only by `tests/aiida/conftest.py` -- is in scope: every test
here uses it and never opens the live `friction2d` profile.
"""
import shutil
from pathlib import Path

import pytest

pytest.importorskip('aiida')

from src.aiida.export import export_results          # noqa: E402
from src.aiida.ingest import ingest_root             # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from src.run.tree import discover_runs               # noqa: E402
from tests.helpers.examples import CASES, HETERO, build  # noqa: E402
from tests.helpers.synthetic_results import (            # noqa: E402
    PES_GRID_CSV, load_kind_for, make_run, write_results)

pytestmark = [pytest.mark.integration, pytest.mark.aiida]

#: Every shipped example. `HETERO` is added below under its own lmp_serial
#: guard; the `flake` family has no shipped example at all and is covered
#: separately, from a synthetic tree.
ROUND_TRIP_CASES = [(name, model) for name, model, _members in CASES]


def _fill_results(simulation_root: Path) -> None:
    """Write synthetic results into a freshly built tree, keeping its provenance.

    A PES run has no `friction_*` tiers at all -- its result is one grid CSV, and
    the registry declares no channels for `pes_tip`/`pes_sheet`, so `write_results`
    would emit a header with no columns. Write the grid instead.
    """
    for locator in discover_runs(simulation_root):
        family = locator.tokens.family
        if family in PES_GRID_CSV:
            locator.results_dir.mkdir(parents=True, exist_ok=True)
            (locator.results_dir / 'pes_scan.csv').write_text(
                PES_GRID_CSV[family], encoding='utf-8')
            continue
        write_results(locator.run_dir, family,
                      load_kind=load_kind_for(locator.tokens.model),
                      loads=(10.0, 30.0), angles=(0.0, 90.0), speeds=(2.0,),
                      tiers=('window', 'raw'), layers=locator.tokens.layers, n=40)


def _result_files(root: Path):
    """Every results file under `root`, keyed by its path relative to `root`."""
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob('results/*')) if path.is_file()}


def _signature(root: Path):
    """A comparable fingerprint of everything DataReader found under `root`.

    Keyed by the nested traversal DataReader builds, valued by
    (column names, row count, the frame serialised) so a silent column shift or
    a dropped run fails the comparison.
    """
    reader = DataReader(results_dir=str(root))
    out = {}

    def _walk(node, path):
        if hasattr(node, 'columns'):
            out[path] = (tuple(node.columns), len(node),
                         node.to_csv(index=False, float_format='%.10g'))
            return
        for key, child in node.items():
            _walk(child, path + (key,))

    _walk(reader.full_data_nested, ())
    return out


@pytest.mark.parametrize('name, model', ROUND_TRIP_CASES)
def test_round_trip(aiida_profile_clean, tmp_path, name, model):
    built = tmp_path / 'built'
    result = build(name, model, built)
    assert not result.failures, [f.render() for f in result.failures]
    root = result.simulation_root
    _fill_results(root)

    report = ingest_root(root, label=f'rt-{name}')
    assert report.ok(), report.render()
    assert report.created_runs
    assert report.created_results

    exported = tmp_path / 'exported'
    export_results(f'rt-{name}', exported)

    # every results file comes back byte-identical (this is what covers the PES
    # grids, which DataReader does not read at all)
    assert _result_files(exported) == _result_files(root)
    assert _signature(exported) == _signature(root)


@pytest.mark.parametrize('name, model', ROUND_TRIP_CASES)
def test_second_ingest_creates_nothing(aiida_profile_clean, tmp_path, name, model):
    built = tmp_path / 'built'
    result = build(name, model, built)
    root = result.simulation_root
    _fill_results(root)
    ingest_root(root, label=f'idem-{name}')

    second = ingest_root(root, label=f'idem-{name}')

    assert second.created_runs == []
    assert second.created_results == []
    assert second.skipped_results


@pytest.mark.skipif(shutil.which('lmp_serial') is None,
                    reason='hetero_stack builds with lmp_serial')
def test_round_trip_hetero(aiida_profile_clean, tmp_path):
    name, model, _members = HETERO
    built = tmp_path / 'built'
    result = build(name, model, built)
    assert not result.failures, [f.render() for f in result.failures]
    root = result.simulation_root
    _fill_results(root)

    report = ingest_root(root, label='rt-hetero')
    assert report.ok(), report.render()

    exported = tmp_path / 'exported'
    export_results('rt-hetero', exported)

    assert _signature(exported) == _signature(root)


def test_round_trip_flake_synthetic(aiida_profile_clean, tmp_path):
    """No example .ini produces family='flake', so the shipped-example cases
    above never exercise it. Cover it directly from a synthetic tree. Since
    flake slides apply a load (2026-09-22) the synthetic window file carries
    `v_fz`, so the ingested results have a COF like any sheet run.
    """
    root = tmp_path / 'simulation_flake'
    rel = 'sheetonsheet/h-MoS2+flake-h-WS2/100x_100y/K300'
    make_run(root, rel, 'flake', loads=(10.0, 30.0), tiers=('window', 'raw'))

    report = ingest_root(root, label='rt-flake')
    assert report.ok(), report.render()
    assert report.created_runs
    assert report.created_results

    from aiida.orm import QueryBuilder
    from src.aiida.data import FrictionResultsData
    nodes = QueryBuilder().append(FrictionResultsData).all(flat=True)
    assert len(nodes) == 2
    for node in nodes:
        assert node.base.attributes.get('mean_cof') is not None
        assert node.base.attributes.get('is_complete') is True

    exported = tmp_path / 'exported'
    export_results('rt-flake', exported)

    assert _result_files(exported) == _result_files(root)
    assert _signature(exported) == _signature(root)
