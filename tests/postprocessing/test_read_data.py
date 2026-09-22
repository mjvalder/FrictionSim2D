"""Reading a generated results tree back in.

The filenames are written by the LAMMPS templates (``results_file_pattern``
in the builders) and parsed here, so the two have to agree character for
character -- a file the parser does not recognise is skipped in silence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run

_HEADER = (
    "# schema=2 lf_units=nN family=afm\n"
    "# TimeStep v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz "
    "v_comx_tip v_comy_tip v_comz_tip\n"
)


def _write_results_file(directory: Path, name: str) -> None:
    rows = "".join(
        f"{1000 * (i + 1)} 5.0 1.0 0.5 1.0 1.0 1.0 2.0 2.0 3.0\n"
        for i in range(5)
    )
    (directory / name).write_text(_HEADER + rows, encoding="utf-8")


@pytest.fixture(name="results_root")
def _results_root(tmp_path: Path) -> Path:
    """An AFM run holding one positive and one negative scan angle."""
    run_dir = make_family_run(tmp_path, 'afm', layers=1, loads=(5.0,),
                              angles=(0.0,), speeds=(1.0,))
    results = run_dir / "results"
    for stale in results.iterdir():
        stale.unlink()
    _write_results_file(results, "friction_f5_a0_s1_layer1.txt")
    _write_results_file(results, "friction_f5_a-15_s1_layer1.txt")
    return tmp_path


def test_negative_scan_angle_is_read(results_root: Path) -> None:
    """A negative scan angle is a documented, buildable run (outer_loop
    scan_angle writes slide_am15.in for -15), and LAMMPS writes its results
    as friction_f5_a-15_s1_layer1.txt. An unsigned filename pattern skips
    that file silently, so the run just disappears from every analysis."""
    reader = DataReader(results_dir=str(results_root))

    assert reader.metadata["loads_by_kind"]["force"]["5.0"] == [-15.0, 0.0]


def test_negative_scan_angle_reaches_the_runs(results_root: Path) -> None:
    """The angle the plotting layer filters on must carry the sign."""
    reader = DataReader(results_dir=str(results_root))

    assert sorted(run.angle for run in reader.runs) == [-15.0, 0.0]
    assert {run.load for run in reader.runs} == {5.0}
