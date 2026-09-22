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

#: One AFM results file: two comment lines then 10 whitespace-separated
#: columns (see DataReader._AFM_FILE_COLUMNS), as fix ave/time writes them.
_HEADER = (
    "# Time-averaged data for fix fc_ave\n"
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
    """An AFM results tree holding one positive and one negative scan angle."""
    results = (tmp_path / "afm" / "h_MoS2" / "100x_100y" /
               "sub_aSi_tip_cSi_r25" / "L1" / "results")
    results.mkdir(parents=True)
    _write_results_file(results, "friction_f5_a0_s1_layer1.txt")
    _write_results_file(results, "friction_f5_a-15_s1_layer1.txt")
    return tmp_path


def test_negative_scan_angle_is_read(tmp_path: Path) -> None:
    """A negative scan angle is a documented, buildable run (outer_loop
    scan_angle writes slide_am15.in for -15), and LAMMPS writes its results
    as friction_f5_a-15_s1_layer1.txt. An unsigned filename pattern skips
    that file silently, so the run just disappears from every analysis.

    Over the helper tree rather than the bare `results_root` fixture below:
    the reader walks with `discover_runs` now, which keys on `provenance.json`
    and finds nothing in a directory that has none."""
    make_family_run(tmp_path, 'afm', loads=(5.0,), angles=(0.0, -15.0),
                    speeds=(1.0,), layers=1)

    reader = DataReader(results_dir=str(tmp_path))

    assert reader.metadata["loads_by_kind"]["force"] == {"5.0": [-15.0, 0.0]}
    assert sorted(run.angle for run in reader.runs) == [-15.0, 0.0]


def test_negative_scan_angle_reaches_the_nested_tree(results_root: Path) -> None:
    """The per-angle leaf keys the plotting layer walks (a{angle}) must carry
    the negative angle too, not just the metadata summary."""
    reader = DataReader(results_dir=str(results_root))

    loads = (reader.full_data_nested["h_MoS2"]["100x100y"]["aSi"]
             ["cSi"]["r25"]["l1"]["s1"])
    assert set(loads["f5.0"]) == {"a0", "a-15"}
