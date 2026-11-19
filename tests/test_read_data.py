"""Reading a generated results tree back in.

The filenames are written by the LAMMPS templates (``results_file_pattern``
in the builders) and parsed here, so the two have to agree character for
character -- a file the parser does not recognise is skipped in silence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.postprocessing.read_data import DataReader

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


def test_negative_scan_angle_is_read(results_root: Path) -> None:
    """A negative scan angle is a documented, buildable run (outer_loop
    scan_angle writes slide_am15.in for -15), and LAMMPS writes its results
    as friction_f5_a-15_s1_layer1.txt. An unsigned filename pattern skips
    that file silently, so the run just disappears from every analysis."""
    reader = DataReader(results_dir=str(results_root))

    assert sorted(reader.metadata["forces_and_angles"][5.0]) == [-15, 0]


def test_negative_scan_angle_reaches_the_nested_tree(results_root: Path) -> None:
    """The per-angle leaf keys the plotting layer walks (a{angle}) must carry
    the negative angle too, not just the metadata summary."""
    reader = DataReader(results_dir=str(results_root))

    loads = (reader.full_data_nested["h_MoS2"]["100x100y"]["aSi"]
             ["cSi"]["r25"]["l1"]["s1"])
    assert set(loads["f5.0"]) == {"a0", "a-15"}


# --- pre-schema-2 sheet lateral-force unit correction ------------------------
# This reader is a fork of the upstream FrictionSim2D one. Sheet runs on the
# LJ branch emitted xfrict as compute group/group's force in eV/A while the
# normal load was already nN, so their COF is a factor 1.602176565 too SMALL.
# Keep this in sync with the upstream tests/test_read_data_schema.py.

_SHEET_HDR = "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"


def _sheet_file(tmp_path, line1, row):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(f"{line1}\n{_SHEET_HDR}1000 {row}\n")
    return p


def _read(p):
    from src.postprocessing.read_data import DataReader
    return DataReader.__new__(DataReader)._read_dataframe(p, 'sheet')


def test_legacy_lj_sheet_file_is_corrected(tmp_path):
    # v_xfrict != v_sx -> LJ branch
    df = _read(_sheet_file(tmp_path, "# Time-averaged data",
                           "2.0 1.0 0.5 0.25 -10.0"))
    assert df.attrs['lf_schema'] == 1
    assert df.attrs['lf_corrected'] is True
    assert abs(df['lfx'].iloc[0] - 2.0 * 1.602176565) < 1e-9


def test_legacy_many_body_sheet_file_is_not_corrected(tmp_path):
    # v_xfrict == v_sx -> the template defined xfrict AS v_sx, already nN
    df = _read(_sheet_file(tmp_path, "# Time-averaged data",
                           "0.5 0.25 0.5 0.25 -10.0"))
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 0.5


def test_schema2_sheet_file_is_not_corrected(tmp_path):
    df = _read(_sheet_file(tmp_path, "# schema=2 lf_units=nN",
                           "2.0 1.0 0.5 0.25 -10.0"))
    assert df.attrs['lf_schema'] == 2
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 2.0


def test_future_schema_is_left_alone(tmp_path):
    # a substring test for 'schema=2' would treat this as schema 1 and scale it
    df = _read(_sheet_file(tmp_path, "# schema=3 lf_units=nN",
                           "2.0 1.0 0.5 0.25 -10.0"))
    assert df.attrs['lf_schema'] == 3
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 2.0
