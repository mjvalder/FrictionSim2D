"""Header-driven parser + strict LEGACY_2026 fallback (spec §1.1/§3.2, Task 3)."""
import pytest

from src.postprocessing.read_data import DataReader, SchemaError

LEGACY_10 = (
    "# Time-averaged data\n"
    "# TimeStep v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz "
    "v_comx_tip v_comy_tip v_comz_tip\n"
    + "\n".join(f"{i * 1000} " + " ".join(["1.0"] * 9) for i in range(5))
)

NEW_12 = (
    "# Time-averaged data\n"
    "# TimeStep v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz "
    "v_comx_tip v_comy_tip v_comz_tip v_etot v_temp_sheet\n"
    + "\n".join(f"{i * 1000} " + " ".join(["1.0"] * 11) for i in range(5))
)

HEADERLESS_12 = "\n".join(
    f"{i * 1000} " + " ".join(["1.0"] * 11) for i in range(5)
)

RAW_7 = (
    "# Time-averaged data\n"
    "# TimeStep v_fx_spr v_fy_spr v_fz_tip v_comx_tip v_comy_tip v_comz_tip\n"
    + "\n".join(f"{i * 100} " + " ".join(["1.0"] * 6) for i in range(5))
)


def _reader():
    return DataReader.__new__(DataReader)  # bypass directory-walking __init__


def test_header_driven_new_schema(tmp_path):
    p = tmp_path / "friction_f30_a0_s2_layer2.txt"
    p.write_text(NEW_12)
    df = _reader()._read_dataframe(p, 'tip')
    assert 'etotal' in df.columns and 'temp_sheet_3d' in df.columns
    assert df['nf'].iloc[0] == 1.0


def test_header_driven_legacy_names_renamed(tmp_path):
    # real 2026 files carry headers; v_comx etc must become comx, not survive raw
    p = tmp_path / "friction_f30_a0_s2_layer2.txt"
    p.write_text(LEGACY_10)
    df = _reader()._read_dataframe(p, 'tip')
    assert 'comz' in df.columns and 'v_comz' not in df.columns
    assert list(df.columns)[:4] == ['time', 'nf', 'lfx', 'lfy']


def test_raw_tier_rename_by_filename_prefix(tmp_path):
    # the same native names map to raw_* when the file is a raw_ trace
    p = tmp_path / "raw_f30_a0_s2_layer2.txt"
    p.write_text(RAW_7)
    df = _reader()._read_dataframe(p, 'tip')
    assert 'raw_lfx' in df.columns and 'lfx' not in df.columns


def test_legacy_positional_still_parses(tmp_path):
    p = tmp_path / "fc_ave_slide_30nN_0angle_2ms_l2.txt"
    p.write_text(HEADERLESS_12[:0] + "\n".join(
        f"{i * 1000} " + " ".join(["1.0"] * 9) for i in range(5)))
    df = _reader()._read_dataframe(p, 'tip')
    assert list(df.columns)[:4] == ['time', 'nf', 'lfx', 'lfy']


def test_headerless_wrong_width_raises_not_shifts(tmp_path):
    # spec §1.1: the old code silently shifted 12 columns into 10 names
    p = tmp_path / "fc_ave_slide_30nN_0angle_2ms_l2.txt"
    p.write_text(HEADERLESS_12)
    with pytest.raises(SchemaError):
        _reader()._read_dataframe(p, 'tip')


def test_tip_path_regex_survives_pot_segment():
    m = DataReader._PATH_PATTERN_TIP.search(
        "afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/pot_sw_h-MoS2/results")
    assert m


def test_sheet_path_regex_survives_pot_segment():
    m = DataReader._PATH_PATTERN_SHEET.search(
        "sheetvsheet/h-MoS2/100x_100y/K300/pot_sw_h-MoS2/results")
    assert m
    # group 3 must NOT swallow the pot_ segment as substrate metadata
    assert m.group(3) != 'pot_sw_h-MoS2'


def test_sheet_path_regex_legacy_unchanged():
    m = DataReader._PATH_PATTERN_SHEET.search(
        "sheetvsheet/h-MoS2/100x_100y/aSi/results")
    assert m and m.group(1) == 'h-MoS2' and m.group(3) == 'aSi'


PINNED_NF_MEAN = 29.99975435  # pinned from the pre-rewrite positional parser


def test_real_2026_file_cof_unchanged():
    from pathlib import Path
    fixdir = Path(__file__).parent / 'fixtures/legacy_2026'
    files = list(fixdir.glob('friction_f*.txt'))
    assert files, "legacy fixture missing"
    df = _reader()._read_dataframe(files[0], 'tip')
    assert abs(df['nf'].mean() - PINNED_NF_MEAN) < 1e-8
    # header path must also have renamed the COM columns (spec review M1)
    assert 'comz' in df.columns and 'v_comz' not in df.columns


SHEET_SCHEMA2 = (
    "# schema=2 lf_units=nN\n"
    "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"
    + "\n".join(f"{i * 1000} 2.0 1.0 0.5 0.25 -10.0" for i in range(5))
)


def test_schema2_file_is_flagged_and_not_rescaled(tmp_path):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA2)
    df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_schema'] == 2
    assert df['lfx'].iloc[0] == 2.0      # already nN, untouched


def test_missing_marker_reads_as_schema1(tmp_path):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA2.replace("# schema=2 lf_units=nN",
                                       "# Time-averaged data for fix fc_win"))
    df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_schema'] == 1


def test_sheet_window_frame_carries_lfx(tmp_path):
    # regression: rename_map's 'xfrict' shadowed _SHEET_COLUMN_RENAME's 'lfx',
    # so read_data's COF guard (needs lfx AND lfy AND nf) never fired and
    # every sheet run silently produced no COF at all
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA2)
    df = _reader()._read_dataframe(p, 'sheet')
    assert 'lfx' in df.columns and 'lfy' in df.columns and 'nf' in df.columns
    assert 'xfrict' not in df.columns
