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
        "afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/pot_sw_h-MoS2/results")
    assert m


def test_tip_path_regex_matches_amorph_dash_token():
    """src.run.naming.output_dir_for (Task 3b) writes 'amorph-<mat>' for an
    amorphous substrate/tip -- the hyphen must survive the \\w-only groups."""
    m = DataReader._PATH_PATTERN_TIP.search(
        "afm/h-MoS2/100x_100y/sub_amorph-Si_tip_amorph-Si_r25/K300/results")
    assert m and m.group(2) == 'amorph-Si' and m.group(3) == 'amorph-Si'


def test_tip_path_regex_legacy_a_prefixed_token_still_matches():
    """Pre-3b trees on disk wrote a bare 'aSi'; still readable."""
    m = DataReader._PATH_PATTERN_TIP.search(
        "afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/results")
    assert m and m.group(2) == 'aSi'


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
    fixdir = Path(__file__).parents[1] / 'fixtures/legacy_2026'
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


# v_xfrict != v_sx -> LJ branch: xfrict is compute group/group's force in
# eV/A and needs correcting to nN.
SHEET_SCHEMA1_LJ = (
    "# Time-averaged data for fix fc_win\n"
    "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"
    + "\n".join(f"{i * 1000} 2.0 1.0 0.5 0.25 -10.0" for i in range(5))
)

# v_xfrict == v_sx -> many-body branch: the template defined xfrict AS v_sx,
# which was already scaled to nN. Nothing to correct.
SHEET_SCHEMA1_MB = (
    "# Time-averaged data for fix fc_win\n"
    "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"
    + "\n".join(f"{i * 1000} 0.5 0.25 0.5 0.25 -10.0" for i in range(5))
)


def test_legacy_lj_file_is_corrected_and_flagged(tmp_path, caplog):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA1_LJ)
    with caplog.at_level('WARNING'):
        df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_schema'] == 1
    assert df.attrs['lf_corrected'] is True
    assert df['lfx'].iloc[0] == pytest.approx(2.0 * 1.602176565)
    assert df['lfy'].iloc[0] == pytest.approx(1.0 * 1.602176565)
    assert any('1.602176565' in r.getMessage() for r in caplog.records)


def test_legacy_many_body_file_is_not_corrected(tmp_path):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA1_MB)
    df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 0.5      # already nN, untouched


def test_schema2_file_is_never_corrected(tmp_path):
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA2)
    df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 2.0


def test_afm_files_are_never_lf_corrected(tmp_path):
    # the AFM lateral force is a spring force, scaled at source, always nN
    p = tmp_path / "friction_f30_a0_s2_layer2.txt"
    p.write_text(NEW_12)
    df = _reader()._read_dataframe(p, 'tip')
    assert df.attrs['lf_corrected'] is False


def test_future_schema_is_not_corrected(tmp_path, caplog):
    # a substring test for 'schema=2' treats every LATER version as schema 1
    # and silently applies the legacy unit correction to a file that never
    # needed it. The parity work is queued to land on top of this.
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA2.replace('schema=2', 'schema=3'))
    with caplog.at_level('WARNING'):
        df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_schema'] == 3
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 2.0          # untouched
    assert any('not one this reader understands' in r.getMessage()
               for r in caplog.records)


def test_correction_is_idempotent(tmp_path):
    # applying it twice would give 1.602^2; attrs survive a re-read
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text(SHEET_SCHEMA1_LJ)
    r = _reader()
    df = r._read_dataframe(p, 'sheet')
    once = df['lfx'].iloc[0]
    again = r._correct_legacy_lf_schema1(df, p, 'sheet')
    assert again['lfx'].iloc[0] == once


def test_unidentifiable_branch_warns_rather_than_going_quiet(tmp_path, caplog):
    # raw_/peratom_ tiers carry no v_sx, so the branch cannot be determined.
    # That exit abandons a possibly-affected file and must say so.
    p = tmp_path / "raw_p1.0_a0_s1.txt"
    p.write_text("# Time-averaged data for fix fc_raw\n"
                 "# TimeStep v_xfrict v_yfrict v_fz\n"
                 "1000 2.0 1.0 -10.0\n")
    with caplog.at_level('WARNING'):
        df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_corrected'] is False
    assert any('cannot be identified' in r.getMessage()
               for r in caplog.records)


def test_no_lateral_column_is_not_warned_about(tmp_path, caplog):
    # 251030_results holds 491 files whose header is v_sx v_sy v_sz v_fx...
    # with no v_xfrict at all. They have no lateral force to correct, so
    # warning that "its COF is a factor 1.602 too small" was wrong.
    p = tmp_path / "friction_p1.0_a0_s1.txt"
    p.write_text("# Time-averaged data for fix fc_win\n"
                 "# TimeStep v_sx v_sy v_sz v_fz\n"
                 "1000 0.5 0.25 1.0 -10.0\n")
    with caplog.at_level('WARNING'):
        df = _reader()._read_dataframe(p, 'sheet')
    assert df.attrs['lf_corrected'] is False
    assert not [r for r in caplog.records if '1.602176565' in r.getMessage()]


def test_correction_provenance_survives_json_export():
    # attrs died in _NpEncoder, so a corrected frame serialised identically
    # to a stale pre-fix export -- and time_series_hash is computed from
    # (nf, lfx, lfy), so the correction changes it and dedup-by-hash cannot
    # recognise the re-ingested run.
    import json
    import pandas as pd
    from src.postprocessing.read_data import _NpEncoder

    df = pd.DataFrame({'nf': [1.0], 'lfx': [2.0], 'lfy': [1.0]})
    df.attrs['lf_schema'] = 1
    df.attrs['lf_corrected'] = True
    round_tripped = json.loads(json.dumps(df, cls=_NpEncoder))
    assert round_tripped['attrs']['lf_corrected'] is True
    assert round_tripped['attrs']['lf_schema'] == 1


def test_ev_a_to_nn_matches_the_corrector_constant():
    # the templates scale by EV_A_TO_NN; the corrector hard-codes its own
    # copy. If one is ever changed the other must move with it.
    from src.data.models import EV_A_TO_NN
    from src.postprocessing.read_data import DataReader
    assert DataReader._LF_SCHEMA1_CORRECTION == EV_A_TO_NN


def test_legacy_lj_file_still_corrects_after_react_x_rename(tmp_path):
    """Declaring react_x with rename_to='v_sx' renames away the column the
    frozen correction uses to tell the LJ branch from the many-body one.
    Without this, 7,166 retained files silently lose the x1.602176565
    correction -- reproduced during the 2026-08-28 review, where lfx reverted
    from 0.01287 to 0.00803 and lf_corrected flipped True -> False.

    Task 9 (not yet landed) is what will actually rename v_sx to react_x
    for a real file's header. Until then this is a characterization test:
    it passes today because v_sx still survives unrenamed, and it must keep
    passing unchanged once Task 9 lands the react_x declaration."""
    from src.postprocessing.read_data import DataReader
    f = tmp_path / 'fc_ave_slide_5GPa_0angle_1ms.txt'
    # no schema marker => schema 1; lfx != v_sx => LJ branch
    f.write_text(
        "# Time-averaged data for fix fc_ave\n"
        "# TimeStep v_xfrict v_yfrict v_sx v_sy v_sz\n"
        "1000 1.0 0.5 2.0 1.0 0.0\n"
    )
    df = DataReader()._read_dataframe(f, 'sheet')
    assert df.attrs['lf_schema'] == 1
    assert df.attrs['lf_corrected'] is True
    assert df['lfx'].iloc[0] == 1.0 * DataReader._LF_SCHEMA1_CORRECTION


def test_many_body_legacy_file_is_left_alone(tmp_path):
    # on the many-body branch the template defines xfrict AS v_sx, so the two
    # columns are bit-identical and there is nothing to correct
    from src.postprocessing.read_data import DataReader
    f = tmp_path / 'fc_ave_slide_5GPa_0angle_1ms.txt'
    f.write_text(
        "# Time-averaged data for fix fc_ave\n"
        "# TimeStep v_xfrict v_yfrict v_sx v_sy v_sz\n"
        "1000 2.0 1.0 2.0 1.0 0.0\n"
    )
    df = DataReader()._read_dataframe(f, 'sheet')
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 2.0


def test_audit_reads_native_names_not_renamed_ones():
    """The audit classifies from raw header tokens, so it is unaffected by
    the react_x rename. Pinned so a later 'tidy-up' does not route it
    through _read_dataframe and silently change what it classifies."""
    import inspect
    from src.postprocessing import audit_lf_units
    src = inspect.getsource(audit_lf_units)
    assert "'v_sx'" in src and '_read_dataframe' not in src


def test_raw_tier_unidentifiable_branch_warns_after_react_x_rename(tmp_path, caplog):
    """After Task 9 lands, a raw-tier sheet file's v_xfrict/v_yfrict rename
    to raw_lfx/raw_lfy (not lfx/lfy), so the discriminator lookup must also
    accept raw_lfx/raw_lfy -- otherwise the 'cannot be identified' warning
    that test_unidentifiable_branch_warns_rather_than_going_quiet pins today
    silently disappears once raw tier renaming changes.

    No file path can produce a raw_lfx column until Task 9 declares the
    RAW-tier channel, so this calls _correct_legacy_lf_schema1 directly on a
    hand-built frame that anticipates that rename."""
    import pandas as pd
    from src.postprocessing.read_data import DataReader

    df = pd.DataFrame({
        'raw_lfx': [2.0],
        'raw_lfy': [1.0],
        'nf': [-10.0],
    })
    df.attrs['lf_schema'] = 1
    r = DataReader.__new__(DataReader)
    with caplog.at_level('WARNING'):
        out = r._correct_legacy_lf_schema1(df, tmp_path / 'raw_p1.0_a0_s1.txt', 'sheet')
    assert out.attrs['lf_corrected'] is False
    assert out['raw_lfx'].iloc[0] == 2.0        # left untouched
    assert any('cannot be identified' in rec.getMessage()
               for rec in caplog.records)


def test_registry_is_the_only_sheet_rename_map():
    from src.postprocessing.read_data import DataReader
    assert not hasattr(DataReader, '_SHEET_COLUMN_RENAME'), (
        'the registry owns sheet renames now; a second hand-maintained map '
        'is what let rename_map shadow lfx and silently kill sheet COF')


def test_headerless_sheet_file_raises(tmp_path):
    """Surveyed 2026-08-29: of 24,064 candidate results files in the retained
    tree, zero non-empty files are headerless, and only 5 of 7,171 sheet files
    match the 17-column list the positional fallback described. Guessing at a
    headerless sheet file is more dangerous than refusing it."""
    import pytest
    from src.postprocessing.read_data import DataReader, SchemaError
    f = tmp_path / 'fc_ave_slide_5GPa_0angle_1ms.txt'
    # line 2 must NOT start with '#': _read_dataframe takes the header path
    # whenever it does, and would happily name the columns after whatever
    # tokens it found there.
    f.write_text("# Time-averaged data for fix fc_ave\n"
                 "1000 1.0 0.5 2.0\n"
                 "2000 1.1 0.6 2.1\n")
    with pytest.raises(SchemaError):
        DataReader()._read_dataframe(f, 'sheet')


def test_sheet_positional_list_is_gone():
    from src.postprocessing.read_data import DataReader
    assert not hasattr(DataReader, '_SHEET_FILE_COLUMNS')
    assert DataReader._AFM_FILE_COLUMNS[:3] == ['time', 'nf', 'lfx']


def test_family_marker_wins_over_path_inference(tmp_path):
    from src.postprocessing.read_data import DataReader
    f = tmp_path / 'fc_ave_slide_5GPa_0angle_1ms.txt'
    f.write_text(
        "# schema=2 lf_units=nN family=hetero\n"
        "# TimeStep v_xfrict v_yfrict\n"
        "1000 1.0 0.5\n"
    )
    assert DataReader._registry_sim_type('sheet', "# ... family=hetero") == 'hetero'


def test_missing_family_marker_falls_back_to_the_traversal_vocabulary():
    # every one of the 7,171 retained sheet files predates the marker
    from src.postprocessing.read_data import DataReader
    assert DataReader._registry_sim_type('sheet', '# schema=2 lf_units=nN') == 'sheetonsheet'
    assert DataReader._registry_sim_type('tip', '') == 'afm'


def test_unknown_family_marker_is_refused():
    import pytest
    from src.postprocessing.read_data import DataReader
    with pytest.raises(ValueError, match='family'):
        DataReader._registry_sim_type('sheet', '# family=bogus')
