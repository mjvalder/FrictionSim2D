"""One parser for a results file, shared by the reader and the store."""
import pandas as pd
import pytest

from frictionsim2d.postprocessing.read_data import DataReader, read_results_file


def test_reads_by_header_and_renames_through_the_registry(tmp_path):
    path = tmp_path / 'friction_f30_a0_s2_layer2.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=afm\n'
        '# TimeStep v_fz_tip v_fx_spr v_fy_spr\n'
        '1000 30.0 3.0 0.0\n2000 30.0 3.5 0.0\n', encoding='utf-8')

    df = read_results_file(path, 'afm')

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['time', 'nf', 'lfx', 'lfy']
    assert df['nf'].tolist() == [30.0, 30.0]


def test_legacy_sheet_file_is_corrected_and_marked(tmp_path):
    """The 1.602x eV/A -> nN rescale must survive the move to module level.

    R3: the sheet family's native normal-load column is ``v_fz`` (AFM's is
    ``v_fz_tip``) -- using the AFM native name here would leave ``nf``
    unresolved and mask the very column the correction is checked against.
    """
    from frictionsim2d.postprocessing.read_data import LF_SCHEMA1_CORRECTION

    path = tmp_path / 'friction_p1.0_a0_s2.txt'
    # no `schema=` marker -> schema 1; lfx differs from v_sx -> LJ branch
    path.write_text(
        '# lf_units=eV/A family=sheetonsheet\n'
        '# TimeStep v_fz v_xfrict v_yfrict v_sx\n'
        '1000 30.0 1.0 0.0 2.0\n2000 30.0 1.0 0.0 2.0\n', encoding='utf-8')

    df = read_results_file(path, 'sheetonsheet')

    assert df.attrs['lf_schema'] == 1
    assert df.attrs['lf_corrected'] is True
    assert df['nf'].iloc[0] == 30.0
    assert df['lfx'].iloc[0] == pytest.approx(1.0 * LF_SCHEMA1_CORRECTION)


def test_schema_two_file_is_left_alone(tmp_path):
    path = tmp_path / 'friction_p1.0_a0_s2.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=sheetonsheet\n'
        '# TimeStep v_fz v_xfrict v_yfrict v_sx\n'
        '1000 30.0 1.0 0.0 2.0\n', encoding='utf-8')

    df = read_results_file(path, 'sheetonsheet')

    assert df.attrs['lf_schema'] == 2
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 1.0


def test_afm_family_is_never_corrected(tmp_path):
    path = tmp_path / 'friction_f30_a0_s2_layer2.txt'
    path.write_text(
        '# lf_units=nN family=afm\n'
        '# TimeStep v_fz_tip v_fx_spr v_fy_spr\n1000 30.0 3.0 0.0\n',
        encoding='utf-8')

    df = read_results_file(path, 'afm')

    assert df.attrs['lf_schema'] == 1
    assert df.attrs['lf_corrected'] is False
    assert df['lfx'].iloc[0] == 3.0


def test_n_layers_extends_the_rename_map(tmp_path):
    """REGISTRY declares only L1/L2; a four-layer run must still rename L3/L4."""
    path = tmp_path / 'friction_f30_a0_s2_layer4.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=afm\n'
        '# TimeStep v_fz_tip v_comx_L3 v_comx_L4\n1000 30.0 1.0 2.0\n',
        encoding='utf-8')

    two = read_results_file(path, 'afm', n_layers=2)
    four = read_results_file(path, 'afm', n_layers=4)

    assert 'v_comx_L3' in two.columns          # unrenamed at the default
    assert 'comx_L3' in four.columns
    assert 'comx_L4' in four.columns


def test_unknown_family_is_refused(tmp_path):
    path = tmp_path / 'friction_f30_a0_s2_layer2.txt'
    path.write_text('# a\n# TimeStep v_fz_tip\n1000 1.0\n', encoding='utf-8')

    with pytest.raises(ValueError, match='unknown family'):
        read_results_file(path, 'poke')


def test_raw_tier_is_picked_from_the_filename_prefix(tmp_path):
    path = tmp_path / 'raw_f30_a0_s2_layer2.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=afm\n'
        '# TimeStep v_fz_tip v_fx_spr\n100 30.0 3.0\n', encoding='utf-8')

    df = read_results_file(path, 'afm')

    assert 'raw_nf' in df.columns
    assert 'nf' not in df.columns


def test_layer_columns_are_named_for_the_requested_layer_count(tmp_path):
    """The third parameter is positional-or-keyword, and the map it builds is
    `rename_map`'s own -- not a second copy kept in this module."""
    from frictionsim2d.core.observables import rename_map

    natives = ['v_fz', 'v_xfrict', 'v_yfrict', 'v_comz_L5']
    path = tmp_path / 'friction_p1_a0_s2.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=sheetonsheet\n'
        '# TimeStep ' + ' '.join(natives) + '\n'
        '1000 1.0 2.0 0.0 6.5\n', encoding='utf-8')

    two = read_results_file(path, 'sheetonsheet')
    five = read_results_file(path, 'sheetonsheet', 5)

    assert 'v_comz_L5' in two.columns
    assert 'comz_L5' in five.columns
    assert rename_map('sheetonsheet', 'window', 5)['v_comz_L5'] == 'comz_L5'
