"""`FrictionSim2D run` exit status."""
def test_run_exits_nonzero_when_a_build_fails(monkeypatch, tmp_path):
    """`FrictionSim2D run` used to print '0/1 successful' and exit 0."""
    from click.testing import CliRunner  # pylint: disable=import-outside-toplevel
    from frictionsim2d import cli as cli_mod  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(cli_mod, 'run_simulations',
                        lambda **_: ([], tmp_path, [object()], None))
    ini = tmp_path / 'x.ini'
    ini.write_text('[general]\n')
    result = CliRunner().invoke(cli_mod.cli, ['run', 'sheetonsheet', str(ini),
                                              '--output-dir', str(tmp_path)])
    assert result.exit_code == 1
    assert '0/1 successful' in result.output
