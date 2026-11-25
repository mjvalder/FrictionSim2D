"""The package version has one literal (``__version__`` in frictionsim2d/__init__.py,
read by hatch's dynamic version); everything else derives."""
from importlib.metadata import version

from click.testing import CliRunner

import frictionsim2d
from frictionsim2d.cli import cli


def test_version_matches_metadata():
    assert frictionsim2d.__version__ == version("FrictionSim2D")


def test_cli_version_matches():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert frictionsim2d.__version__ in result.output
