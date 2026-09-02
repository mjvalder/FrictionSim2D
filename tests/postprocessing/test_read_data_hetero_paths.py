"""A heterostructure result must be filed under its own material key.

`src.run.naming.material_token` joins a hetero run's materials with `+`
(`h_MoS2+h_WS2`) and appends a `+flake-<mat>` suffix for a flake. The sheet
path pattern's character class omitted `+`, so `re.search` matched from after
it: `h_MoS2+h_WS2` resolved to material `h_WS2`. The run was not skipped --
it was pooled with homogeneous WS2 data under the wrong key, which is worse
than dropping it, because nothing downstream can tell.

These cases pin the resolutions. The homogeneous and `pot_`-segment cases are
here because widening a character class is exactly the kind of fix that
quietly changes what a greedy match consumes.
"""
import pytest

from src.postprocessing.read_data import DataReader


@pytest.mark.parametrize("path,expected", [
    # the bug: both materials must survive into the key
    ("sheetonsheet/h_MoS2+h_WS2/100x_100y/K300/results",
     ("h_MoS2+h_WS2", "100x_100y", "K300")),
    # a flake suffix is part of the material token too
    ("sheetonsheet/h_MoS2+flake-h_WS2/100x_100y/K300/results",
     ("h_MoS2+flake-h_WS2", "100x_100y", "K300")),
    # homogeneous must be untouched by the widening
    ("sheetonsheet/h_MoS2/100x_100y/K300/results",
     ("h_MoS2", "100x_100y", "K300")),
    # the pot_ segment stays non-capturing, so group 3 keeps its legacy meaning
    ("sheetvsheet/h_MoS2/100x_100y/K300/pot_sw_x/results",
     ("h_MoS2", "100x_100y", "K300")),
    # hetero + potential sweep together
    ("sheetvsheet/h_MoS2+h_WS2/100x_100y/K300/pot_sw_x/results",
     ("h_MoS2+h_WS2", "100x_100y", "K300")),
])
def test_sheet_path_resolves(path, expected):
    match = DataReader._PATH_PATTERN_SHEET.search(path)
    assert match is not None, f"no match for {path}"
    assert match.groups() == expected


def test_hetero_and_homogeneous_do_not_collide():
    """The whole point: two runs sharing a constituent stay distinct."""
    hetero = DataReader._PATH_PATTERN_SHEET.search(
        "sheetonsheet/h_MoS2+h_WS2/100x_100y/K300/results").group(1)
    homogeneous = DataReader._PATH_PATTERN_SHEET.search(
        "sheetonsheet/h_WS2/100x_100y/K300/results").group(1)
    assert hetero != homogeneous, (
        "a MoS2/WS2 heterostructure is being filed under the same key as "
        "homogeneous WS2 -- the '+' is missing from the material character class"
    )
