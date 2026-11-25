"""Which `outer_loop` keys each builder splits into separate run scripts.

`run.expand._run_scripts_per_build` consumes `SPLIT_OUTER_LOOPS` to promise a
job count, so a wrong value makes the GUI promise scripts the build never
writes. The static scans and the indent hold+retract never read the lateral
drive, so they split by load only (indent) or not at all (PES).
"""
import pytest

from frictionsim2d.builders.afm.hetero import HeteroIndentSimulation
from frictionsim2d.builders.afm.indent import IndentSimulation
from frictionsim2d.builders.afm.pes_tip import HeteroPESTipSimulation, PESTipSimulation
from frictionsim2d.builders.sheetonsheet.pes_sheet import (
    HeteroPESSheetSimulation,
    PESSheetSimulation,
)


@pytest.mark.parametrize("cls, expected", [
    (IndentSimulation, frozenset({"force"})),
    (HeteroIndentSimulation, frozenset({"force"})),
    (PESTipSimulation, frozenset()),
    (HeteroPESTipSimulation, frozenset()),
    (PESSheetSimulation, frozenset()),
    (HeteroPESSheetSimulation, frozenset()),
])
def test_split_outer_loops_by_class(cls, expected):
    assert cls.SPLIT_OUTER_LOOPS == expected


def test_hetero_indent_runs_the_indent_template_not_the_hetero_slide():
    assert HeteroIndentSimulation.SLIDE_TEMPLATE == "afm/indent.lmp"
