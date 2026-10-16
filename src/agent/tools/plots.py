"""Plot tools: propose a plot for the Plots tab.

The plot waits for the user (or is added with an Undo, in direct mode);
nothing here touches the Plots tab or a file.
"""
from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field

from ...postprocessing.plot_config import PLOT_TYPES
from ..changes import ToolInputError
from ..plot_changes import build_plot_change


class PlotField(BaseModel):
    """One plot field; plot_schema lists the keys a plot type accepts."""

    key: str = Field(description='Field name, e.g. x_axis')
    value: Any


class PlotProposalRequest(BaseModel):
    """One plot: its type, the datasets it reads, and its fields."""

    plot_type: Literal[PLOT_TYPES] = Field(description='One of plot_types')
    datasets: List[str] = Field(
        default_factory=list,
        description='Dataset folders as find_result_sets shows them, e.g. workspace:/campaign')
    fields: List[PlotField] = Field(
        default_factory=list,
        description='Plot fields from plot_schema, e.g. key x_axis value force')
    title: str = Field('', description='A short name for the proposal')


def propose_plot(deps, request: PlotProposalRequest) -> str:
    """Propose a plot in the Plots tab, adding the datasets it needs."""
    fields = {}
    for item in request.fields:
        if item.key in fields:
            raise ToolInputError(f'field {item.key} given twice; give each field once')
        fields[item.key] = item.value
    return deps.propose(build_plot_change(deps, request.title, request.plot_type,
                                          fields, request.datasets))
