"""One plot spec under edit.

Presents a `plot_config` model through the same small interface as
:class:`~frictionsim2d.gui.controllers.document.ConfigDocument`, so the Plots tab is
built by the same `ConfigForm` machinery as the config and settings tabs
rather than a third parallel one -- and so a field added to `plot_config.py`
appears in the GUI with no GUI change at all.

Two sections, not one: a plot model has sixty-odd fields, and the split
between "what every plot takes" and "what this type adds" is the split the
schema already makes.

Qt-free.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ...postprocessing.plot_config import (PLOT_TYPES, CofHistogramPlot,
                                           ContactCofScatterPlot,
                                           CorrelationPlot, IntegrityPanelPlot,
                                           PlotBase, RankFrictionPlot,
                                           ScatterComparisonPlot,
                                           SheetProfilePlot,
                                           StickslipPeriodHistogramPlot,
                                           StickSlipPlot, SummaryPlot,
                                           TimeseriesPlot)
from ..schema.introspect import SectionSpec, describe_section
from ..schema.overrides import PLOT_MODEL

__all__ = ['PlotEditorDocument', 'MODEL_BY_PLOT_TYPE', 'COMMON_SECTION']

#: The section holding `PlotBase`'s fields.
COMMON_SECTION = 'common'

MODEL_BY_PLOT_TYPE: Dict[str, type] = {
    'summary': SummaryPlot,
    'timeseries': TimeseriesPlot,
    'stick_slip_analysis': StickSlipPlot,
    'scatter_comparison': ScatterComparisonPlot,
    'rank_friction': RankFrictionPlot,
    'correlation': CorrelationPlot,
    'cof_histogram': CofHistogramPlot,
    'stickslip_period_histogram': StickslipPeriodHistogramPlot,
    'contact_cof_scatter': ContactCofScatterPlot,
    'sheet_profile': SheetProfilePlot,
    'integrity_panel': IntegrityPanelPlot,
}

#: Every `PlotBase` field name, including `plot_type`.
_PLOT_BASE_NAMES = frozenset(PlotBase.model_fields.keys())

#: The editable ones: `plot_type` is set by the type selector.
_BASE_FIELDS = tuple(k for k in _PLOT_BASE_NAMES if k != 'plot_type')


@dataclass
class PlotEditorDocument:
    """One plot spec, editable through the shared form widgets."""

    values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    plot_type: str = 'summary'
    dirty: bool = False

    #: Duck-types ConfigDocument for the shared form widgets; also selects the
    #: plot override table so the two sections get labels and help.
    model: str = PLOT_MODEL

    # ------------------------------------------------------------------ load

    @classmethod
    def blank(cls, plot_type: str = 'summary') -> 'PlotEditorDocument':
        if plot_type not in PLOT_TYPES:
            raise ValueError(f"unknown plot_type {plot_type!r}; "
                             f"known: {PLOT_TYPES}")
        return cls(values={COMMON_SECTION: {}, plot_type: {}},
                   plot_type=plot_type)

    @classmethod
    def from_plot(cls, plot: PlotBase) -> 'PlotEditorDocument':
        """Split a validated plot model into the two editable sections."""
        data = plot.model_dump(exclude_unset=True)
        data.pop('plot_type', None)
        common = {k: v for k, v in data.items() if k in _BASE_FIELDS}
        specific = {k: v for k, v in data.items() if k not in _BASE_FIELDS}
        return cls(values={COMMON_SECTION: common, plot.plot_type: specific},
                   plot_type=plot.plot_type)

    # ---------------------------------------------------------------- schema

    def sections(self) -> List[SectionSpec]:
        """`[common, <plot_type>]`, described straight off the models."""
        model = MODEL_BY_PLOT_TYPE[self.plot_type]
        common = describe_section(model, name='Common', key=COMMON_SECTION,
                                  optional=False)
        common.fields = [f for f in common.fields if f.name in _BASE_FIELDS]
        specific = describe_section(model, name=self.plot_type,
                                    key=self.plot_type, optional=False)
        specific.fields = [f for f in specific.fields
                           if f.name not in _PLOT_BASE_NAMES]
        return [common, specific]

    def section(self, key: str) -> Optional[SectionSpec]:
        return next((s for s in self.sections() if s.key == key), None)

    def set_plot_type(self, plot_type: str) -> None:
        """Switch type, keeping the common values and dropping the old extras."""
        if plot_type not in PLOT_TYPES:
            raise ValueError(f"unknown plot_type {plot_type!r}")
        if plot_type == self.plot_type:
            return
        common = self.values.get(COMMON_SECTION, {})
        self.values = {COMMON_SECTION: common, plot_type: {}}
        self.plot_type = plot_type
        self.dirty = True

    # ---------------------------------------------------------------- values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        current = self.values.setdefault(section, {})
        if key in current and current[key] == value:
            return
        current[key] = value
        self.dirty = True

    def has_section(self, key: str) -> bool:
        return key in (COMMON_SECTION, self.plot_type)

    def enable_section(self, key: str, enabled: bool) -> None:
        """Both plot sections are always present; nothing to toggle."""
        if enabled and key not in self.values:
            self.values[key] = {}

    def is_section_locked(self, key: str) -> bool:  # pylint: disable=unused-argument
        return True

    def section_lock_reason(self, key: str) -> Optional[str]:  # pylint: disable=unused-argument
        return 'Every plot group is always present.'

    # ------------------------------------------------------------ validation

    def validate(self):
        """Validate against the concrete plot model, reporting per field."""
        from ..config_validation import (ConfigIssue,  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
                                         ConfigValidation)

        try:
            plot = self.to_plot()
        except ValidationError as exc:
            issues = []
            for error in exc.errors():
                loc = error.get('loc', ())
                key = str(loc[-1]) if loc else None
                section = (COMMON_SECTION if key in _BASE_FIELDS
                           else self.plot_type)
                issues.append(ConfigIssue(
                    message=error.get('msg', 'invalid value'),
                    section=section, key=key, kind='validation'))
            return ConfigValidation(issues=issues)
        return ConfigValidation(issues=[], config=plot)

    def to_plot(self) -> PlotBase:
        """The validated plot model this document describes."""
        data: Dict[str, Any] = {'plot_type': self.plot_type}
        for section in (COMMON_SECTION, self.plot_type):
            data.update(copy.deepcopy(self.values.get(section, {})))
        return MODEL_BY_PLOT_TYPE[self.plot_type].model_validate(data)
