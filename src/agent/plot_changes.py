"""Plot proposals: one plot for the Plots tab, and the datasets it needs.

A `PlotChange` is checked when it is built -- the plot against the plot
schema (`plot_config.AnyPlot`), each dataset as a dataset folder inside the
workspace -- and applied by the GUI on its own thread with `apply_plot_change`,
which only edits a plot-document dict: nothing here writes a file.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import TypeAdapter, ValidationError

from ..gui.datasets import is_dataset
from ..postprocessing import plot_config
from .changes import ToolInputError, _capped, _clean_error, _new_id
from .sandbox import SandboxError, SearchRoots, display_path, resolve_readable, within_roots

_PLOT = TypeAdapter(plot_config.AnyPlot)


@dataclass
class PlotChange:  # pylint: disable=too-many-instance-attributes
    """A plot to append to the Plots tab's document."""

    id: str
    title: str
    #: The validated plot, as `model_dump(exclude_unset=True)` gives it.
    plot: Dict[str, Any]
    #: Dataset folders to add first, relative to the workspace (posix).
    add_datasets: List[str] = field(default_factory=list)
    #: The label of every dataset the plot reads, in request order.
    labels: List[str] = field(default_factory=list)
    #: How many of those were already in the Plots document.
    already_loaded: int = 0
    notes: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    #: The Plots document as it will be once applied (for a direct-apply turn).
    plot_preview: Optional[dict] = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """No issue found while checking it."""
        return not self.issues

    @property
    def plot_type(self) -> str:
        """The plot's type."""
        return str(self.plot.get('plot_type', ''))

    def handle(self, direct: bool = False) -> str:
        """What the model sees: short, and explicit about whether it is applied."""
        state = 'valid' if self.ok else (
            f'{len(self.issues)} issue(s): ' + '; '.join(self.issues[:3]))
        loaded = (f'; {self.already_loaded} dataset(s) already loaded'
                  if self.already_loaded else '')
        head = (f"plot {self.id} proposed ('{self.title}'): {self.plot_type} on "
                f"{', '.join(self.labels) or 'the loaded datasets'}; adds "
                f"{len(self.add_datasets)} dataset(s){loaded}; {state}.")
        if direct:
            return _capped(head, 'It will be added to the Plots tab unless it is refused; '
                                 'the user sees it and can undo it.')
        return _capped(head, 'Proposed only: the user sees it as a card and decides '
                             'whether to add it; do not say it is applied.')

    def summary(self) -> str:
        """The chat line once applied."""
        return f"Added plot: {self.plot_type} on {', '.join(self.labels) or 'the loaded datasets'}"

    def describe(self) -> str:
        """Plain-text body of the proposal card."""
        lines = [self.title, f'Plot: {self.plot_type}']
        lines += [f'  {key}: {value!r}' for key, value in self.plot.items()
                  if key != 'plot_type']
        if self.add_datasets:
            lines += ['Adds datasets:'] + [f'  - workspace:/{d}' for d in self.add_datasets]
        if self.notes:
            lines += ['Notes:'] + [f'  - {n}' for n in self.notes]
        lines.append('Validation: OK' if self.ok else
                     'Issues:\n' + '\n'.join(f'  - {i}' for i in self.issues))
        return '\n'.join(lines)


def _plot_model(plot_type: str):
    if plot_type not in plot_config.PLOT_TYPES:
        raise ToolInputError(f'unknown plot type {plot_type!r}; one of: '
                             + ', '.join(plot_config.PLOT_TYPES))
    return type(_PLOT.validate_python({'plot_type': plot_type}))


def _check_fields(plot_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """``fields`` without ``plot_type``; an unknown name lists the valid ones."""
    fields = dict(fields or {})
    given = fields.pop('plot_type', plot_type)
    if given != plot_type:
        raise ToolInputError(f'fields.plot_type {given!r} contradicts plot_type {plot_type!r}')
    valid = [name for name in _plot_model(plot_type).model_fields if name != 'plot_type']
    unknown = sorted(set(fields) - set(valid))
    if unknown:
        raise ToolInputError(f"{plot_type} has no field {', '.join(unknown)}; valid fields: "
                             + ', '.join(valid))
    return fields


def _norm(path) -> str:
    """``path`` absolute and normalised (lexically, as the sandbox compares)."""
    return os.path.normpath(os.path.abspath(str(path)))


def _real(path: str, workspace: Path) -> str:
    """A ``data_dirs`` entry, relative ones taken from the workspace, normalised."""
    real = Path(path).expanduser()
    return _norm(real if real.is_absolute() else workspace / real)


def resolve_dataset(path: str, roots: SearchRoots) -> Path:
    """A dataset folder inside the workspace, or a refusal saying why not."""
    workspace = SearchRoots(Path(roots.workspace))
    real = resolve_readable(path, roots)
    if not within_roots(real, workspace):
        raise SandboxError(f'{path}: not in the workspace; plots read datasets from the '
                           'workspace only')
    if not real.is_dir() or not is_dataset(real):
        raise ToolInputError(f'{path}: not a dataset (needs outputs/output_full_*.json); '
                             'use a folder find_result_sets lists as a dataset')
    return real


def _loaded(document: Optional[dict], workspace: Path) -> Dict[str, str]:
    """``{normalised data_dir: label}`` of a plot-document dict."""
    document = document or {}
    dirs, labels = document.get('data_dirs') or [], document.get('labels') or []
    return {_real(d, workspace): label for d, label in zip(dirs, labels)}


def build_plot_change(deps, title: str, plot_type: str, fields: Dict[str, Any],
                      datasets: Sequence[str]) -> PlotChange:
    """Check one plot and its datasets against ``deps``; refuse bad input."""
    workspace = Path(deps.roots.workspace)
    fields = _check_fields(plot_type, fields)
    loaded = _loaded(deps.plot_document, workspace)
    labels, add, already = [], [], 0
    for path in datasets:
        real = resolve_dataset(path, deps.roots)
        key = _norm(real)
        if key in loaded:
            already += 1
            label = loaded[key]
        else:
            relative = Path(key).relative_to(_norm(workspace)).as_posix()
            if relative not in add:
                add.append(relative)
            label = real.name
        if label not in labels:
            labels.append(label)
    if not datasets and not loaded:
        raise ToolInputError('name at least one dataset (find_result_sets lists them); '
                             'the Plots tab has none loaded')
    data = {'plot_type': plot_type, **fields}
    if labels and 'datasets' not in fields:
        data['datasets'] = labels
    try:
        plot = _PLOT.validate_python(data)
    except ValidationError as exc:
        raise ToolInputError(_clean_error(exc)) from exc
    change = PlotChange(id=_new_id(), title=title or f'Add a {plot_type} plot',
                        plot=plot.model_dump(mode='json', exclude_unset=True),
                        add_datasets=add, labels=labels, already_loaded=already)
    known = set(loaded.values()) | {Path(d).name for d in add}
    change.issues = [f'datasets: {name!r} is not a dataset label in the Plots document'
                     for name in (fields.get('datasets') or []) if name not in known]
    change.notes = [f'adds {display_path(workspace / d, deps.roots)}' for d in add]
    change.plot_preview, _added = apply_plot_change(change, deps.plot_document, workspace,
                                                    recheck=False)
    return change


def apply_plot_change(change: PlotChange, document: Optional[dict], workspace,
                      recheck: bool = True) -> Tuple[dict, List[str]]:
    """``document`` (a plot-document dict) with the change's missing datasets
    added -- label = folder name -- and its plot appended; ``document`` itself
    is not touched. Returns the new dict and the labels added.

    ``recheck``: refuse a dataset that is no longer one (the GUI applies a
    change some time after it was checked)."""
    workspace = Path(workspace)
    new = copy.deepcopy(document) if document else {}
    new.setdefault('data_dirs', [])
    new.setdefault('labels', [])
    new.setdefault('plots', [])
    loaded = _loaded(new, workspace)
    added = []
    for relative in change.add_datasets:
        real = Path(_norm(workspace / relative))
        if str(real) in loaded:
            continue
        if recheck:
            resolve_dataset(str(real), SearchRoots(workspace))
        new['data_dirs'].append(str(real))
        new['labels'].append(real.name)
        loaded[str(real)] = real.name
        added.append(real.name)
    new['plots'].append(copy.deepcopy(change.plot))
    return new, added
