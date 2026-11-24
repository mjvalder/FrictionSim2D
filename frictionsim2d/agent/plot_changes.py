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
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import TypeAdapter, ValidationError

from ..gui.datasets import is_dataset
from ..postprocessing import plot_config
from .changes import ToolInputError
from .tools import capped, clean_error, new_id
from .sandbox import SandboxError, SearchRoots, display_path, resolve_readable, within_roots

_PLOT = TypeAdapter(plot_config.AnyPlot)


@dataclass
class PlotChange:  # pylint: disable=too-many-instance-attributes
    """A plot to append to the Plots tab's document."""

    id: str
    title: str
    #: The validated plot, as `model_dump(exclude_unset=True)` gives it.
    plot: Dict[str, Any]
    #: Every dataset folder the plot reads (relative to the workspace, posix)
    #: with its label as previewed; Apply maps each to the label it really has.
    folders: List[Tuple[str, str]] = field(default_factory=list)
    #: Those of ``folders`` the Plots document lacked when it was checked.
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
            return capped(head, 'It will be added to the Plots tab unless it is refused; '
                                 'the user sees it and can undo it.')
        return capped(head, 'Proposed only: the user sees it as a card and decides '
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


def _identity(path: str, workspace: Path) -> str:
    """Which folder a ``data_dirs`` entry is: symlinks resolved, so a workspace
    reached through a link matches entries stored in either spelling."""
    return os.path.realpath(_real(path, workspace))


def _relative(real: Path, workspace: Path) -> str:
    """``real`` relative to the workspace (posix), lexically or else resolved."""
    for path, root in ((_norm(real), _norm(workspace)),
                       (os.path.realpath(real), os.path.realpath(workspace))):
        try:
            return Path(path).relative_to(root).as_posix()
        except ValueError:
            continue
    raise ToolInputError(f'{real.name}: not inside the workspace')


def _free_label(folder: str, hint: Optional[str], taken: Sequence[str]) -> str:
    """A label no other dataset has: ``hint``, the folder name, ``parent/name``,
    then ``name-2``, ``name-3``, ..."""
    parts = PurePosixPath(folder).parts
    name = parts[-1] if parts else folder
    candidates = [hint, name, '/'.join(parts[-2:]) if len(parts) > 1 else None]
    for candidate in candidates:
        if candidate and candidate not in taken:
            return candidate
    number = 2
    while f'{name}-{number}' in taken:
        number += 1
    return f'{name}-{number}'


def _place(document: Optional[dict], folders: Sequence[Tuple[str, Optional[str]]],
           workspace: Path, recheck: bool) -> Tuple[dict, List[str], List[str]]:
    """A copy of ``document`` holding every folder of ``folders`` (``(folder,
    label hint)``): the label of each, in order, and the labels added."""
    new = copy.deepcopy(document) if document else {}
    for key in ('data_dirs', 'labels', 'plots'):
        new.setdefault(key, [])
    index = {_identity(d, workspace): i for i, d in enumerate(new['data_dirs'])}
    labels, added = [], []
    for folder, hint in folders:
        real = _norm(workspace / folder)
        found = index.get(_identity(real, workspace))
        if found is not None and found < len(new['labels']):
            labels.append(new['labels'][found])
            continue
        if recheck:
            resolve_dataset(real, SearchRoots(workspace))
        label = _free_label(folder, hint, new['labels'])
        index[_identity(real, workspace)] = len(new['data_dirs'])
        new['data_dirs'].append(real)
        new['labels'].append(label)
        labels.append(label)
        added.append(label)
    return new, labels, added


def _check_labels(plot: dict, labels: Sequence[str]) -> None:
    missing = [name for name in plot.get('datasets') or [] if name not in labels]
    if missing:
        raise ToolInputError(f"datasets {', '.join(map(repr, missing))} not in the Plots "
                             f"document (labels: {', '.join(labels) or 'none'})")


def build_plot_change(deps, title: str, plot_type: str, fields: Dict[str, Any],
                      datasets: Sequence[str]) -> PlotChange:
    """Check one plot and its datasets against ``deps``; refuse bad input."""
    workspace = Path(deps.roots.workspace)
    fields = _check_fields(plot_type, fields)
    folders = []
    for path in datasets:
        folder = _relative(resolve_dataset(path, deps.roots), workspace)
        if folder not in folders:
            folders.append(folder)
    if not folders and not (deps.plot_document or {}).get('data_dirs'):
        raise ToolInputError('name at least one dataset (find_result_sets lists them); '
                             'the Plots tab has none loaded')
    preview, labels, added = _place(deps.plot_document, [(f, None) for f in folders],
                                    workspace, recheck=False)
    data = {'plot_type': plot_type, **fields}
    if labels and 'datasets' not in fields:
        data['datasets'] = labels
    try:
        plot = _PLOT.validate_python(data).model_dump(mode='json', exclude_unset=True)
    except ValidationError as exc:
        raise ToolInputError(clean_error(exc)) from exc
    _check_labels(plot, preview['labels'])
    add = [f for f, label in zip(folders, labels) if label in added]
    change = PlotChange(id=new_id(), title=title or f'Add a {plot_type} plot', plot=plot,
                        folders=list(zip(folders, labels)), add_datasets=add,
                        labels=labels, already_loaded=len(folders) - len(add))
    change.notes = [f'adds {display_path(workspace / d, deps.roots)}' for d in add]
    preview['plots'].append(copy.deepcopy(plot))
    change.plot_preview = preview
    return change


def apply_plot_change(change: PlotChange, document: Optional[dict], workspace,
                      recheck: bool = True) -> Tuple[dict, List[str]]:
    """``document`` (a plot-document dict) with every dataset the change reads
    -- those missing added under a label no other dataset has -- and its plot
    appended, its ``datasets`` naming the labels the document really uses;
    ``document`` itself is not touched. Returns the new dict and the labels added.

    ``recheck``: refuse a dataset that is no longer one (the GUI applies a
    change some time after it was checked). A label the plot names that the
    document no longer has refuses the whole apply (`ToolInputError`)."""
    workspace = Path(workspace)
    new, labels, added = _place(document, change.folders, workspace, recheck)
    renamed = {hint: label for (_folder, hint), label in zip(change.folders, labels)}
    plot = copy.deepcopy(change.plot)
    if plot.get('datasets'):
        plot['datasets'] = [renamed.get(name, name) for name in plot['datasets']]
    _check_labels(plot, new['labels'])
    new['plots'].append(plot)
    return new, added
