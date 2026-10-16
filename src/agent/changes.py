"""Change sets: what the assistant proposes, previewed on copies, applied on click.

A `ChangeSet` is an ordered list of the *same calls the Builder form makes*
(`ConfigDocument.set`, `set_model`, `write_material_names`, ...). Building one
replays it on deep copies and validates the result; applying replays it on the
live documents. Nothing here writes except `apply_changeset` (the material
list, via `write_material_names`) and `apply_fs_changeset`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

from ..gui.config_validation import PATH_KEYS
from ..gui.controllers.document import DocumentError
from .sandbox import (PathLike, SandboxError, check_write_target, display_path, from_display,
                      resolve_readable, within_roots, write_path)
from .tools import compact

#: Most characters a change handle (a tool result) may take.
HANDLE_LIMIT = 600

OpKind = Literal['set', 'set_model', 'enable_section', 'material_dirs',
                 'materials', 'settings_set']


class ToolInputError(ValueError):
    """Bad tool arguments; the message goes back to the model as a retry hint."""


@dataclass(frozen=True)
class Op:
    """One document operation, exactly as the form would perform it."""

    kind: OpKind
    section: str = ''
    key: str = ''
    value: Any = None


@dataclass(frozen=True)
class FieldDiff:
    """One value that differs between the current and the proposed documents."""

    scope: str
    section: str
    key: str
    before: Any
    after: Any


@dataclass(frozen=True)
class ApplyResult:
    """What happened when the user clicked Apply."""

    applied: bool
    stale: bool
    message: str
    #: Not applied, but the live documents may hold some of the operations
    #: (a file error part-way through); the caller should offer to undo.
    partial: bool = False


@dataclass
class ChangeSet:
    """A proposal against the Builder's config and settings documents."""

    id: str
    title: str
    ops: List[Op]
    notes: List[str]
    base_fingerprint: str
    diff: List[FieldDiff] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    expansion: Any = None
    expansion_text: str = ''
    #: ``(document, settings_document)`` as they will be once applied: the
    #: copies the preview ran on. Lets a direct-apply turn keep its snapshot
    #: in step with the GUI (`AgentDeps.propose`).
    preview: Any = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """Valid as previewed: no validation issues and no blocked side effect."""
        return not self.issues

    def handle(self, direct: bool = False) -> str:
        """What the model sees: short, and explicit about whether it is applied.

        ``direct``: the GUI applies the change as it arrives (the user can
        undo it); otherwise it waits on a card for the user.
        """
        state = 'valid' if self.ok else (
            f'{len(self.issues)} issue(s): ' + '; '.join(self.issues[:3]))
        runs = f'; runs: {self.expansion_text}' if self.expansion_text else ''
        head = (f"change {self.id} proposed ('{self.title}'): {len(self.diff)} "
                f"field(s), {state}{runs}.")
        if direct:
            return _capped(head, "It will be applied unless it is refused; the user "
                                 "sees the outcome and can undo it.")
        return _capped(head, "Proposed only: the user sees it as a card and decides "
                             "whether to apply it; do not say it is applied.")

    def changes_model(self, current: str) -> bool:
        """Whether applying this switches the simulation model away from ``current``."""
        return any(op.kind == 'set_model' and op.value != current for op in self.ops)

    def touches_settings(self) -> bool:
        """Whether applying this edits the settings document."""
        return any(op.kind == 'settings_set' for op in self.ops)


@dataclass
class FsChangeSet:
    """A proposal to create folders or copy found files into the workspace."""

    id: str
    title: str
    ops: List[Tuple[str, ...]]
    issues: List[str] = field(default_factory=list)
    #: ``ops`` with every path as ``workspace:/...`` / ``examples:/...``, for the card.
    shown: List[Tuple[str, ...]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No refused operation."""
        return not self.issues

    def handle(self, direct: bool = False) -> str:  # pylint: disable=unused-argument
        """What the model sees. File operations always wait for the user, so
        ``direct`` (the config apply mode) does not change it."""
        state = 'valid' if self.ok else 'refused: ' + '; '.join(self.issues[:3])
        return _capped(f"change {self.id} proposed ('{self.title}'): {len(self.ops)} "
                       f"file operation(s), {state}.",
                       "The user decides whether to apply it; do not say it is done.")


def _capped(head: str, instruction: str) -> str:
    """``head`` cut so the whole handle fits `HANDLE_LIMIT`; ``instruction`` kept whole."""
    return compact(head, HANDLE_LIMIT - len(instruction) - 1) + ' ' + instruction


def _clean_error(exc: Exception) -> str:
    """An exception's text without pydantic's "For further information visit" lines."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return '\n'.join(line for line in lines
                     if not line.startswith('For further information'))


def fingerprint(document, settings_document) -> str:
    """Hash of the model and both documents' values, to detect edits since a proposal."""
    payload = json.dumps([document.model, document.values, settings_document.values],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _new_id() -> str:
    return uuid.uuid4().hex[:6]


def _replay(ops: Sequence[Op], document, settings_document,
            materials_target: Optional[Path]) -> List[str]:
    """Perform ``ops``; return problems that are not exceptions."""
    problems = []
    for op in ops:
        if op.kind == 'set_model':
            document.set_model(op.value)
        elif op.kind == 'set':
            document.set(op.section, op.key, op.value)
        elif op.kind == 'enable_section':
            document.enable_section(op.section, bool(op.value))
        elif op.kind == 'material_dirs':
            structure_dir, potential_dir, extension = op.value
            document.apply_material_directories(structure_dir, potential_dir, extension)
        elif op.kind == 'materials':
            written = document.write_material_names(list(op.value), target=materials_target)
            if written is None and op.value:
                problems.append('material list not written: save the config first '
                                '(Builder > Save), then apply again')
        elif op.kind == 'settings_set':
            settings_document.set(op.section, op.key, op.value)
        else:
            raise ValueError(f'unknown operation {op.kind!r}')
    return problems


def _is_path_op(op: Op) -> bool:
    return (op.kind == 'set' and (op.key in PATH_KEYS or op.key == 'materials_list')
            and isinstance(op.value, str) and bool(op.value))


def _real_path(value: str, base_dir, roots) -> str:
    """``examples:/a`` / ``workspace:/a`` (what tools show) as a config value:
    relative to the config's folder when it can be; anything else unchanged."""
    real = from_display(value, roots)
    if real is None:
        return value
    if base_dir is not None:
        try:
            return str(real.relative_to(Path(base_dir)))
        except ValueError:
            pass
    return str(real)


def _undisplay(ops: Sequence[Op], base_dir, roots) -> List[Op]:
    """``ops`` with display-form path values mapped back to real paths."""
    out = []
    for op in ops:
        if _is_path_op(op):
            op = Op(op.kind, op.section, op.key, _real_path(op.value, base_dir, roots))
        elif op.kind == 'material_dirs':
            dirs = tuple(_real_path(str(v), base_dir, roots) for v in op.value[:2])
            op = Op(op.kind, op.section, op.key, (*dirs, *op.value[2:]))
        out.append(op)
    return out


def _check_proposed_paths(ops: Sequence[Op], base_dir, roots) -> None:
    """Refuse a path-valued field the agent proposes outside the readable roots."""
    candidates = []
    for op in ops:
        if _is_path_op(op):
            candidates.append(op.value)
        if op.kind == 'material_dirs':
            candidates.extend(str(v) for v in op.value[:2])
    for value in candidates:
        path = Path(value)
        if '{' in value:
            path = path.parent
        if not path.is_absolute():
            path = Path(base_dir or roots.workspace) / path
        if not within_roots(path, roots):
            raise ToolInputError(f'{value}: outside the workspace and bundled examples; '
                                 'use a path returned by search_structures/search_potentials')


def _has_line_break(value: Any) -> bool:
    if isinstance(value, str):
        return '\n' in value or '\r' in value
    if isinstance(value, (list, tuple)):
        return any(_has_line_break(v) for v in value)
    return False


def _check_proposed_values(ops: Sequence[Op]) -> None:
    """Refuse multi-line strings: config and settings values end up in LAMMPS
    scripts, where a line break starts a new command."""
    for op in ops:
        if op.kind in ('set', 'settings_set') and _has_line_break(op.value):
            raise ToolInputError(f'{op.section}.{op.key}: a value may not contain a line '
                                 'break')


def field_diff(scope: str, before: dict, after: dict) -> List[FieldDiff]:
    """Every value that differs between two ``{section: {key: value}}`` maps."""
    out = []
    for section in sorted(set(before) | set(after)):
        old, new = before.get(section) or {}, after.get(section) or {}
        if not isinstance(old, dict) or not isinstance(new, dict):
            if old != new:
                out.append(FieldDiff(scope, section, '', old, new))
            continue
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                out.append(FieldDiff(scope, section, key, old.get(key), new.get(key)))
    return out


def _render_issue(issue) -> str:
    where = '.'.join(p for p in (issue.section, issue.key) if p)
    return f'{where}: {issue.message}' if where else issue.message


def build_changeset(deps, title: str, ops: Sequence[Op],
                    notes: Sequence[str] = ()) -> ChangeSet:
    """Preview ``ops`` on copies of ``deps``' documents and package the result."""
    live = deps.document
    ops = _undisplay(ops, live.base_dir, deps.roots)
    _check_proposed_paths(ops, live.base_dir, deps.roots)
    _check_proposed_values(ops)
    document = copy.deepcopy(live)
    settings_document = copy.deepcopy(deps.settings_document)
    change = ChangeSet(id=_new_id(), title=title, ops=list(ops), notes=list(notes),
                       base_fingerprint=fingerprint(live, deps.settings_document))
    scratch = _scratch(Path(deps.roots.workspace), 'agent-', ops)
    try:
        problems = _replay(ops, document, settings_document,
                           scratch / 'materials.txt' if scratch else None)
        settings = deps.settings
        if change.touches_settings():
            try:
                settings = settings_document.as_settings()
            except Exception as exc:  # pylint: disable=broad-except
                problems.append('settings: ' + _clean_error(exc))
        validation = document.validate(settings)
        change.issues = [_render_issue(i) for i in validation.issues] + problems
        try:
            change.expansion = document.expansion()
            change.expansion_text = change.expansion.render()
        except Exception as exc:  # pylint: disable=broad-except
            change.expansion_text = f'unavailable ({exc})'
    except DocumentError as exc:
        raise ToolInputError(str(exc)) from exc
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    target = live.material_list_target()
    if target is not None and any(op.kind == 'materials' and op.value for op in ops):
        # The preview wrote to scratch; show what Apply will really set.
        document.set('2D', 'materials_list',
                     str(live._relative_to_config(target)))  # pylint: disable=protected-access
    change.preview = (document, settings_document)
    if live.model != document.model:
        change.diff.append(FieldDiff('config', '(model)', '', live.model, document.model))
    change.diff += field_diff('config', live.values, document.values)
    change.diff += field_diff('settings', deps.settings_document.values, settings_document.values)
    for op in ops:
        if op.kind == 'materials' and op.value:
            target = live.material_list_target()
            if target is None:
                change.issues.append('material list needs somewhere to go: save the config first')
            else:
                change.side_effects.append(
                    f'writes {display_path(target, deps.roots)} ({len(op.value)} names)')
    return change


def _scratch(root: Optional[Path], prefix: str, ops: Sequence[Op]) -> Optional[Path]:
    """A temporary folder under ``root/.preview/_agent`` for a preview's material
    list -- only when an operation writes one; None otherwise."""
    if not any(op.kind == 'materials' for op in ops):
        return None
    parent = Path(root) / '.preview' / '_agent' if root else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def _settings_problem(settings_document) -> str:
    """Why ``settings_document`` is not a valid `GlobalSettings`, or ''."""
    try:
        settings_document.as_settings()
    except Exception as exc:  # pylint: disable=broad-except
        lines = _clean_error(exc).splitlines()
        return '; '.join(lines[1:] or lines)[:300]
    return ''


def apply_changeset(change: ChangeSet, document, settings_document, *,
                    force: bool = False) -> ApplyResult:
    """Replay ``change`` on the live documents, unless they changed since the preview."""
    if not force and fingerprint(document, settings_document) != change.base_fingerprint:
        return ApplyResult(False, True, 'The Builder changed since this was proposed.')
    root = getattr(document, 'workspace_root', None) or document.base_dir
    live = False
    try:
        scratch = _scratch(root, 'apply-', change.ops)
        try:  # dry run on copies so a failing op cannot half-apply
            dry_settings = copy.deepcopy(settings_document)
            _replay(change.ops, copy.deepcopy(document), dry_settings,
                    scratch / 'materials.txt' if scratch else None)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
        if change.touches_settings():
            invalid = _settings_problem(dry_settings)
            if invalid:
                return ApplyResult(False, False, f'Not applied: settings invalid: {invalid}')
        live = True
        problems = _replay(change.ops, document, settings_document, None)
    except DocumentError as exc:
        return ApplyResult(False, False, f'Not applied: {exc}')
    except OSError as exc:
        if not live:
            return ApplyResult(False, False, f'Not applied (file error): {exc}')
        return ApplyResult(False, False, f'Not applied (file error, the Builder '
                                         f'may be partly changed): {exc}', partial=True)
    message = f'Applied: {change.title}'
    if problems:
        message += ' (' + '; '.join(problems) + ')'
    return ApplyResult(True, False, message)


def build_fs_changeset(deps, title: str, ops: Sequence[Tuple[str, ...]]) -> FsChangeSet:
    """Check folder/copy operations; refused ones are listed as issues."""
    workspace = Path(deps.roots.workspace)
    checked, issues = [], []
    for op in ops:
        try:
            if op[0] == 'mkdir':
                target = write_path(op[1], deps.roots)
                target = target if target.is_absolute() else workspace / target
                if target.is_dir():
                    continue
                checked.append(('mkdir', str(check_write_target(target, workspace))))
            elif op[0] == 'copy':
                source = resolve_readable(op[1], deps.roots)
                if not source.is_file():
                    raise SandboxError(f'{op[1]}: not a file')
                checked.append(('copy', str(source), str(check_write_target(
                    write_path(op[2], deps.roots), workspace))))
            else:
                raise SandboxError(f'unknown operation {op[0]!r}')
        except SandboxError as exc:
            issues.append(str(exc))
    shown = [(op[0], *(display_path(p, deps.roots) for p in op[1:])) for op in checked]
    return FsChangeSet(id=_new_id(), title=title, ops=checked, issues=issues, shown=shown)


def _copy_new(source: PathLike, target: PathLike) -> None:
    """Copy ``source`` to a *new* file: ``'xb'`` fails if ``target`` appeared
    since it was checked, so nothing is ever overwritten."""
    with open(source, 'rb') as src, open(target, 'xb') as dst:
        shutil.copyfileobj(src, dst)
    shutil.copystat(source, target)


def apply_fs_changeset(change: FsChangeSet, workspace_root) -> ApplyResult:
    """Run the operations in order, re-checking each; stop at the first refusal."""
    if not change.ok:
        return ApplyResult(False, False, 'Not applied: ' + '; '.join(change.issues))
    done = []
    for op in change.ops:
        try:
            if op[0] == 'mkdir':
                if not Path(op[1]).is_dir():
                    check_write_target(op[1], workspace_root).mkdir(parents=True)
            else:
                target = check_write_target(op[2], workspace_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_new(op[1], target)
        except (SandboxError, OSError) as exc:
            finished = f" after {', '.join(done)}" if done else ''
            return ApplyResult(False, False, f'Stopped{finished}: {exc}')
        done.append(f'{op[0]} {Path(op[-1]).name}')
    return ApplyResult(True, False, f'Applied: {change.title}')


SUMMARY_FIELDS = 4


def applied_summary(change: ChangeSet) -> str:
    """One chat line for an applied change set: which fields, and the model."""
    fields = [d for d in change.diff if d.section != '(model)']
    model = next((d.after for d in change.diff if d.section == '(model)'), None)
    names = [('settings ' if d.scope == 'settings' else '')
             + (f'[{d.section}] {d.key}' if d.key else f'[{d.section}]') for d in fields]
    shown = ', '.join(names[:SUMMARY_FIELDS]) + (', …' if len(names) > SUMMARY_FIELDS else '')
    noun = 'field' if len(fields) == 1 else 'fields'
    if not fields:
        return f'Switched model → {model}' if model else 'No field changed'
    text = f'Set {len(fields)} {noun}: {shown}'
    return f'{text} (+ model → {model})' if model else text


CARD_WIDTH = 100


def _wrapped(line: str, indent: str = '      ') -> List[str]:
    """``line`` broken into card-width pieces; nothing is ever cut off."""
    return textwrap.wrap(line, CARD_WIDTH, subsequent_indent=indent,
                         break_long_words=True, break_on_hyphens=False,
                         drop_whitespace=False) or ['']


def describe_change(change) -> str:
    """Plain-text body of a proposal card."""
    lines = [change.title]
    if isinstance(change, FsChangeSet):
        shown = change.shown or [(op[0], *(Path(p).name for p in op[1:]))
                                 for op in change.ops]
        lines += [f'  {op[0]}: ' + ' -> '.join(op[1:]) for op in shown]
    else:
        lines.append('Changes:')
        for d in change.diff:
            where = f'[{d.section}] {d.key}'.strip() if d.key else d.section
            lines += _wrapped(f'  {d.scope} {where}: {d.before!r} -> {d.after!r}')
        if change.side_effects:
            lines += ['Writes:'] + [f'  - {s}' for s in change.side_effects]
        if change.notes:
            lines += ['Notes:'] + [f'  - {n}' for n in change.notes]
        if change.expansion_text:
            lines.append(f'Runs: {change.expansion_text}')
    lines.append('Validation: OK' if change.ok else
                 'Issues:\n' + '\n'.join(f'  - {i}' for i in change.issues))
    return '\n'.join(lines)
