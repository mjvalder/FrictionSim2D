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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

from ..gui.config_validation import PATH_KEYS
from ..gui.controllers.document import DocumentError
from .sandbox import (SandboxError, check_write_target, display_path,
                      resolve_readable, within_roots)

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

    @property
    def ok(self) -> bool:
        """Valid as previewed: no validation issues and no blocked side effect."""
        return not self.issues

    def handle(self) -> str:
        """What the model sees: short, and explicit that nothing is applied yet."""
        state = 'valid' if self.ok else (
            f'{len(self.issues)} issue(s): ' + '; '.join(self.issues[:3]))
        runs = f'; runs: {self.expansion_text}' if self.expansion_text else ''
        return (f"change {self.id} proposed ('{self.title}'): {len(self.diff)} "
                f"field(s), {state}{runs}. The user sees it as a card and decides "
                f"whether to apply it; do not say it is applied.")

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

    @property
    def ok(self) -> bool:
        """No refused operation."""
        return not self.issues

    def handle(self) -> str:
        """What the model sees."""
        state = 'valid' if self.ok else 'refused: ' + '; '.join(self.issues[:3])
        return (f"change {self.id} proposed ('{self.title}'): {len(self.ops)} file "
                f"operation(s), {state}. The user decides whether to apply it; do "
                f"not say it is done.")


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


def _check_proposed_paths(ops: Sequence[Op], base_dir, roots) -> None:
    """Refuse a path-valued field the agent proposes outside the readable roots."""
    candidates = []
    for op in ops:
        if op.kind == 'set' and op.key in PATH_KEYS and isinstance(op.value, str) and op.value:
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


def _diff(scope: str, before: dict, after: dict) -> List[FieldDiff]:
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
    _check_proposed_paths(ops, live.base_dir, deps.roots)
    document = copy.deepcopy(live)
    settings_document = copy.deepcopy(deps.settings_document)
    change = ChangeSet(id=_new_id(), title=title, ops=list(ops), notes=list(notes),
                       base_fingerprint=fingerprint(live, deps.settings_document))
    scratch_root = Path(deps.roots.workspace) / '.preview' / '_agent'
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='agent-', dir=scratch_root))
    try:
        problems = _replay(ops, document, settings_document, scratch / 'materials.txt')
        settings = deps.settings
        if change.touches_settings():
            try:
                settings = settings_document.as_settings()
            except Exception as exc:  # pylint: disable=broad-except
                problems.append(f'settings: {exc}')
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
        shutil.rmtree(scratch, ignore_errors=True)

    if live.model != document.model:
        change.diff.append(FieldDiff('config', '(model)', '', live.model, document.model))
    change.diff += _diff('config', live.values, document.values)
    change.diff += _diff('settings', deps.settings_document.values, settings_document.values)
    for op in ops:
        if op.kind == 'materials' and op.value:
            target = live.material_list_target()
            if target is None:
                change.issues.append('material list needs somewhere to go: save the config first')
            else:
                change.side_effects.append(
                    f'writes {display_path(target, deps.roots)} ({len(op.value)} names)')
    return change


def apply_changeset(change: ChangeSet, document, settings_document, *,
                    force: bool = False) -> ApplyResult:
    """Replay ``change`` on the live documents, unless they changed since the preview."""
    if not force and fingerprint(document, settings_document) != change.base_fingerprint:
        return ApplyResult(False, True, 'The Builder changed since this was proposed.')
    try:
        problems = _replay(change.ops, document, settings_document, None)
    except DocumentError as exc:
        return ApplyResult(False, False, f'Not applied: {exc}')
    message = f'Applied: {change.title}'
    if problems:
        message += ' (' + '; '.join(problems) + ')'
    return ApplyResult(True, False, message)


def build_fs_changeset(deps, title: str, ops: Sequence[Tuple[str, ...]]) -> FsChangeSet:
    """Check folder/copy operations; refused ones are listed as issues."""
    workspace = Path(deps.roots.workspace)
    checked, issues = [], []
    planned = set()
    for op in ops:
        try:
            if op[0] == 'mkdir':
                target = Path(op[1])
                target = target if target.is_absolute() else workspace / target
                if target.is_dir():
                    continue
                checked.append(('mkdir', str(check_write_target(target, workspace))))
                planned.add(str(target))
            elif op[0] == 'copy':
                source = resolve_readable(op[1], deps.roots)
                if not source.is_file():
                    raise SandboxError(f'{op[1]}: not a file')
                checked.append(('copy', str(source), str(check_write_target(op[2], workspace))))
            else:
                raise SandboxError(f'unknown operation {op[0]!r}')
        except SandboxError as exc:
            issues.append(str(exc))
    return FsChangeSet(id=_new_id(), title=title, ops=checked, issues=issues)


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
                shutil.copy2(op[1], target)
        except (SandboxError, OSError) as exc:
            finished = f" after {', '.join(done)}" if done else ''
            return ApplyResult(False, False, f'Stopped{finished}: {exc}')
        done.append(f'{op[0]} {Path(op[-1]).name}')
    return ApplyResult(True, False, f'Applied: {change.title}')


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + '...'


def describe_change(change) -> str:
    """Plain-text body of a proposal card."""
    lines = [change.title]
    if isinstance(change, FsChangeSet):
        lines += [f'  {op[0]}: ' + ' -> '.join(Path(p).name if i else p
                                               for i, p in enumerate(op[1:]))
                  for op in change.ops]
    else:
        lines.append('Changes:')
        for d in change.diff:
            where = f'[{d.section}] {d.key}'.strip() if d.key else d.section
            lines.append(f'  {d.scope} {where}: {_short(d.before)} -> {_short(d.after)}')
        if change.side_effects:
            lines += ['Writes:'] + [f'  - {s}' for s in change.side_effects]
        if change.notes:
            lines += ['Notes:'] + [f'  - {n}' for n in change.notes]
        if change.expansion_text:
            lines.append(f'Runs: {change.expansion_text}')
    lines.append('Validation: OK' if change.ok else
                 'Issues:\n' + '\n'.join(f'  - {i}' for i in change.issues))
    return '\n'.join(lines)
