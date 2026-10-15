"""Setup tools: read the Builder's documents, propose field changes, build sweeps."""
from __future__ import annotations

import difflib
import math
import re
from typing import Any, List, Literal

from pydantic import BaseModel, Field

from ...gui.schema.overrides import override_for_section
from ...gui.schema.simulation_types import FAMILIES, MODES
from ..changes import Op, ToolInputError, build_changeset
from . import compact

#: Fields `explain_current` reports when set, in this order.
KEY_FIELDS = (('2D', 'x'), ('2D', 'y'), ('2D', 'layers'), ('2D', 'pot_type'),
              ('tip', 'r'), ('general', 'temp'), ('general', 'force'),
              ('general', 'pressure'), ('general', 'scan_speed'),
              ('general', 'scan_angle'))


class FieldChange(BaseModel):
    """One value to propose; find the section and key with find_parameter first."""

    scope: Literal['config', 'settings'] = 'config'
    section: str = Field(description='Section, e.g. general, 2D, tip')
    key: str = Field(description='Key within the section, e.g. scan_speed')
    value: Any


def list_modes(deps) -> str:  # pylint: disable=unused-argument
    """Families and modes, with the model name each builds."""
    lines = []
    for family in FAMILIES:
        lines.append(f'{family.label}: {family.description}')
        for mode in family.modes:
            lines.append(f'  - {mode.label} [{mode.model}]: {mode.description}')
    return compact('\n'.join(lines))


def _render_issue(issue) -> str:
    where = '.'.join(p for p in (issue.section, issue.key) if p)
    return f'{where}: {issue.message}' if where else issue.message


def explain_current(deps) -> str:
    """Summarise the Builder's config and its validation issues."""
    doc = deps.document
    family, mode = MODES.get(doc.model, (None, None))
    label = f' ({family.label} / {mode.label})' if family and mode else ''
    lines = [f'model: {doc.model}{label}']
    materials = doc.material_names() or [doc.get('2D', 'mat')]
    lines.append('materials: ' + (', '.join(str(m) for m in materials if m) or 'none'))
    for section, key in KEY_FIELDS:
        value = doc.get(section, key)
        if value not in (None, '', []):
            lines.append(f'{section}.{key} = {value}')
    issues = [_render_issue(i) for i in doc.validate(deps.settings).issues]
    lines.append('issues: none' if not issues else
                 'issues:\n' + '\n'.join(f'- {i}' for i in issues[:8]))
    try:
        lines.append(f'runs: {doc.expansion().render()}')
    except Exception as exc:  # pylint: disable=broad-except
        lines.append(f'runs: unavailable ({exc})')
    return compact('\n'.join(lines))


def _entries(deps):
    """``(scope, section, FieldSpec, FieldOverride|None, current value)`` for every field."""
    for scope, doc in (('config', deps.document), ('settings', deps.settings_document)):
        for spec in doc.sections():
            override = override_for_section(doc.model, spec.key)
            for spec_field in spec.fields:
                field_override = override.fields.get(spec_field.key) if override else None
                yield scope, spec.key, spec_field, field_override, doc.get(spec.key, spec_field.key)


def find_parameter(deps, query: str) -> str:
    """Search config and settings fields by name, label and help text."""
    tokens = re.findall(r'[a-z0-9]+', query.lower())
    scored = []
    for scope, section, spec_field, override, value in _entries(deps):
        label = (override.label if override and override.label else spec_field.name) or ''
        helptext = (override.help if override and override.help else spec_field.doc) or ''
        unit = override.unit if override and override.unit else ''
        haystack = ' '.join([section, spec_field.key, label, helptext, unit]).lower()
        score = sum(token in haystack for token in tokens)
        score += 2 * sum(token == spec_field.key.lower() for token in tokens)
        if score:
            unit_text = f' ({unit})' if unit else ''
            scored.append((-score, f'{scope}:{section}.{spec_field.key} = {value!r} '
                                   f'-- {label}{unit_text}: {helptext[:100]}'))
    if not scored:
        return 'no matching parameter'
    return compact('\n'.join(line for _, line in sorted(scored)[:8]))


def _split_key(section: str, key: str):
    if '.' in key:
        head, tail = key.split('.', 1)
        if not section or head == section:
            return head, tail
    return section, key


def set_fields(deps, changes: List[FieldChange], title: str = 'Set fields') -> str:
    """Propose values for config or settings fields (shown as a card)."""
    ops = []
    for change in changes:
        section, key = _split_key(change.section, change.key)
        doc = deps.document if change.scope == 'config' else deps.settings_document
        spec = doc.section(section)
        if spec is None:
            raise ToolInputError(f"unknown {change.scope} section '{section}'; "
                                 'use find_parameter to locate the field')
        known = list(spec.field_map())
        if key not in known:
            close = difflib.get_close_matches(key, known, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ''
            raise ToolInputError(f"'{section}' has no key '{key}'{hint}")
        if change.scope == 'config':
            if not doc.has_section(section):
                ops.append(Op('enable_section', section, value=True))
            ops.append(Op('set', section, key, change.value))
        else:
            ops.append(Op('settings_set', section, key, change.value))
    return deps.propose(build_changeset(deps, title, ops))


def make_sweep(start: float, stop: float, n: int,
               spacing: Literal['linear', 'log'] = 'linear') -> List[float]:
    """``n`` values from ``start`` to ``stop``, linear or logarithmic."""
    if n < 1:
        raise ToolInputError('n must be at least 1')
    if n == 1:
        return [float(start)]
    if spacing == 'log':
        if start <= 0 or stop <= 0:
            raise ToolInputError('log spacing needs positive start and stop')
        a, b = math.log10(start), math.log10(stop)
        return [round(10 ** (a + i * (b - a) / (n - 1)), 6) for i in range(n)]
    return [round(start + i * (stop - start) / (n - 1), 6) for i in range(n)]


def make_sweep_tool(deps, start: float, stop: float, n: int,  # pylint: disable=unused-argument
                    spacing: Literal['linear', 'log'] = 'linear') -> str:
    """The values as text, for the model to pass on to a setup tool."""
    return 'values: ' + ', '.join(f'{v:g}' for v in make_sweep(start, stop, n, spacing))


def suggest_values(deps, quantity: str, descriptor: Literal['low', 'mid', 'high'],  # pylint: disable=unused-argument
                   n: int, spacing: Literal['linear', 'log'] = 'linear') -> str:  # pylint: disable=unused-argument
    """Descriptor -> values. No scaling rules exist yet (spec §4.3), so it says to ask."""
    return (f"No scaling rule for '{descriptor} {quantity}' yet: what counts as "
            f"{descriptor} depends on tip size, model size and speed; ask the user "
            f"for a range (min, max and unit), then call make_sweep with n={n}.")
