"""Setup tools: read the Builder's documents, propose field changes, build sweeps."""
from __future__ import annotations

import copy
import difflib
import math
import re
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat

from ...gui.schema.overrides import override_for_section
from ...gui.schema.simulation_types import FAMILIES, MODES
from ...gui.controllers.document import DocumentError
from ..changes import Op, ToolInputError, build_changeset
from ..sandbox import display_path
from . import compact
from .materials import config_path, material_ops, resolve_material

#: Fields `explain_current` reports when set, in this order.
KEY_FIELDS = (('2D', 'x'), ('2D', 'y'), ('2D', 'layers'), ('2D', 'pot_type'),
              ('tip', 'r'), ('general', 'temp'), ('general', 'force'),
              ('general', 'pressure'), ('general', 'scan_speed'),
              ('general', 'scan_angle'))


#: Most values `make_sweep` produces: every value becomes at least one run.
MAX_SWEEP = 200

#: Settings the assistant may not propose, because they are executed rather
#: than read: LAMMPS runs the ``*_command`` strings verbatim, and the whole
#: ``hpc`` section (modules, conda_env, mpi_command, *_dir, ...) is written into
#: job scripts the scheduler runs. The rule, for keys added later: the ``hpc``
#: section, any key ending in ``_command`` or ``_dir``, ``modules``, ``conda_env``
#: (pinned against GlobalSettings by tests/agent/test_tools_setup.py).
REFUSED_SETTINGS = frozenset({
    ('simulation', 'minimization_command'), ('simulation', 'neigh_modify_command'),
    ('hpc', 'scheduler_type'), ('hpc', 'queue'), ('hpc', 'partition'), ('hpc', 'account'),
    ('hpc', 'hpc_host'), ('hpc', 'hpc_home'), ('hpc', 'log_dir'), ('hpc', 'scratch_dir'),
    ('hpc', 'num_nodes'), ('hpc', 'num_cpus'), ('hpc', 'memory_gb'),
    ('hpc', 'walltime_hours'), ('hpc', 'max_array_size'), ('hpc', 'modules'),
    ('hpc', 'conda_env'), ('hpc', 'mpi_command'), ('hpc', 'use_tmpdir'),
    ('hpc', 'preflight'), ('hpc', 'lammps_scripts'),
})


def _refused_setting(section: str, key: str) -> bool:
    return ((section, key) in REFUSED_SETTINGS or section == 'hpc'
            or key.endswith(('_command', '_dir')) or key in ('modules', 'conda_env'))


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
            if _refused_setting(section, key):
                raise ToolInputError(
                    f"{section}.{key}: this setting is executed by LAMMPS/the HPC "
                    "scheduler; change it in the Settings tab yourself")
            ops.append(Op('settings_set', section, key, change.value))
    return deps.propose(build_changeset(deps, title, ops))


def make_sweep(start: float, stop: float, n: int,
               spacing: Literal['linear', 'log'] = 'linear') -> List[float]:
    """``n`` values from ``start`` to ``stop``, linear or logarithmic."""
    if n < 1:
        raise ToolInputError('n must be at least 1')
    if n > MAX_SWEEP:
        raise ToolInputError(f'n must be at most {MAX_SWEEP}; each value is a run')
    if not (math.isfinite(start) and math.isfinite(stop)):
        raise ToolInputError('start and stop must be finite numbers')
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


class _SetupBase(BaseModel):
    materials: List[str] = Field(min_length=1, description='Material names exactly as '
                                 'search_structures returns them, e.g. h-MoS2')
    potential_type: Optional[str] = Field(
        None, description="e.g. 'sw', 'tersoff', 'rebomos', 'reaxff'")
    size_A: Optional[Tuple[PositiveFloat, PositiveFloat]] = Field(
        None, description='Sheet x, y in Angstrom')
    layers: Optional[List[int]] = Field(None, description='Layer counts to sweep, e.g. [1, 2, 3]')


_TIP = Field('Si', description='Tip material, a structure name as search_structures '
                                'returns it; the default Si tip is usually right')
_SUB = Field('Si', description='Substrate material under the sheet (built amorphous); '
                               'the default amorphous Si is usually right')
_TIP_POT = Field('sw', description='Potential type for the tip and substrate')


class AFMSetupRequest(_SetupBase):
    """AFM tip sliding on a sheet (model afm)."""

    tip_material: str = _TIP
    substrate_material: str = _SUB
    tip_potential_type: str = _TIP_POT

    loads_nN: List[PositiveFloat] = Field(default_factory=list, description='Normal loads in nN')
    speed_m_s: Optional[PositiveFloat] = Field(None, description='Sliding speed in m/s')
    tip_radius_A: Optional[PositiveFloat] = Field(None, description='Tip radius in Angstrom')
    temperature_K: Optional[NonNegativeFloat] = None
    scan_angles_deg: Optional[List[float]] = None


class SheetSetupRequest(_SetupBase):
    """Sheet sliding on sheet (model sheetonsheet)."""

    pressures_GPa: List[float] = Field(default_factory=list, description='Normal pressures in GPa')
    speed_m_s: Optional[PositiveFloat] = None
    temperature_K: Optional[NonNegativeFloat] = None
    scan_angles_deg: Optional[List[float]] = None


class IndentSetupRequest(_SetupBase):
    """Approach, hold and retract without sliding (model indent)."""

    tip_material: str = _TIP
    substrate_material: str = _SUB
    tip_potential_type: str = _TIP_POT

    loads_nN: List[PositiveFloat] = Field(default_factory=list)
    tip_radius_A: Optional[PositiveFloat] = None
    temperature_K: Optional[NonNegativeFloat] = None


class PESTipSetupRequest(_SetupBase):
    """Static energy map of the tip over the sheet (model pes_tip)."""

    tip_material: str = _TIP
    substrate_material: str = Field('Si', description='Substrate material, used only when '
                                    'the config has a [sub] section')
    tip_potential_type: str = _TIP_POT

    load_nN: Optional[PositiveFloat] = Field(None, description='One low normal load in nN')
    grid_n: Optional[int] = Field(None, ge=2, description='N for the N x N grid')
    tip_radius_A: Optional[PositiveFloat] = None


class PESSheetSetupRequest(_SetupBase):
    """Static energy map of one sheet over another (model pes_sheet)."""

    grid_n: Optional[int] = Field(None, ge=2)


#: Request field -> (section, key) it sets.
FIELD_MAP = {
    'loads_nN': ('general', 'force'), 'load_nN': ('general', 'force'),
    'speed_m_s': ('general', 'scan_speed'), 'tip_radius_A': ('tip', 'r'),
    'temperature_K': ('general', 'temp'), 'scan_angles_deg': ('general', 'scan_angle'),
    'pressures_GPa': ('general', 'pressure'), 'grid_n': ('pes', 'grid_n'),
    'layers': ('2D', 'layers'),
}


#: Request field -> (config section, what the card calls it, extra ops' values).
_TIP_PARTS = (('tip_material', 'tip', 'tip', {}),
              ('substrate_material', 'sub', 'substrate', {'amorph': 'a'}))
_MATERIAL_KEYS = ('mat', 'cif_path', 'pot_path', 'pot_type')


def _tip_and_substrate_ops(deps, model: str, request: BaseModel) -> Tuple[List[Op], List[str]]:
    """Point [tip] (and [sub]) at a material when they are unset or one is named.

    A blank AFM-family document has no tip or substrate material, so without
    this no setup could ever validate. The defaults (Si tip, amorphous Si
    substrate) are what examples/afm_config.ini uses.
    """
    probe = copy.deepcopy(deps.document)
    try:
        probe.set_model(model)
    except DocumentError as exc:
        raise ToolInputError(str(exc)) from exc
    ops, described = [], []
    for field_name, section, label, extra in _TIP_PARTS:
        name = getattr(request, field_name, None)
        if name is None:
            continue
        named = field_name in request.model_fields_set
        if not probe.has_section(section) and not named:
            continue  # an optional [sub] (pes_tip) stays off unless asked for
        unset = any(probe.get(section, key) in (None, '') for key in _MATERIAL_KEYS)
        if not unset and not named:
            continue
        structure, potential = resolve_material(deps, name, request.tip_potential_type)
        if not probe.has_section(section):
            ops.append(Op('enable_section', section, value=True))
        ops += [Op('set', section, 'mat', name),
                Op('set', section, 'cif_path', config_path(deps, structure.path)),
                Op('set', section, 'pot_path', config_path(deps, potential.path)),
                Op('set', section, 'pot_type', potential.pot_type)]
        ops += [Op('set', section, key, value) for key, value in extra.items()]
        kind = 'amorphous ' if extra.get('amorph') == 'a' else ''
        described.append(f'{label}: {kind}{name} ({potential.pot_type}) from '
                         f'{display_path(structure.path, deps.roots)}')
    notes = ['; '.join(described) + ' -- change if needed'] if described else []
    return ops, notes


def _setup(deps, model: str, request: _SetupBase, title: str) -> str:
    ops = [] if deps.document.model == model else [Op('set_model', value=model)]
    material, notes = material_ops(deps, request.materials, request.potential_type)
    ops += material
    tip_ops, tip_notes = _tip_and_substrate_ops(deps, model, request)
    ops += tip_ops
    notes += tip_notes
    for name, (section, key) in FIELD_MAP.items():
        value = getattr(request, name, None)
        if value is None or value == []:
            continue
        is_seq = isinstance(value, (list, tuple))
        ops.append(Op('set', section, key, list(value) if is_seq else value))
    if request.size_A is not None:
        ops += [Op('set', '2D', 'x', request.size_A[0]), Op('set', '2D', 'y', request.size_A[1])]
    return deps.propose(build_changeset(deps, title, ops, notes))


def setup_afm(deps, request: AFMSetupRequest) -> str:
    """Propose an AFM sliding simulation."""
    return _setup(deps, 'afm', request, 'AFM: ' + ', '.join(request.materials))


def setup_sheet(deps, request: SheetSetupRequest) -> str:
    """Propose a sheet-on-sheet sliding simulation."""
    return _setup(deps, 'sheetonsheet', request, 'Sheet on sheet: ' + ', '.join(request.materials))


def setup_indent(deps, request: IndentSetupRequest) -> str:
    """Propose an indentation (hold and retract) simulation."""
    return _setup(deps, 'indent', request, 'Indent: ' + ', '.join(request.materials))


def setup_pes_tip(deps, request: PESTipSetupRequest) -> str:
    """Propose a static PES scan of the tip over a sheet."""
    return _setup(deps, 'pes_tip', request, 'PES tip scan: ' + ', '.join(request.materials))


def setup_pes_sheet(deps, request: PESSheetSetupRequest) -> str:
    """Propose a static PES scan of a sheet over a sheet."""
    return _setup(deps, 'pes_sheet', request, 'PES sheet scan: ' + ', '.join(request.materials))
