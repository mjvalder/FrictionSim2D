"""Setup tools: read the Builder's documents, propose field changes, build sweeps."""
from __future__ import annotations

import copy
import difflib
import math
import re
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, conint, model_validator

from ...core.scaling import ScaleContext, scale
from ...gui.schema.overrides import override_for_section
from ...gui.schema.simulation_types import FAMILIES, MODES
from ...gui.controllers.document import DocumentError
from ..changes import Op, ToolInputError, build_changeset
from ..sandbox import display_path
from . import compact
from .materials import config_path, folder_material_ops, material_ops, resolve_material

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


def _suggest(name: str, known: List[str]) -> str:
    """``; did you mean ...?`` for ``name`` among ``known``, ignoring case; or ''."""
    lowered = {k.lower(): k for k in known}
    close = [lowered[m] for m in difflib.get_close_matches(name.lower(), list(lowered), n=3)]
    return f"; did you mean {', '.join(close)}?" if close else ''


def _section_spec(doc, scope: str, section: str):
    """``(section, spec)``: the exact section, else the one that differs only in case."""
    spec = doc.section(section)
    if spec is not None:
        return section, spec
    known = [s.key for s in doc.sections()]
    same = [k for k in known if k.lower() == section.lower()]
    if len(same) == 1:
        return same[0], doc.section(same[0])
    raise ToolInputError(f"unknown {scope} section '{section}'{_suggest(section, known)}; "
                         'use find_parameter to locate the field')


def set_fields(deps, changes: List[FieldChange], title: str = 'Set fields') -> str:
    """Propose values for config or settings fields (shown as a card)."""
    ops, enabled = [], set()
    for change in changes:
        section, key = _split_key(change.section, change.key)
        doc = deps.document if change.scope == 'config' else deps.settings_document
        section, spec = _section_spec(doc, change.scope, section)
        known = list(spec.field_map())
        if key not in known:
            raise ToolInputError(f"'{section}' has no key '{key}'{_suggest(key, known)}")
        if change.scope == 'config':
            if not doc.has_section(section) and section not in enabled:
                enabled.add(section)
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
        return [_significant(10 ** (a + i * (b - a) / (n - 1))) for i in range(n)]
    return [_significant(start + i * (stop - start) / (n - 1)) for i in range(n)]


def _significant(value: float) -> float:
    """``value`` to 9 significant figures: float noise goes, tiny values stay."""
    return float(f'{value:.9g}')


def make_sweep_tool(deps, start: float, stop: float, n: int,  # pylint: disable=unused-argument
                    spacing: Literal['linear', 'log'] = 'linear') -> str:
    """The values as text, for the model to pass on to a setup tool."""
    return 'values: ' + ', '.join(f'{v:g}' for v in make_sweep(start, stop, n, spacing))


def _scale_context(doc) -> ScaleContext:
    """What `scale` needs from the Builder's document (None for absent sections)."""
    def value(section, key):
        return doc.get(section, key) if doc.has_section(section) else None
    return ScaleContext(model=doc.model, tip_radius_A=value('tip', 'r'),
                        driving_spring_N_m=value('general', 'driving_spring'),
                        temperature_K=value('general', 'temp'))


def suggest_values(deps, quantity: str, descriptor: str, n: int) -> str:
    """Descriptor -> n linear values from a reference band (src.core.scaling), else the examples.

    Descriptors: low (also small, light, lower), mid (moderate, medium, middle,
    intermediate), high (large, heavy, higher); a phrase such as 'low load'
    works too. Quantities: load (force, normal load). Bands exist for AFM,
    indent and PES-tip loads; anything else gets the example-config advice.
    """
    scaled = scale(quantity, descriptor, _scale_context(deps.document))
    if scaled is None:
        return (f"No scaling rule for '{descriptor} {quantity}' yet: what counts as "
                f"{descriptor} depends on tip size, model size and speed. Offer the "
                f"example-config values from the knowledge (defaults) as a starting point, "
                f"say they come from the example config, and ask the user to confirm them "
                f"or give a range (min, max and unit); then call make_sweep with n={n}.")
    values = make_sweep(scaled.low, scaled.high, n, 'linear')
    lines = [f"values: {', '.join(f'{v:g}' for v in values)} {scaled.unit}",
             f'working: {scaled.working}']
    lines += [f'warning: {w}' for w in scaled.warnings]
    lines.append('Tell the user where the numbers come from and confirm before proposing.')
    return compact('\n'.join(lines))


class _SetupBase(BaseModel):
    materials: List[str] = Field(default_factory=list, description='Material names exactly '
                                 'as search_structures returns them, e.g. h-MoS2')
    materials_folder: Optional[str] = Field(
        None, description="Instead of materials: every structure in this folder, e.g. "
                          "'workspace:/materials' (for many materials or 'all of them')")
    potential_type: Optional[str] = Field(
        None, description="e.g. 'sw', 'tersoff', 'rebomos', 'reaxff'")
    size_A: Optional[Tuple[PositiveFloat, PositiveFloat]] = Field(
        None, description='Sheet x, y in Angstrom')
    layers: Optional[List[conint(ge=1)]] = Field(  # type: ignore[valid-type]
        None, description='Layer counts to sweep, e.g. [1, 2, 3]')

    @model_validator(mode='after')
    def _one_source(self):
        if bool(self.materials) == bool(self.materials_folder):
            raise ValueError('give either materials or materials_folder, not both or neither '
                             '(materials_folder is the folder of .cif structure files; the '
                             'potentials named after them are found on their own)')
        return self

    def describe(self) -> str:
        """The materials as a card title names them."""
        return ', '.join(self.materials) or f'every material in {self.materials_folder}'


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


#: Model -> (source example, {(section, key): value}): geometry a blank
#: document leaves unset, copied from the example configs so a setup can
#: validate without the model inventing numbers. Filled only when neither the
#: request nor the document has a value; the card names the source.
#: tests/agent/test_tools_recipes.py checks every value against its file.
EXAMPLE_DEFAULTS = {
    # examples/afm_config.ini sweeps layers [1,2,3]; one layer is the default here.
    'afm': ('examples/afm_config.ini', {
        ('2D', 'x'): 100, ('2D', 'y'): 100, ('2D', 'layers'): [1],
        ('tip', 'r'): 25, ('sub', 'thickness'): 12}),
    # examples/sheet_config.ini: three layers is SheetOnSheetSimulation.MIN_LAYERS.
    'sheetonsheet': ('examples/sheet_config.ini', {
        ('2D', 'x'): 100, ('2D', 'y'): 100, ('2D', 'layers'): [3]}),
    # examples/indent_config.ini sweeps layers [1, 2]; one layer is the default here.
    'indent': ('examples/indent_config.ini', {
        ('2D', 'x'): 60, ('2D', 'y'): 60, ('2D', 'layers'): [1],
        ('tip', 'r'): 15, ('sub', 'thickness'): 12}),
    # examples/pes_tip_config.ini
    'pes_tip': ('examples/pes_tip_config.ini', {
        ('2D', 'x'): 60, ('2D', 'y'): 60, ('2D', 'layers'): [1],
        ('tip', 'r'): 25, ('sub', 'thickness'): 12}),
    # examples/pes_sheet_config.ini: a frozen bottom layer + a scanned top one.
    'pes_sheet': ('examples/pes_sheet_config.ini', {
        ('2D', 'x'): 30, ('2D', 'y'): 30, ('2D', 'layers'): [2]}),
}


#: Example values that would fail validation, replaced with the reason shown on
#: the change note. examples/pes_tip_config.ini (fixture data, left as is) uses a
#: 60 x 60 Angstrom sheet with [tip] r = 25, below the 2x-tip-diameter footprint
#: (100 Angstrom) the AFM models require.
EXAMPLE_OVERRIDES = {
    'pes_tip': ({('2D', 'x'): 100, ('2D', 'y'): 100},
                'sheet raised to 100 x 100 so it is at least 2x the tip diameter'),
}

#: Request field -> (config section, what the card calls it, extra ops' values).
_TIP_PARTS = (('tip_material', 'tip', 'tip', {}),
              ('substrate_material', 'sub', 'substrate', {'amorph': 'a'}))
_MATERIAL_KEYS = ('mat', 'cif_path', 'pot_path', 'pot_type')


def _probe(deps, model: str):
    """A copy of the Builder's document switched to ``model``."""
    probe = copy.deepcopy(deps.document)
    try:
        probe.set_model(model)
    except DocumentError as exc:
        raise ToolInputError(str(exc)) from exc
    return probe


def _tip_and_substrate_ops(deps, probe, request: BaseModel) -> Tuple[List[Op], List[str]]:
    """Point [tip] (and [sub]) at a material when they are unset or one is named.

    A blank AFM-family document has no tip or substrate material, so without
    this no setup could ever validate. The defaults (Si tip, amorphous Si
    substrate) are what examples/afm_config.ini uses.
    """
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


def _example_default_ops(probe, model: str, ops: List[Op]) -> Tuple[List[Op], List[str]]:
    """Fill geometry still unset after ``ops`` from the model's example config."""
    source, values = EXAMPLE_DEFAULTS.get(model, ('', {}))
    overrides, reason = EXAMPLE_OVERRIDES.get(model, ({}, ''))
    values = {**values, **overrides}
    proposed = {(op.section, op.key) for op in ops if op.kind == 'set'}
    filled, described = [], []
    for (section, key), value in values.items():
        spec = probe.section(section)
        if (spec is None or key not in spec.field_map() or not probe.has_section(section)
                or (section, key) in proposed
                or probe.get(section, key) not in (None, '', [])):
            continue
        filled.append(Op('set', section, key, copy.deepcopy(value)))
        described.append(f'{section}.{key}={value}')
    if not described:
        return [], []
    note = f"defaults from {source} -- change if needed: {', '.join(described)}"
    if reason and any((op.section, op.key) in overrides for op in filled):
        note += f' ({reason})'
    return filled, [note]


def _setup(deps, model: str, request: _SetupBase, title: str) -> str:
    ops = [] if deps.document.model == model else [Op('set_model', value=model)]
    if request.materials_folder:
        material, notes = folder_material_ops(deps, request.materials_folder,
                                              request.potential_type)
    else:
        material, notes = material_ops(deps, request.materials, request.potential_type)
    ops += material
    probe = _probe(deps, model)
    tip_ops, tip_notes = _tip_and_substrate_ops(deps, probe, request)
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
    default_ops, default_notes = _example_default_ops(probe, model, ops)
    ops += default_ops
    notes += default_notes
    return deps.propose(build_changeset(deps, title, ops, notes))


def setup_afm(deps, request: AFMSetupRequest) -> str:
    """Propose an AFM sliding simulation."""
    return _setup(deps, 'afm', request, 'AFM: ' + request.describe())


def setup_sheet(deps, request: SheetSetupRequest) -> str:
    """Propose a sheet-on-sheet sliding simulation."""
    return _setup(deps, 'sheetonsheet', request, 'Sheet on sheet: ' + request.describe())


def setup_indent(deps, request: IndentSetupRequest) -> str:
    """Propose an indentation (hold and retract) simulation."""
    return _setup(deps, 'indent', request, 'Indent: ' + request.describe())


def setup_pes_tip(deps, request: PESTipSetupRequest) -> str:
    """Propose a static PES scan of the tip over a sheet."""
    return _setup(deps, 'pes_tip', request, 'PES tip scan: ' + request.describe())


def setup_pes_sheet(deps, request: PESSheetSetupRequest) -> str:
    """Propose a static PES scan of a sheet over a sheet."""
    return _setup(deps, 'pes_sheet', request, 'PES sheet scan: ' + request.describe())
