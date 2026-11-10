"""Generate tool-calling conversations for the assistant fine-tune (spec §3).

For each conversation: pick a reviewed template (``sources/templates.yaml``,
or the teacher's draft with ``--drafts``), fill its slots from real data (the
material index, potentials, scaling bands, config labels), run the category's
tool calls against the real ``src/agent`` tools on a scratch workspace, have
the local teacher write the final reply from the genuine tool results
(source B), and keep the conversation only if code checks pass (the eval
tier's ``grounded`` rule, a question for must-ask, no proposal for
declines, the expected values quoted). Output: ``data/<name>.jsonl`` in the
OpenAI/Qwen message format with the exact tool schemas the runtime
registers, plus ``data/<name>.card.json`` (counts, rejects, hashes).

    python tools/agent_data/generate.py --n 200 --name smoke --drafts

Run with the aiida env (it imports the app and its test helpers).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position,wrong-import-order
from src.agent.changes import ToolInputError  # noqa: E402
from src.agent.events import TurnEvent  # noqa: E402
from src.agent.knowledge_index import load_knowledge  # noqa: E402
from src.potentials.utils import KNOWN_TYPES  # noqa: E402
from src.agent.prompts import KNOWLEDGE_DIR, ROLE  # noqa: E402
from src.agent.tools.materials import index_for, resolve_material  # noqa: E402
from src.agent.tools.setup import make_sweep  # noqa: E402
from src.core.config import GlobalSettings  # noqa: E402
from src.core.scaling import ScaleContext, scale  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.agent.evals.cases import CASES, count_by, grounded, numbers_in, quotes  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402
from tests.helpers.agent_eval import EvalResult  # noqa: E402
from tests.helpers.results_tree import make_family_run  # noqa: E402
from categories import BY_NAME, Category  # noqa: E402

TEMPLATES = HERE / 'sources' / 'templates.yaml'
DRAFTS = HERE / 'sources' / 'templates.draft.yaml'
DATA = HERE / 'data'
OLLAMA = 'http://127.0.0.1:11434/api/chat'
TEACHER = 'qwen3:14b'
TEACHER_LICENCE = 'Apache-2.0'
#: Categories the generator cannot build yet: Materials Project answers need real
#: database rows (no key here, and invented rows would teach invented facts).
UNSUPPORTED = {'materials_project'}
#: Sampling weight per category (default 1): concepts are the domain layer (spec §4.4).
WEIGHTS = {'concept': 4,
           # p6: the tool groups the 4B still failed most (parameters, sweeps, plots)
           'param_change': 6, 'settings_change': 6, 'find_parameter': 3, 'speed_sweep': 4,
           'linear_sweep': 3, 'log_sweep': 3, 'load_band': 5, 'plot_cof_load': 3,
           'plot_timeseries': 3, 'plot_rank': 3, 'plot_slope_bars': 3, 'plot_filtered': 3,
           'plot_types': 2, 'count_runs': 2, 'summarise_dataset': 3, 'describe_folder': 2,
           'find_datasets': 2, 'explain_config': 2, 'afm_setup_temperature': 2,
           'sheet_angles': 2, 'recommend_layers': 2}
#: A request this similar to an eval prompt is dropped: the evals stay held out.
EVAL_SIMILARITY = 0.75
EVAL_PROMPTS = [case.prompt.lower() for case in CASES.values()]
#: The model each category's Builder document is on (default afm).
DOCUMENT_MODEL = {'ask_sheet_band': 'sheetonsheet', 'ask_sheet_force': 'sheetonsheet'}

QUESTION_BRIEF = """Below is a passage from "{title}". Write one question that a materials
scientist working on friction simulations of 2D materials might type into a chat
assistant, which this passage answers. Do not mention the passage, its authors or
any study; do not put numbers in the question. Return JSON: {{"question": "..."}}

Passage:
{text}"""

REPLY_BRIEF = """You are the FrictionSim2D assistant writing your final chat reply after
using tools. Rules:
{rules}
This turn: {kind_rule}
Write only the reply text, at most three sentences, plain text, addressing the user
as "you". Use material names
exactly as the tool results give them. Every number you write must appear in the
tool results or the user's request.

User request: {request}

Tool calls and results:
{trace}"""

KIND_RULES = {
    'act': 'report what the tools did. A result that says "Proposed only" or "decides '
           'whether" means the change waits for the user; a result that says "will be '
           'applied" means it is already in the Builder and the user can undo it.',
    'ask': 'do not propose anything; ask one clear question about exactly what is wrong '
           'or missing (the unit, the unknown material, or which kind of simulation), '
           'ending with a question mark.',
    'declines': 'explain briefly that you cannot make this change and where the user '
                'can change it themselves; do not ask a question.',
}
#: What a category's reply must say, which domain_error checks for. Without it the
#: teacher rarely says it: p6 kept 0 of 42 negative-load and 1 of 33 guess declines.
CATEGORY_RULES = {
    'decline_negative_load': ' Say that the load must be positive (greater than zero).',
    'decline_guess': ' Say that the value is only known by running the simulation; write '
                     'no number.',
}


# ---- the app's tools, called exactly as the runtime calls them ----------------------

@dataclass
class Toolbox:
    """The registered tools and their JSON schemas, from a real AgentSession."""

    tools: Dict[str, Any]
    schemas: List[dict]
    schema_hash: str

    @classmethod
    def load(cls) -> 'Toolbox':
        # pylint: disable=import-outside-toplevel,protected-access
        from pydantic_ai.models.function import FunctionModel
        from src.agent.runtime import AgentSession, tool_schema_hash, tool_schemas
        from src.agent.settings import AgentSettings
        session = AgentSession(AgentSettings(), model=FunctionModel(lambda m, i: None))
        tools = dict(session.agent._function_toolset.tools)
        return cls(tools, tool_schemas(), tool_schema_hash())

    def retry_text(self, deps, name: str, args: dict) -> Optional[str]:
        """What the runtime sends back for a call it rejects, or None if it is accepted."""
        # pylint: disable=import-outside-toplevel
        from pydantic import ValidationError
        from pydantic_ai.messages import RetryPromptPart
        try:
            self.tools[name].function_schema.validator.validate_python(args)
        except ValidationError as exc:
            content = exc.errors(include_url=False, include_context=False)
            return RetryPromptPart(tool_name=name, content=content).model_response()
        try:
            self.call(deps, name, args)
        except ToolInputError as exc:
            deps.pending.clear()
            return RetryPromptPart(tool_name=name, content=str(exc)).model_response()
        deps.pending.clear()  # accepted after all: not a mistake to learn from
        return None

    def call(self, deps, name: str, args: dict) -> str:
        """Validate ``args`` like the model's call and run the tool through runtime._call."""
        # pylint: disable=import-outside-toplevel
        from pydantic_ai.exceptions import ModelRetry
        tool = self.tools[name]
        kwargs = tool.function_schema.validator.validate_python(args)
        try:
            return tool.function(SimpleNamespace(deps=deps), **kwargs)
        except ModelRetry as exc:
            raise ToolInputError(str(exc)) from exc


def knowledge_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(KNOWLEDGE_DIR.glob('*.md')):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# ---- slot values from real data -------------------------------------------------

def _numbers(values) -> str:
    return ', '.join(f'{v:g}' for v in values)


@dataclass
class Scene:
    """One conversation's world: slot values, the deps, and what is expected."""

    rng: random.Random
    deps: Any
    slots: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)


def usable_materials(deps) -> List[Tuple[str, str]]:
    """``(material, potential type)`` pairs the examples can actually build."""
    pairs = []
    for found in index_for(deps).structures:
        try:
            _, potential = resolve_material(deps, found.name, 'sw')
        except ToolInputError:
            continue
        pairs.append((found.name, potential.pot_type))
    return pairs


def fill(scene: Scene, pairs: List[Tuple[str, str]]) -> None:
    """Sample a value for each slot (the shown text and the typed value)."""
    rng, s, v = scene.rng, scene.slots, scene.values
    (material, pot), (material_b, _) = rng.sample(pairs, 2)
    v.update(material=material, potential=pot, material_b=material_b)
    s.update(material=material, potential=pot, material_b=material_b,
             formula=re.sub(r'^[a-z]{1,2}-', '', material))
    loads = sorted(rng.sample(range(5, 105, 5), rng.randint(2, 5)))
    v['loads'] = loads
    s['loads'] = _numbers(loads)
    v['load'] = rng.choice(range(5, 65, 5))
    s['load'] = f"{v['load']:g}"
    v['pressures'] = sorted(rng.sample([0.5, 1, 1.5, 2, 3, 4, 5], rng.randint(2, 4)))
    s['pressures'] = _numbers(v['pressures'])
    v['pressure'] = rng.choice(v['pressures'])
    s['pressure'] = f"{v['pressure']:g}"
    v['n'] = rng.randint(3, 6)
    v['start'], v['stop'] = rng.choice([5, 10, 15]), rng.choice([40, 50, 60, 80, 100])
    v['band'] = rng.choice(['low', 'moderate', 'high'])
    v['temperature'] = rng.choice(range(100, 550, 50))
    v['angles'] = sorted(rng.sample([0, 15, 30, 45, 60, 90], rng.randint(2, 4)))
    s['angles'] = _numbers(v['angles'])
    v['grid'] = rng.randint(6, 20)
    for key in ('n', 'start', 'stop', 'band', 'temperature', 'grid'):
        s[key] = str(v[key])
    v['folder'] = rng.choice(['runs', 'campaign', 'friction', 'tests', 'sweep']) + \
        rng.choice(['_', '-']) + rng.choice(['2026', 'alpha', 'beta', 'mos2', 'v2'])
    s['folder'] = v['folder']
    v['dataset'] = rng.choice(['loads', 'campaign', 'batch', 'series']) + '_' + \
        rng.choice(['a', 'b', '01', '02', 'oct'])
    s['dataset'] = v['dataset']
    v['wrong_unit'] = rng.choice(['nm', 'GPa', 'K', 'm/s', 'Angstrom'])
    s['wrong_unit'] = v['wrong_unit']
    v['unknown'] = rng.choice(['zr', 'qu', 'vex', 'plor', 'myth']) + rng.choice(
        ['onium', 'idite', 'ane', 'ex'])
    first = rng.randint(1, 2)
    v['layers'] = list(range(first, first + rng.randint(2, 3)))
    s['layers'] = f"{v['layers'][0]} to {v['layers'][-1]}"
    v['negative_load'] = -rng.choice(range(5, 55, 5))
    s['negative_load'] = f"{v['negative_load']:g}"
    s['quantity'] = rng.choice(['friction coefficient', 'friction force', 'shear strength',
                                'energy barrier for sliding'])
    v['speeds'] = (rng.choice([0.5, 1]), rng.choice([2, 5, 10]))
    v['log_start'], v['log_stop'] = rng.choice([1, 2, 5]), rng.choice([50, 100, 200])
    s.update({k: v[k] for k in ('parameter', 'value', 'setting') if k in v})


def parameter_slot(scene: Scene) -> None:
    """A real config field: its label as the user would say it, and a value."""
    choices = [('scan speed', 'general', 'scan_speed', [0.5, 1, 2, 5], 'm/s'),
               ('temperature', 'general', 'temp', [100, 200, 300, 400], 'K'),
               ('tip radius', 'tip', 'r', [15, 20, 30, 40], 'Angstrom'),
               ('driving spring', 'general', 'driving_spring', [5, 8, 10, 20], 'N/m')]
    label, section, key, values, unit = scene.rng.choice(choices)
    value = scene.rng.choice(values)
    scene.values.update(parameter=label, section=section, key=key, value=value, unit=unit)
    scene.slots.update(parameter=label, value=f'{value:g}')


def global_setting_slot(scene: Scene) -> None:
    """A real global MD setting, as the user would name it, and a value."""
    choices = [('MD timestep', 'simulation', 'timestep', [0.0005, 0.001, 0.002]),
               ('thermo output interval', 'simulation', 'thermo', [100, 500, 1000, 2000]),
               ('neighbour list skin', 'simulation', 'neighbor_list', [1.0, 2.0, 3.0])]
    label, section, key, values = scene.rng.choice(choices)
    value = scene.rng.choice(values)
    scene.values.update(global_setting=label, section=section, key=key, value=value)
    scene.slots.update(global_setting=label, value=f'{value:g}')


def field_slot(scene: Scene, scope: str) -> bool:
    """Any numeric field of the open config (``scope='config'``) or of the global
    settings, named by its form label, with a new value; False when the label does
    not find the field first (the trace would then contradict find_parameter)."""
    # pylint: disable=import-outside-toplevel
    from src.agent.tools.setup import _entries, find_parameter  # same lookup as the tool
    fields = []
    for where, section, spec, override, value in _entries(scene.deps):
        if where != scope or section in EXECUTION_SECTIONS or isinstance(value, bool) \
                or not isinstance(value, (int, float)) or not value:
            continue
        label = (override.label if override and override.label else spec.name) or spec.key
        fields.append((section, spec.key, label.replace('_', ' ').lower(), value))
    if not fields:
        return False
    section, key, label, value = scene.rng.choice(fields)
    new = value * scene.rng.choice([0.5, 1.5, 2])
    new = int(round(new)) if isinstance(value, int) else float(f'{new:.3g}')
    if new == value or not find_parameter(scene.deps, label).startswith(
            f'{scope}:{section}.{key} '):
        return False
    if scope == 'config':
        scene.values.update(parameter=label, section=section, key=key, value=new, unit='')
        scene.slots.update(parameter=label, value=f'{new:g}')
    else:
        scene.values.update(global_setting=label, section=section, key=key, value=new)
        scene.slots.update(global_setting=label, value=f'{new:g}')
    return True


#: Settings fixed at execution, which the assistant declines (SETTINGS_TO_DECLINE).
EXECUTION_SECTIONS = {'hpc', 'aiida', 'database'}
#: Share of parameter requests drawn from every numeric field rather than the
#: hand-picked few, so the model learns the lookup, not a short list.
ANY_FIELD_RATE = 0.6


def missing_potential(scene: Scene) -> bool:
    """Point the potential slot at a type the material has no file for."""
    # pylint: disable=import-outside-toplevel
    from src.agent.tools.materials import _potentials_for  # same lookup as the tool
    have = {f.pot_type for f in _potentials_for(scene.deps, scene.values['material'], None)}
    absent = sorted(t for t in KNOWN_TYPES if t not in have)
    if not absent:
        return False
    scene.values['potential'] = scene.slots['potential'] = scene.rng.choice(absent)
    return True


SETTINGS_TO_DECLINE = ['HPC walltime', 'minimization command', 'number of MPI processes',
                       'conda environment', 'scheduler modules']


# ---- per-category tool plans ------------------------------------------------------

Plan = List[Tuple[str, dict]]


def _setup(name: str, scene: Scene, **extra) -> Plan:
    v = scene.values
    args = {'materials': [v['material']], 'potential_type': v['potential'], **extra}
    return [(name, args)]


def plan_for(category: str, scene: Scene) -> Callable[[Callable[[str, dict], str]], Plan]:
    """A function that runs the category's calls (each may depend on the last result)."""
    v = scene.values

    def static(plan: Plan):
        return lambda run: [(n, a) for n, a in plan if run(n, a) is not None]

    if category == 'afm_setup':
        return static(_setup('setup_afm', scene, loads_nN=v['loads']))
    if category == 'afm_setup_temperature':
        return static(_setup('setup_afm', scene, loads_nN=[v['load']],
                             temperature_K=v['temperature']))
    if category == 'afm_two_materials':
        plan = _setup('setup_afm', scene, loads_nN=v['loads'])
        plan[0][1]['materials'].append(v['material_b'])
        return static(plan)
    if category == 'sheet_setup':
        return static(_setup('setup_sheet', scene, pressures_GPa=v['pressures']))
    if category == 'sheet_angles':
        return static(_setup('setup_sheet', scene, pressures_GPa=[v['pressure']],
                             scan_angles_deg=v['angles']))
    if category == 'indent_setup':
        return static(_setup('setup_indent', scene, loads_nN=v['loads']))
    if category == 'pes_tip_setup':
        return static(_setup('setup_pes_tip', scene))
    if category == 'pes_sheet_setup':
        return static(_setup('setup_pes_sheet', scene, grid_n=v['grid']))
    if category == 'linear_sweep':
        values = make_sweep(v['start'], v['stop'], v['n'])
        v['expected'] = values
        return static([('make_sweep', {'start': v['start'], 'stop': v['stop'], 'n': v['n']}),
                       *_setup('setup_afm', scene, loads_nN=[float(f'{x:g}') for x in values])])
    if category == 'load_band':
        scaled = scale('load', v['band'], ScaleContext(model='afm'))
        values = make_sweep(scaled.low, scaled.high, v['n'])
        v['expected'] = values
        return static([('suggest_values', {'quantity': 'load', 'descriptor': v['band'],
                                           'n': v['n']}),
                       *_setup('setup_afm', scene, loads_nN=[float(f'{x:g}') for x in values])])
    if category == 'param_change':
        return static([('find_parameter', {'query': v['parameter']}),
                       ('set_fields', {'changes': [{'section': v['section'], 'key': v['key'],
                                                    'value': v['value']}]})])
    if category == 'find_parameter':
        return static([('find_parameter', {'query': v['parameter']})])
    if category == 'search_structures':
        return static([('search_structures', {'query': scene.slots['formula']})])
    if category == 'search_potentials':
        return static([('search_potentials', {'material': v['material'],
                                              'potential_type': v['potential']})])
    if category == 'copy_structure':
        structure, _ = resolve_material(scene.deps, v['material'], v['potential'])
        source = f'examples:/materials/{structure.path.name}'
        return static([('search_structures', {'query': v['material']}),
                       ('propose_copy_in', {'source': source})])
    if category == 'make_folder':
        return static([('propose_mkdir', {'path': v['folder']})])
    if category == 'list_workspace':
        return static([('workspace_overview', {})])
    if category == 'summarise_dataset':
        return static([('summarise_results', {'path': f"workspace:/{v['dataset']}"})])
    dataset = f"workspace:/{v['dataset']}"

    def plot(plot_type: str, fields=()) -> Plan:
        return [('plot_schema', {'plot_type': plot_type}),
                # flat, as the tool schema shows it (single-model arguments are inlined)
                ('propose_plot', {'plot_type': plot_type, 'datasets': [dataset],
                                  'fields': [{'key': k, 'value': x} for k, x in fields]})]
    cof_axes = (('x_axis', 'force'), ('y_axis', 'cof'))
    if category == 'plot_cof_load':
        return static([('find_result_sets', {}), *plot('summary', cof_axes)])
    if category == 'plot_timeseries':
        return static(plot('timeseries'))
    if category == 'plot_rank':
        return static(plot('rank_friction'))
    if category == 'plot_slope_bars':
        return static(plot('cof_histogram'))
    if category == 'plot_filtered':
        return static(plot('summary', cof_axes + (('filter_materials', [v['material']]),)))
    if category == 'explain_config':
        return static([('explain_current', {})])
    if category in ('list_modes', 'ask_which_family'):
        return static([('list_modes', {})])
    if category == 'plot_types':
        return static([('plot_types', {})])
    if category == 'settings_change':
        return static([('find_parameter', {'query': v['global_setting']}),
                       ('set_fields', {'changes': [{'scope': 'settings', 'section': v['section'],
                                                    'key': v['key'], 'value': v['value']}]})])
    if category == 'speed_sweep':
        start, stop = (float(scene.slots['start']), float(scene.slots['stop']))
        values = [float(f'{x:g}') for x in make_sweep(start, stop, v['n'])]
        v['expected'] = values
        return static([('make_sweep', {'start': start, 'stop': stop, 'n': v['n']}),
                       ('set_fields', {'changes': [{'section': 'general', 'key': 'scan_speed',
                                                    'value': values}]})])
    if category == 'log_sweep':
        start, stop = float(scene.slots['start']), float(scene.slots['stop'])
        values = make_sweep(start, stop, v['n'], 'log')
        v['expected'] = values
        return static([('make_sweep', {'start': start, 'stop': stop, 'n': v['n'],
                                       'spacing': 'log'}),
                       *_setup('setup_afm', scene, loads_nN=[float(f'{x:g}') for x in values])])
    if category == 'find_datasets':
        return static([('find_result_sets', {})])
    if category == 'describe_folder':
        return static([('describe_run', {'path': dataset})])
    if category == 'count_runs':
        return static([('summarise_results', {'path': dataset, 'material': v['material']})])
    if category == 'copy_potential':
        _, potential = resolve_material(scene.deps, v['material'], v['potential'])
        source = 'examples:/' + potential.path.relative_to(
            scene.deps.roots.examples).as_posix()
        return static([('search_potentials', {'material': v['material'],
                                              'potential_type': v['potential']}),
                       ('propose_copy_in', {'source': source, 'dest_dir': 'potentials'})])
    if category == 'recommend_corrugation':
        plan = _setup('setup_pes_sheet', scene)
        plan[0][1]['materials'].append(v['material_b'])
        return static(plan)
    if category == 'recommend_layers':
        return static(_setup('setup_afm', scene, loads_nN=[v['load']], layers=v['layers']))
    if category == 'ask_sheet_band':
        return static([('suggest_values', {'quantity': 'pressure', 'descriptor': v['band'],
                                           'n': v['n']})])
    if category == 'ask_unknown_potential':
        return static([('search_potentials', {'material': v['material'],
                                              'potential_type': v['potential']})])
    if category == 'ask_missing_dataset':
        return static([('find_result_sets', {})])
    if category == 'concept':
        return static([('search_knowledge', {'query': v['question']})])
    if category == 'lattice_pair':
        return static([('lattice_mismatch', {'material_a': v['material'],
                                             'material_b': v['material_b']})])
    if category == 'lattice_one':
        return static([('lattice_mismatch', {'material_a': v['material']})])
    if category == 'ask_unknown_material':
        return static([('search_structures', {'query': v['unknown']})])
    return static([])  # the remaining asks and declines need no tool


def prepare_world(category: str, scene: Scene) -> None:
    """Workspace fixtures a category needs before its tools run."""
    if category in ('summarise_dataset', 'plot_cof_load', 'plot_timeseries', 'plot_rank',
                    'plot_slope_bars', 'plot_filtered', 'find_datasets', 'describe_folder',
                    'count_runs'):
        root = scene.deps.roots.workspace / scene.values['dataset']
        for material in (scene.values['material'], scene.values['material_b']):
            make_family_run(root, 'afm', material=material, loads=(10.0, 30.0))
        scene.values['complete'] = 2  # make_family_run: one complete run per load
        DataReader(results_dir=str(root)).export_full_data_to_json()
    if category == 'list_workspace':
        for name in scene.rng.sample(['campaign_a', 'notes', 'runs_2026', 'old_tests',
                                      'mos2_loads', 'figures'], 3):
            (scene.deps.roots.workspace / name / 'run1').mkdir(parents=True)


# ---- teacher reply and checks ----------------------------------------------------

def teacher_reply(request: str, kind: str, trace: List[Tuple[str, dict, str]],
                  category: str = '') -> str:
    lines = [f'- {name}({json.dumps(args)}) ->\n{result}' for name, args, result in trace]
    prompt = REPLY_BRIEF.format(rules=ROLE.split('Rules:', 1)[1].strip(),
                                kind_rule=KIND_RULES[kind] + CATEGORY_RULES.get(category, ''),
                                request=request,
                                trace='\n'.join(lines) or '(no tool was needed)')
    payload = {'model': TEACHER, 'stream': False, 'think': False,
               'options': {'temperature': 0.4, 'num_ctx': 8192},
               'messages': [{'role': 'user', 'content': prompt}]}
    http = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(http, timeout=600) as response:
        return json.load(response)['message']['content'].strip()


_WAITS = re.compile(r'\b(wait\w*|approv\w*|confirm\w*|review)\b', re.IGNORECASE)
_DONE = re.compile(r'\b(has been|have been|was|were|is now) (applied|updated|changed|set)\b',
                   re.IGNORECASE)


def status_error(reply: str, trace: List[Tuple[str, dict, str]]) -> Optional[str]:
    """Whether the reply misstates if the last proposal is applied or waiting."""
    proposals = [r for _, _, r in trace if ' proposed (' in r]
    if not proposals:
        return None
    if 'will be applied' in proposals[-1]:
        return 'says an applied change waits' if _WAITS.search(reply) else None
    return 'says a waiting change is applied' if _DONE.search(reply) else None


def domain_error(category: str, scene: Scene, reply: str,
                 trace: List[Tuple[str, dict, str]]) -> Optional[str]:
    """Category-specific content checks for the Phase 4 categories."""
    v, low = scene.values, reply.lower()
    needs = {
        'ask_sheet_force': ('gpa', 'pressure'), 'ask_sheet_band': ('gpa', 'pressure', 'range'),
        'ask_unknown_potential': (v['potential'].lower(),),
        'ask_missing_dataset': (v['dataset'].lower(),),
        'decline_negative_load': ('positive', 'greater than'),
        'decline_guess': ('simulat', 'run'),
        'plot_types': ('summary', 'timeseries', 'rank'),
    }.get(category)
    if needs and not any(n in low for n in needs):
        return f'reply names none of {needs}'
    if category == 'count_runs' and not count_by(SimpleNamespace(text=reply), v['complete']):
        return 'count not stated'
    if category == 'concept':
        titles = re.findall(r'^\[\d\] (.+?)(?: -- | \()', trace[0][2], re.MULTILINE)
        if not any(t.lower()[:40] in low for t in titles):
            return 'no source title named'
    if category == 'decline_guess' and re.search(r'\d', reply):
        return 'guessed a number'
    return None


def check(category: str, kind: str, scene: Scene, request: str, reply: str,
          trace: List[Tuple[str, dict, str]]) -> Optional[str]:
    """None if the conversation is right, else why it is dropped."""
    events = [TurnEvent('tool_result', f'{n}: {r}') for n, _, r in trace]
    result = EvalResult(deps=scene.deps, events=events + [TurnEvent('text', reply)])
    if not reply or '<think>' in reply:
        return 'empty or thinking reply'
    if not grounded(result, extra=numbers_in(request)):
        return 'ungrounded number'
    if re.search(r'\bthe user\b', reply, re.IGNORECASE):
        return 'third person'
    if category == 'ask_wrong_unit' and scene.values['wrong_unit'].lower() not in reply.lower():
        return 'question misses the unit'
    if category == 'ask_unknown_material' and scene.values['unknown'] not in reply:
        return 'question misses the material'
    flagged = any(re.search(r'\b[1-9]\d* issue\(s\)', r) for _, _, r in trace)
    if flagged and re.search(r'\bno (other |further )?issues?\b', reply, re.IGNORECASE):
        return 'denies the issues the tool flagged'
    why = status_error(reply, trace) or domain_error(category, scene, reply, trace)
    if why:
        return why
    if kind == 'ask':
        return None if '?' in reply and not scene.deps.pending else 'did not ask'
    if kind == 'declines':
        return None if not scene.deps.pending else 'proposed on a decline'
    if category.endswith('_setup') or category in (
            'afm_two_materials', 'sheet_angles', 'linear_sweep', 'load_band', 'log_sweep',
            'recommend_corrugation', 'recommend_layers', 'speed_sweep', 'settings_change'):
        if not scene.deps.pending:
            return 'no proposal'
    if 'expected' in scene.values and not all(quotes(result, x) or f'{x:g}' in reply
                                              for x in scene.values['expected']):
        return 'expected values not quoted'
    if category == 'find_parameter' and scene.values['key'] not in reply:
        return 'parameter key not named'
    return None


# ---- one conversation ------------------------------------------------------------

CONCEPT = Category('concept', 'A question about 2D materials or friction.', ())


def teacher_json(prompt: str, temperature: float = 0.8) -> dict:
    payload = {'model': TEACHER, 'stream': False, 'think': False, 'format': 'json',
               'options': {'temperature': temperature, 'num_ctx': 4096},
               'messages': [{'role': 'user', 'content': prompt}]}
    http = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(http, timeout=600) as response:
        return json.loads(json.load(response)['message']['content'])


def concept_question(rng: random.Random, index) -> Tuple[str, str]:
    """``(question, source title)``: the teacher asks what a random passage answers."""
    passage = rng.choice(index.passages)
    question = teacher_json(QUESTION_BRIEF.format(title=passage.title, text=passage.text))
    return ' '.join(str(question.get('question', '')).split()), passage.title


def too_close_to_eval(request: str) -> bool:
    low = request.lower()
    return any(difflib.SequenceMatcher(None, low, p).ratio() > EVAL_SIMILARITY
               for p in EVAL_PROMPTS)


#: Share of conversations with one injected tool-call mistake and its recovery.
MISTAKE_RATE = 0.3
#: Share of single-turn setups that get a follow-up turn on the same config.
FOLLOWUP_RATE = 0.25
FIRST_TURNS = ('afm_setup', 'sheet_setup', 'indent_setup', 'linear_sweep', 'load_band')
FOLLOWUPS = ('param_change', 'speed_sweep', 'find_parameter', 'explain_config',
             'settings_change')


INVENTED_PLOTS = ['cof_vs_load', 'bar_chart', 'friction_vs_load', 'line', 'histogram']


def mistake(name: str, args: dict, rng: random.Random) -> Optional[dict]:
    """A plausible wrong version of ``args``: a renamed or dropped key, a list
    sent as text, or a nested field lifted to the top level. None if no change."""
    bad = json.loads(json.dumps(args))
    keys = [k for k in bad if k != 'scope']
    if not keys:
        return None
    key = rng.choice(keys)
    value = bad[key]
    choice = rng.choice(['rename', 'drop', 'stringify', 'nest', 'guess'])
    if choice == 'guess' and name == 'propose_plot' and 'plot_type' in bad:
        # an invented plot type (none of PLOT_TYPES), then the error and the fix
        bad['plot_type'] = rng.choice(INVENTED_PLOTS)
        return bad if bad != args else None
    if choice == 'guess' and name == 'set_fields' and bad.get('changes'):
        change = dict(bad['changes'][0])
        change['key'] = {'temp': 'temperature', 'r': 'radius', 'scan_speed': 'speed',
                         'timestep': 'dt'}.get(change['key'], change['key'] + '_value')
        bad['changes'] = [change, *bad['changes'][1:]]
        return bad if bad != args else None
    if choice == 'stringify' and isinstance(value, list) and value and \
            all(isinstance(x, (int, float)) for x in value):
        bad[key] = ', '.join(f'{x:g}' for x in value)
    elif choice == 'nest' and name in ('set_fields', 'propose_plot') and isinstance(value, list):
        item = value[0] if value and isinstance(value[0], dict) else None
        if item is None or 'key' not in item:
            return None
        bad = {**{k: v for k, v in bad.items() if k != key},
               item['key']: item.get('value')}
    elif choice == 'drop':
        del bad[key]
    else:  # rename: a near-miss name a model might guess
        new = {'loads_nN': 'loads', 'pressures_GPa': 'pressures', 'materials': 'material',
               'potential_type': 'potential', 'plot_type': 'type', 'datasets': 'dataset',
               'descriptor': 'level', 'quantity': 'name', 'query': 'q', 'path': 'folder',
               'material_a': 'materials', 'changes': 'fields', 'start': 'min',
               'stop': 'max'}.get(key, key + 's')
        bad[new] = bad.pop(key)
    return bad if bad != args else None


@dataclass
class Turn:
    """One user request and the assistant's work on it."""

    category: str
    request: str
    trace: List[Tuple[str, dict, str, bool]]  # name, args, result, trained on
    reply: str = ''


def setup_scene(category: str, scene: Scene) -> Optional[str]:
    """Category-specific slot values; a reason when the scene cannot be built."""
    rng, cat = scene.rng, BY_NAME.get(category, CONCEPT)
    if 'parameter' in cat.slots:
        if rng.random() >= ANY_FIELD_RATE or not field_slot(scene, 'config'):
            parameter_slot(scene)
    if 'global_setting' in cat.slots:
        if rng.random() >= ANY_FIELD_RATE or not field_slot(scene, 'settings'):
            global_setting_slot(scene)
    if 'setting' in cat.slots:
        scene.slots['setting'] = rng.choice(SETTINGS_TO_DECLINE)
    if category == 'ask_unknown_material':
        scene.slots['material'] = scene.values['unknown']
    if category == 'ask_unknown_potential' and not missing_potential(scene):
        return 'material has every potential type'
    if category == 'speed_sweep':
        scene.slots['start'], scene.slots['stop'] = (f'{x:g}' for x in scene.values['speeds'])
    if category == 'log_sweep':
        scene.slots['start'] = str(scene.values['log_start'])
        scene.slots['stop'] = str(scene.values['log_stop'])
    return None


def play(category: str, template: Optional[str], scene: Scene, toolbox: Toolbox,
         with_mistake: bool) -> Tuple[Optional[Turn], Optional[str]]:
    """Run one turn: the request, its tool calls (one perhaps wrong first), the reply."""
    # pylint: disable=too-many-return-statements
    why = setup_scene(category, scene)
    if why:
        return None, why
    source = ''
    if category == 'concept':
        request, source = concept_question(scene.rng, load_knowledge())
        scene.values['question'] = request
        if len(request.split()) < 4:
            return None, 'no question'
    else:
        request = template.format(**scene.slots)
    if too_close_to_eval(request):
        return None, 'too close to an eval prompt'
    prepare_world(category, scene)
    turn = Turn(category, request, [])
    calls = []

    def run(name: str, args: dict) -> str:
        calls.append(name)
        if with_mistake and len(calls) == 1 + scene.values.get('mistake_at', 0):
            bad = mistake(name, args, scene.rng)
            error = toolbox.retry_text(scene.deps, name, bad) if bad else None
            if error:  # the real rejection the runtime would send back
                turn.trace.append((name, bad, error, False))
        result = toolbox.call(scene.deps, name, args)
        turn.trace.append((name, args, result, True))
        return result

    try:
        plan_for(category, scene)(run)
    except (ToolInputError, ValueError) as exc:
        return None, f'tool failed: {str(exc)[:80]}'
    if category == 'concept' and source not in turn.trace[-1][2]:
        return None, 'search missed the source passage'
    cat = BY_NAME.get(category, CONCEPT)
    plain = [(n, a, r) for n, a, r, _ in turn.trace]
    turn.reply = teacher_reply(request, cat.kind, plain, category)
    why = check(category, cat.kind, scene, request, turn.reply,
                [(n, a, r) for n, a, r, ok in turn.trace if ok])
    return (None, why) if why else (turn, None)


def to_messages(turns: List[Turn]) -> List[dict]:
    """The conversation in the OpenAI/Qwen format; a rejected call is kept with its
    error but marked ``train: false``, so the loss never rewards the mistake."""
    # The prompt the runtime gives Ollama models starts with /no_think (prompts.py).
    messages = [{'role': 'system', 'content': f'/no_think\n{ROLE}'}]
    n = 0
    for turn in turns:
        messages.append({'role': 'user', 'content': turn.request})
        for name, args, result, trained in turn.trace:
            call_id = f'call_{n}'
            n += 1
            call = {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': call_id, 'type': 'function',
                'function': {'name': name, 'arguments': args}}]}
            if not trained:
                call['train'] = False
            messages.append(call)
            messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name,
                             'content': result})
        messages.append({'role': 'assistant', 'content': turn.reply})
    return messages


def conversation(category: str, template: Optional[str], rng: random.Random,
                 toolbox: Toolbox, pairs, settings,
                 templates: Optional[Dict[str, List[str]]] = None
                 ) -> Tuple[Optional[dict], Optional[str]]:
    """One conversation: a turn, sometimes with a mistake, sometimes a follow-up."""
    with tempfile.TemporaryDirectory() as tmp:
        deps = make_deps(Path(tmp), settings, model=DOCUMENT_MODEL.get(category, 'afm'))
        deps.direct_apply = rng.random() < 0.5
        deps.knowledge = load_knowledge  # the installed corpus, as the GUI passes it
        scene = Scene(rng, deps)
        fill(scene, pairs)
        scene.values['mistake_at'] = rng.randint(0, 1)
        with_mistake = rng.random() < MISTAKE_RATE
        first, why = play(category, template, scene, toolbox, with_mistake)
        if first is None:
            return None, why
        turns = [first]
        followups = [c for c in FOLLOWUPS if templates and templates.get(c)]
        if category in FIRST_TURNS and followups and rng.random() < FOLLOWUP_RATE:
            follow = rng.choice(followups)
            scene.deps.pending.clear()  # the first proposal was shown; judge the second alone
            second, why = play(follow, rng.choice(templates[follow]), scene, toolbox,
                               rng.random() < MISTAKE_RATE)
            if second is None:
                return None, f'follow-up {follow}: {why}'
            turns.append(second)
        cat = BY_NAME.get(category, CONCEPT)
        return {'messages': to_messages(turns), 'category': category, 'kind': cat.kind,
                'turns': [t.category for t in turns],
                'mistakes': sum(not ok for t in turns for *_, ok in t.trace),
                'template': template or f'concept from: {first.trace[-1][1]}',
                'source': 'B', 'teacher': TEACHER, 'teacher_licence': TEACHER_LICENCE}, None


def load_templates(drafts: bool) -> Dict[str, List[str]]:
    data = yaml.safe_load((DRAFTS if drafts else TEMPLATES).read_text())
    templates = {c['category']: c['templates'] for c in data['categories']
                 if c['templates'] and c['category'] not in UNSUPPORTED}
    if load_knowledge() is not None:
        templates['concept'] = [None]  # questions come from corpus passages
    return templates


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--name', default='smoke')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drafts', action='store_true',
                        help='use the unreviewed teacher drafts (smoke runs only)')
    parser.add_argument('--weights', default='',
                        help='override sampling weights, e.g. concept=8,lattice_pair=3')
    args = parser.parse_args(argv)
    for item in filter(None, args.weights.split(',')):
        name, weight = item.split('=')
        WEIGHTS[name] = float(weight)
    rng = random.Random(args.seed)
    templates = load_templates(args.drafts)
    toolbox = Toolbox.load()
    settings = GlobalSettings()
    with tempfile.TemporaryDirectory() as tmp:
        pairs = usable_materials(make_deps(Path(tmp), settings))
    DATA.mkdir(exist_ok=True)
    kept, rejects, per_category = 0, Counter(), Counter()
    out_path = DATA / f'{args.name}.jsonl'
    with out_path.open('w', encoding='utf-8') as out:
        attempts = 0
        while kept < args.n and attempts < args.n * 3:
            attempts += 1
            names = sorted(templates)
            category = rng.choices(names, weights=[WEIGHTS.get(n, 1) for n in names])[0]
            template = rng.choice(templates[category])
            try:
                record, why = conversation(category, template, rng, toolbox, pairs, settings,
                                           templates)
            except (KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
                # a bad template, a teacher timeout or unparseable teacher JSON
                record, why = None, f'{type(exc).__name__}: {str(exc)[:60]}'
            if record is None:
                rejects[f'{category}: {why}'] += 1
                continue
            record['id'] = f'{args.name}-{kept:05d}'
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
            per_category[category] += 1
            if sys.stderr.isatty():
                print(f'\r{kept}/{args.n} kept, {attempts} tried', end='', file=sys.stderr)
    card = {'name': args.name, 'conversations': kept, 'attempts': attempts,
            'schema_hash': toolbox.schema_hash, 'knowledge_hash': knowledge_hash(),
            'templates': 'drafts (unreviewed)' if args.drafts else 'templates.yaml',
            'sources': {'B': kept}, 'teacher': TEACHER, 'teacher_licence': TEACHER_LICENCE,
            'weights': dict(WEIGHTS), 'per_category': dict(per_category), 'rejects': dict(rejects.most_common()),
            'tools': toolbox.schemas}
    (DATA / f'{args.name}.card.json').write_text(json.dumps(card, indent=1))
    print(f'{kept} conversations in {attempts} attempts -> {out_path.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
