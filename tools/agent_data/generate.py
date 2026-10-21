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
from src.agent.prompts import KNOWLEDGE_DIR, ROLE  # noqa: E402
from src.agent.tools.materials import index_for, resolve_material  # noqa: E402
from src.agent.tools.setup import make_sweep  # noqa: E402
from src.core.config import GlobalSettings  # noqa: E402
from src.core.scaling import ScaleContext, scale  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.agent.evals.cases import grounded, numbers_in, quotes  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402
from tests.helpers.agent_eval import EvalResult  # noqa: E402
from tests.helpers.results_tree import make_family_run  # noqa: E402
from categories import BY_NAME  # noqa: E402

TEMPLATES = HERE / 'sources' / 'templates.yaml'
DRAFTS = HERE / 'sources' / 'templates.draft.yaml'
DATA = HERE / 'data'
OLLAMA = 'http://127.0.0.1:11434/api/chat'
TEACHER = 'qwen3:14b'
TEACHER_LICENCE = 'Apache-2.0'
#: Categories the generator cannot build yet: Materials Project answers need real
#: database rows (no key here, and invented rows would teach invented facts).
UNSUPPORTED = {'materials_project'}

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
        from src.agent.runtime import AgentSession
        from src.agent.settings import AgentSettings
        session = AgentSession(AgentSettings(), model=FunctionModel(lambda m, i: None))
        tools = dict(session.agent._function_toolset.tools)
        schemas = [{'type': 'function', 'function': {
            'name': name, 'description': tool.tool_def.description or '',
            'parameters': tool.tool_def.parameters_json_schema}}
            for name, tool in sorted(tools.items())]
        digest = hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()
        return cls(tools, schemas, digest[:16])

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
    if category == 'plot_cof_load':
        return static([('find_result_sets', {}), ('plot_schema', {'plot_type': 'summary'}),
                       ('propose_plot', {'request': None})])
    if category == 'lattice_pair':
        return static([('lattice_mismatch', {'material_a': v['material'],
                                             'material_b': v['material_b']})])
    if category == 'lattice_one':
        return static([('lattice_mismatch', {'material_a': v['material']})])
    if category == 'ask_unknown_material':
        return static([('search_structures', {'query': v['unknown']})])
    if category == 'ask_which_family':
        return static([('list_modes', {})])
    return static([])  # ask_wrong_unit, decline_execution_setting: no tool


def prepare_world(category: str, scene: Scene) -> None:
    """Workspace fixtures a category needs before its tools run."""
    if category in ('summarise_dataset', 'plot_cof_load'):
        root = scene.deps.roots.workspace / scene.values['dataset']
        for material in (scene.values['material'], scene.values['material_b']):
            make_family_run(root, 'afm', material=material, loads=(10.0, 30.0))
        DataReader(results_dir=str(root)).export_full_data_to_json()
    if category == 'list_workspace':
        for name in scene.rng.sample(['campaign_a', 'notes', 'runs_2026', 'old_tests',
                                      'mos2_loads', 'figures'], 3):
            (scene.deps.roots.workspace / name / 'run1').mkdir(parents=True)


# ---- teacher reply and checks ----------------------------------------------------

def teacher_reply(request: str, kind: str, trace: List[Tuple[str, dict, str]]) -> str:
    lines = [f'- {name}({json.dumps(args)}) ->\n{result}' for name, args, result in trace]
    prompt = REPLY_BRIEF.format(rules=ROLE.split('Rules:', 1)[1].strip(),
                                kind_rule=KIND_RULES[kind], request=request,
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
    why = status_error(reply, trace)
    if why:
        return why
    if kind == 'ask':
        return None if '?' in reply and not scene.deps.pending else 'did not ask'
    if kind == 'declines':
        return None if not scene.deps.pending else 'proposed on a decline'
    if category.endswith('_setup') or category in ('afm_two_materials', 'sheet_angles',
                                                    'linear_sweep', 'load_band'):
        if not scene.deps.pending:
            return 'no proposal'
    if 'expected' in scene.values and not all(quotes(result, x) or f'{x:g}' in reply
                                              for x in scene.values['expected']):
        return 'expected values not quoted'
    if category == 'find_parameter' and scene.values['key'] not in reply:
        return 'parameter key not named'
    return None


# ---- one conversation ------------------------------------------------------------

def conversation(category: str, template: str, rng: random.Random, toolbox: Toolbox,
                 pairs, settings) -> Tuple[Optional[dict], Optional[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        deps = make_deps(Path(tmp), settings)
        deps.direct_apply = rng.random() < 0.5
        scene = Scene(rng, deps)
        cat = BY_NAME[category]
        fill(scene, pairs)
        if 'parameter' in cat.slots:
            parameter_slot(scene)
        if 'setting' in cat.slots:
            scene.slots['setting'] = rng.choice(SETTINGS_TO_DECLINE)
        if category == 'ask_unknown_material':
            scene.slots['material'] = scene.values['unknown']
        request = template.format(**scene.slots)
        prepare_world(category, scene)
        trace: List[Tuple[str, dict, str]] = []

        def run(name: str, args: dict) -> str:
            if name == 'propose_plot':
                dataset = f"workspace:/{scene.values['dataset']}"
                args = {'plot_type': 'summary', 'datasets': [dataset],
                        'fields': [{'key': 'x_axis', 'value': 'force'},
                                   {'key': 'y_axis', 'value': 'cof'}]}
            result = toolbox.call(deps, name, args)
            trace.append((name, args, result))
            return result

        try:
            plan_for(category, scene)(run)
        except (ToolInputError, ValueError) as exc:
            return None, f'tool failed: {str(exc)[:80]}'
        reply = teacher_reply(request, cat.kind, trace)
        why = check(category, cat.kind, scene, request, reply, trace)
        if why:
            return None, why
        # The prompt the runtime gives Ollama models starts with /no_think (prompts.py).
        messages = [{'role': 'system', 'content': f'/no_think\n{ROLE}'},
                    {'role': 'user', 'content': request}]
        for i, (name, args, result) in enumerate(trace):
            call_id = f'call_{i}'
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': call_id, 'type': 'function',
                'function': {'name': name, 'arguments': args}}]})
            messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name,
                             'content': result})
        messages.append({'role': 'assistant', 'content': reply})
        return {'messages': messages, 'category': category, 'kind': cat.kind,
                'template': template, 'source': 'B', 'teacher': TEACHER,
                'teacher_licence': TEACHER_LICENCE}, None


def load_templates(drafts: bool) -> Dict[str, List[str]]:
    data = yaml.safe_load((DRAFTS if drafts else TEMPLATES).read_text())
    return {c['category']: c['templates'] for c in data['categories']
            if c['templates'] and c['category'] not in UNSUPPORTED}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--name', default='smoke')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drafts', action='store_true',
                        help='use the unreviewed teacher drafts (smoke runs only)')
    args = parser.parse_args(argv)
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
            category = rng.choice(sorted(templates))
            template = rng.choice(templates[category])
            try:
                record, why = conversation(category, template, rng, toolbox, pairs, settings)
            except (KeyError, IndexError, OSError) as exc:  # a bad template or teacher timeout
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
            'per_category': dict(per_category), 'rejects': dict(rejects.most_common()),
            'tools': toolbox.schemas}
    (DATA / f'{args.name}.card.json').write_text(json.dumps(card, indent=1))
    print(f'{kept} conversations in {attempts} attempts -> {out_path.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
