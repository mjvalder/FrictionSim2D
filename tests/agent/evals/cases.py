"""The eval cases: a prompt, the fixture it runs on, and a check computed by code.

Every expected value comes from code -- `make_sweep`, `scale`,
`resolve_material`, the material index, the fixture dataset's JSON or the fake
materials client's rows -- never typed by hand. Values the user states in a
prompt are module constants used both to write the prompt and to check it.

`verdict` is the single pass/fail rule. Must-ask cases (``kind='ask'``) pass
only with no proposal AND a question mark in the reply (plus their own check,
if any); declining cases (``kind='declines'``: a request the assistant must
refuse) need no proposal and their check, but no question. Every other case
has a positive check, and
``tests/agent/test_eval_cases.py`` proves none of them passes on a turn that
proposed nothing and only asked back.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple

from src.agent.changes import ChangeSet, FsChangeSet, ToolInputError
from src.agent.knowledge_index import load_knowledge
from src.agent.materials_db import NO_KEY, MaterialsDBError
from src.agent.plot_changes import PlotChange
from src.agent.tools.materials import index_for, resolve_material
from src.agent.tools.setup import make_sweep
from src.core.scaling import ScaleContext, scale
from src.core.utils.structures import cifread
from src.postprocessing.plot_config import PLOT_TYPES
from src.postprocessing.read_data import DataReader
from tests.helpers.agent import FakeMaterialsClient, materials_factory
from tests.helpers.results_tree import make_family_run

# ---- fixtures ------------------------------------------------------------------

#: The fake Materials Project rows (formula MoS2) and 2DMatPedia rows.
MP_ROWS = (
    {'material_id': 'mp-2815', 'formula': 'MoS2', 'e_above_hull': 0.0,
     'formation_energy': -0.94, 'band_gap': 1.23, 'is_metal': False, 'density': 5.0,
     'bulk_modulus': 52.0, 'shear_modulus': 31.0, 'spacegroup': 'P6_3/mmc'},
    {'material_id': 'mp-1434', 'formula': 'MoS2', 'e_above_hull': 0.004,
     'formation_energy': -0.93, 'band_gap': 0.87, 'is_metal': False, 'density': 5.0,
     'bulk_modulus': 48.0, 'shear_modulus': 29.0, 'spacegroup': 'R3m'},
)
DM_ROWS = (
    {'dm_id': '2dm-3', 'mp_id': 'mp-2815', 'formula': 'MoS2', 'structure_id': 'sid3',
     'process': 'top-down exfoliation', 'E': -7.1, 'Ed': 0.07, 'Ex': 0.02, 'dE': 0.01,
     'Evdw': -0.3, 'mu': 0.0},
    {'dm_id': '2dm-17', 'mp_id': 'mp-1434', 'formula': 'MoS2', 'structure_id': 'sid17',
     'process': 'top-down exfoliation', 'E': -7.0, 'Ed': 0.09, 'Ex': 0.03, 'dE': 0.02,
     'Evdw': -0.31, 'mu': 0.0},
)
DATASET = 'campaign'
DATASET_MATERIALS = ('h-MoS2', 'h-WS2')
WORKSPACE_FOLDERS = ('campaign_alpha', 'notes_2025')


def fake_materials_db(deps) -> None:
    """Every case gets this: the fake client, so no case reaches Materials Project."""
    deps.materials_db = materials_factory(FakeMaterialsClient(mp_rows=MP_ROWS,
                                                              dm_rows=DM_ROWS))


#: The fixture knowledge corpus: eval-only text (never trained on), with a title
#: a reply can only know from the search result.
KNOWLEDGE = (
    {'id': 'k:1', 'title': 'Structural lubricity', 'section': 'Mechanism',
     'url': 'https://example.org/structural-lubricity', 'licence': 'CC-BY-4.0',
     'text': 'Superlubricity between crystalline surfaces arises when the two lattices '
             'are incommensurate: the lateral forces on individual atoms point in many '
             'directions and largely cancel, so the energy barrier to sliding nearly '
             'vanishes. Twisting one layer can switch a commensurate contact into an '
             'incommensurate one.'},
    {'id': 'k:2', 'title': 'Puckering of thin sheets', 'section': 'Layer dependence',
     'url': 'https://example.org/puckering', 'licence': 'CC-BY-4.0',
     'text': 'Friction on atomically thin sheets often decreases as layers are added, '
             'because a thin, weakly bound sheet puckers in front of a sliding tip and '
             'increases the contact area.'},
)


def knowledge_corpus(deps) -> None:
    """Every case gets this: a two-passage corpus beside the workspace."""
    path = deps.roots.workspace.parent / 'knowledge' / 'passages.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(p) + '\n' for p in KNOWLEDGE), encoding='utf-8')
    deps.knowledge = lambda: load_knowledge(path)


def no_key_materials_db(deps) -> None:
    """Materials Project lookups switched on but without a key."""
    def factory(should_stop=None):  # pylint: disable=unused-argument
        raise MaterialsDBError(NO_KEY)
    deps.materials_db = factory


def results_dataset(deps) -> None:
    """``workspace/campaign``: two AFM materials at two loads, exported as a dataset."""
    root = deps.roots.workspace / DATASET
    for material in DATASET_MATERIALS:
        make_family_run(root, 'afm', material=material, loads=(10.0, 30.0))
    DataReader(results_dir=str(root)).export_full_data_to_json()


def workspace_folders(deps) -> None:
    """A few named folders in the workspace."""
    for name in WORKSPACE_FOLDERS:
        (deps.roots.workspace / name / 'run1').mkdir(parents=True)


def dataset_runs(deps) -> List[dict]:
    """The fixture dataset's runs, read back from its JSON."""
    files = sorted((deps.roots.workspace / DATASET / 'outputs').glob('output_full_*.json'))
    return [run for f in files for run in json.loads(f.read_text())['runs']]


# ---- reading a result ------------------------------------------------------------

def norm(text: Any) -> str:
    """Case, dashes, underscores and spaces ignored."""
    return re.sub(r'[\s_\-]', '', str(text)).casefold()


def mentions(result, *words: str) -> bool:
    """Whether the reply names every word (normalised)."""
    reply = norm(result.text)
    return bool(words) and all(norm(w) in reply for w in words)


def mentions_any(result, *words: str) -> bool:
    """Whether the reply names at least one of ``words`` (normalised)."""
    reply = norm(result.text)
    return any(norm(w) in reply for w in words)


def config_changes(result) -> List[ChangeSet]:
    """Config/settings change sets."""
    return [p for p in result.pending if isinstance(p, ChangeSet)]


def sets(result, kind: str = 'set') -> dict:
    """``{(section, key): value}`` over every change set (later wins)."""
    return {(op.section, op.key): op.value for p in config_changes(result)
            for op in p.ops if op.kind == kind}


def target_model(result) -> Optional[str]:
    """The model the proposals leave the Builder on; None without a change set."""
    changes = config_changes(result)
    if not changes:
        return None
    switched = [op.value for p in changes for op in p.ops if op.kind == 'set_model']
    return switched[-1] if switched else result.deps.document.model


def materials(result) -> set:
    """The [2D] materials the proposals set (a sweep list or one ``mat``)."""
    for change in config_changes(result):
        for op in change.ops:
            if op.kind == 'materials' and op.value:
                return set(op.value)
    mat = sets(result).get(('2D', 'mat'))
    return {mat} if mat else set()


def fs_ops(result) -> List[Tuple[str, ...]]:
    """File operations as the card shows them (``workspace:/...``)."""
    return [op for p in result.pending if isinstance(p, FsChangeSet) for op in p.shown]


def plots(result) -> List[PlotChange]:
    """Plot proposals."""
    return [p for p in result.pending if isinstance(p, PlotChange)]


#: Relative tolerance for numbers: a model may round to 3 significant figures,
#: which still tells a linear sweep from a log one and one band from another.
REL_TOL = 2e-3
_NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.]*\d)')


def numbers_in(text: str) -> List[float]:
    """The standalone numbers in ``text`` ('MoS2' and '2026' give no 2 or 20)."""
    return [float(m) for m in _NUMBER.findall(text)]


def has_numbers(text: str, expected: Iterable[float]) -> bool:
    """Whether every expected value is one of the numbers in ``text``."""
    found = numbers_in(text)
    expected = list(expected)
    return bool(expected) and all(any(math.isclose(f, float(e), rel_tol=REL_TOL)
                                      for f in found) for e in expected)


#: A decimal standing alone: '1.23' and '0.5', not '1T', '2D', 'MoS2' or a '1.' list mark.
_DECIMAL = re.compile(r'(?<![\w.])\d+\.\d+(?![\w.]*\d)(?![A-Za-z])')


def _fixture_numbers() -> List[float]:
    """Every number in the fake database rows: what a lookup can legitimately return."""
    return [float(v) for row in MP_ROWS + DM_ROWS for v in row.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def grounded(result, extra: Iterable[float] = ()) -> bool:
    """Whether every decimal in the reply came from a tool result, the fixture rows
    or ``extra``, allowing the reply to round to its own printed precision.

    Integers are not checked: layer counts, polytypes (2H) and list numbers are
    integers, while invented property values (gaps, lattice constants, COFs)
    are decimals."""
    allowed = [*extra, *_fixture_numbers()]
    for event in result.events:
        if event.kind == 'tool_result':
            allowed.extend(numbers_in(event.payload or event.text))
    for token in _DECIMAL.findall(result.text):
        value, places = float(token), len(token.split('.')[1])
        if not any(math.isclose(a, value, rel_tol=REL_TOL) or round(a, places) == value
                   for a in allowed):
            return False
    return True


def same(value: Any, expected: Iterable[float]) -> bool:
    """``value`` is a list matching ``expected`` within `REL_TOL`
    (make_sweep's tool text prints 6 figures, the function returns 9)."""
    expected = list(expected)
    if not isinstance(value, (list, tuple)):
        value = [value]
    try:
        return len(value) == len(expected) and all(
            math.isclose(float(v), float(e), rel_tol=REL_TOL) for v, e in zip(value, expected))
    except (TypeError, ValueError):
        return False


def called(result, *tools: str) -> bool:
    """Whether the model called any of ``tools``."""
    return any(name in tools for name in result.tool_calls)


def setup_ok(result, model: str, names: Iterable[str], pot: str = 'sw') -> bool:
    """A setup proposal for ``model`` on ``names`` with potential type ``pot``.

    The potential type expected is the one `resolve_material` finds, so a
    fixture without that potential fails here loudly, not in the model.
    """
    names = set(names)
    expected_pot = {resolve_material(result.deps, n, pot)[1].pot_type for n in names}
    return (target_model(result) == model and materials(result) == names
            and {sets(result).get(('2D', 'pot_type'))} == expected_pot)


def band(model: str, descriptor: str, n: int) -> List[float]:
    """``n`` linear values across the reference band for ``descriptor`` loads."""
    scaled = scale('load', descriptor, ScaleContext(model=model))
    return make_sweep(scaled.low, scaled.high, n, 'linear')


def proposed_or_asked(result, key: Tuple[str, str], values: List[float]) -> bool:
    """Either ``key`` set to ``values``, or no proposal and a question quoting them."""
    if same(sets(result).get(key), values):
        return True
    return not result.pending and '?' in result.text and has_numbers(result.text, values)


def raises(func: Callable, *args) -> bool:
    """Whether ``func(*args)`` raises `ToolInputError` (a fixture precondition)."""
    try:
        func(*args)
    except ToolInputError:
        return True
    return False


# ---- the cases ---------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One prompt and how to judge the turn."""

    prompt: str
    check: Optional[Callable[[Any], bool]] = None
    #: Tools a good answer uses (documents intent; the coverage test reads it).
    tools: Tuple[str, ...] = ()
    #: 'act' (a positive check), 'ask' (no proposal + '?') or 'declines'
    #: (no proposal + the check).
    kind: str = 'act'
    model: str = 'afm'
    prepare: Tuple[Callable[[Any], None], ...] = ()


def verdict(case: Case, result) -> bool:
    """Pass or fail. A must-ask case needs no proposal and a '?'; a declining
    one no proposal and its check."""
    if case.kind == 'ask':
        return (not result.pending and '?' in result.text
                and (case.check is None or bool(case.check(result))))
    if case.kind == 'declines':
        return not result.pending and case.check is not None and bool(case.check(result))
    return case.check is not None and bool(case.check(result))


LOADS = (10, 20, 30)
TWO_LOADS = (10, 20, 40)
ANGLES = (0, 30, 60)
LAYERS = (1, 2, 3)
PRESSURES = (0.5, 1, 2)
SPEED, RADIUS, TEMP, SIDE, GRID, SPRING = 1, 30, 150, 150, 12, 8
TIMESTEP, THERMO = 0.0005, 500


def _plot(result, plot_type: str, y_axis: str = '') -> bool:
    return any(p.plot['plot_type'] == plot_type
               and DATASET in [folder for folder, _ in p.folders]
               and y_axis in str(p.plot.get('y_axis') or '').lower() for p in plots(result))


def _fs(result, kind: str, suffix: str) -> bool:
    return any(op[0] == kind and op[-1].endswith(suffix) for op in fs_ops(result))


def material_core(name: str) -> str:
    """A material without its structure prefix: 'h_MoS2' -> 'MoS2'."""
    return re.sub(r'^[a-z][_-]', '', str(name))


def mentions_materials(result, names: Iterable[str]) -> bool:
    """Every material named, with or without its structure prefix."""
    return mentions(result, *{material_core(n) for n in names})


def count_by(result, number: int, word: str = 'complet') -> bool:
    """``number`` within four words before ``word``, or right after it."""
    n = rf'(?<![\w.]){number}(?![\w.]*\d)'
    text = result.text.replace('*', '')  # markdown bold round the number
    return re.search(rf'{n},?\s+(?:[\w-]+\s+){{0,4}}{word}|{word}\w*\W+(?:\w+\W+)?{n}',
                     text, re.IGNORECASE) is not None


def _complete_ws2(result) -> int:
    return sum(1 for run in dataset_runs(result.deps)
               if norm(run['material']) == norm('h-WS2') and run['status'] == 'complete')


def _stable_mp() -> dict:
    return min(MP_ROWS, key=lambda row: row['e_above_hull'])


def _dm_file(dm_id: str) -> str:
    row = next(r for r in DM_ROWS if r['dm_id'] == dm_id)
    return f"{re.sub(r'[^A-Za-z0-9]', '', row['formula'])}_{dm_id}.cif"


def _structure_names(result, query: str) -> List[str]:
    return [f.name for f in index_for(result.deps).structures if norm(query) in norm(f.name)]


SETUP = {
    'afm_single': Case(
        f"Set up an AFM simulation of h-MoS2 with an sw potential at "
        f"{', '.join(map(str, LOADS))} nN.",
        lambda r: setup_ok(r, 'afm', ['h-MoS2']) and same(sets(r).get(('general', 'force')),
                                                           LOADS),
        tools=('setup_afm', 'search_structures')),
    'afm_two_materials': Case(
        f"Set up an AFM simulation on h-MoS2 and h-WS2 with sw potentials at "
        f"{', '.join(map(str, TWO_LOADS))} nN.",
        lambda r: setup_ok(r, 'afm', ['h-MoS2', 'h-WS2'])
        and same(sets(r).get(('general', 'force')), TWO_LOADS),
        tools=('setup_afm',)),
    'afm_speed_radius': Case(
        f'AFM on h-WS2 with sw at 20 nN, tip radius {RADIUS} Angstrom, sliding at '
        f'{SPEED} m/s.',
        lambda r: setup_ok(r, 'afm', ['h-WS2']) and same(sets(r).get(('tip', 'r')), [RADIUS])
        and same(sets(r).get(('general', 'scan_speed')), [SPEED]),
        tools=('setup_afm',)),
    'afm_temperature': Case(
        f'Set up AFM sliding on h-MoSe2 with sw at 15 nN and {TEMP} K.',
        lambda r: setup_ok(r, 'afm', ['h-MoSe2']) and same(sets(r).get(('general', 'temp')),
                                                            [TEMP]),
        tools=('setup_afm',)),
    'afm_scan_angles': Case(
        f"AFM on h-MoS2 with sw at 20 nN, scanning at {', '.join(map(str, ANGLES))} degrees.",
        lambda r: setup_ok(r, 'afm', ['h-MoS2'])
        and same(sets(r).get(('general', 'scan_angle')), ANGLES),
        tools=('setup_afm',)),
    'afm_layers': Case(
        f"AFM on h-MoS2 with sw at 20 nN for {', '.join(map(str, LAYERS))} layers.",
        lambda r: setup_ok(r, 'afm', ['h-MoS2']) and same(sets(r).get(('2D', 'layers')), LAYERS),
        tools=('setup_afm',)),
    'afm_sheet_size': Case(
        f'AFM on h-MoS2 with sw at 20 nN on a {SIDE} x {SIDE} Angstrom sheet.',
        lambda r: setup_ok(r, 'afm', ['h-MoS2']) and same(sets(r).get(('2D', 'x')), [SIDE])
        and same(sets(r).get(('2D', 'y')), [SIDE]),
        tools=('setup_afm',)),
    'afm_from_sheet_doc': Case(
        'Set up an AFM simulation of h-MoS2 with sw at 20 nN.',
        lambda r: setup_ok(r, 'afm', ['h-MoS2'])
        and any(op.kind == 'set_model' for p in config_changes(r) for op in p.ops),
        tools=('setup_afm',), model='sheetonsheet'),
    'list_modes': Case(
        'What kinds of simulation can you set up for me? Just list them.',
        lambda r: called(r, 'list_modes') and mentions(r, 'AFM', 'sheet', 'indent', 'PES'),
        tools=('list_modes',)),
    'sheet_single': Case(
        'Set up a sheet-on-sheet simulation of h-MoS2 with sw potentials at 1 GPa.',
        lambda r: setup_ok(r, 'sheetonsheet', ['h-MoS2'])
        and same(sets(r).get(('general', 'pressure')), [1]),
        tools=('setup_sheet',)),
    'sheet_pressures': Case(
        f"Sheet-on-sheet sliding of h-WS2 with sw at {', '.join(map(str, PRESSURES))} GPa.",
        lambda r: setup_ok(r, 'sheetonsheet', ['h-WS2'])
        and same(sets(r).get(('general', 'pressure')), PRESSURES),
        tools=('setup_sheet',)),
    'indent_single': Case(
        'Set up an indentation (hold and retract) of h-MoS2 with sw at 10 and 20 nN.',
        lambda r: setup_ok(r, 'indent', ['h-MoS2'])
        and same(sets(r).get(('general', 'force')), [10, 20]),
        tools=('setup_indent',)),
    'pes_tip': Case(
        'Run a PES scan of the AFM tip over h-WS2 with sw potentials.',
        lambda r: setup_ok(r, 'pes_tip', ['h-WS2']),
        tools=('setup_pes_tip',)),
    'pes_sheet': Case(
        f'Map the potential energy surface of h-MoS2 sliding on itself (sheet over '
        f'sheet) with sw, on a {GRID} x {GRID} grid.',
        lambda r: setup_ok(r, 'pes_sheet', ['h-MoS2'])
        and same(sets(r).get(('pes', 'grid_n')), [GRID]),
        tools=('setup_pes_sheet',)),
}

PARAMS = {
    'change_speed': Case(
        'Change the sliding speed of the current AFM config to 5 m/s.',  # 2 is the default
        lambda r: same(sets(r).get(('general', 'scan_speed')), [5]),
        tools=('find_parameter', 'set_fields')),
    'change_temperature': Case(
        'Set the temperature of the current config to 250 K.',
        lambda r: same(sets(r).get(('general', 'temp')), [250]),
        tools=('set_fields',)),
    'change_tip_radius': Case(
        'Make the AFM tip radius 40 Angstrom in the current config.',
        lambda r: same(sets(r).get(('tip', 'r')), [40]),
        tools=('set_fields',)),
    'change_driving_spring': Case(
        f'Set the driving spring of the current AFM config to {SPRING} N/m.',
        lambda r: same(sets(r).get(('general', 'driving_spring')), [SPRING]),
        tools=('find_parameter', 'set_fields')),
    'settings_timestep': Case(
        f'In the global settings, set the MD timestep to {TIMESTEP} ps.',
        lambda r: same(sets(r, 'settings_set').get(('simulation', 'timestep')), [TIMESTEP]),
        tools=('find_parameter', 'set_fields')),
    'settings_thermo': Case(
        f'In the settings, print thermo output every {THERMO} steps.',
        lambda r: same(sets(r, 'settings_set').get(('simulation', 'thermo')), [THERMO]),
        tools=('find_parameter', 'set_fields')),
    'find_parameter_only': Case(
        "Which parameter sets the driving spring stiffness? Don't change anything.",
        lambda r: not r.pending and called(r, 'find_parameter')
        and mentions(r, 'driving_spring'),
        tools=('find_parameter',)),
    'explain_current': Case(
        "What's wrong with my current config?",
        lambda r: called(r, 'explain_current') and bool(r.text.strip()),
        tools=('explain_current',)),
}

SWEEPS = {
    'even_load_sweep': Case(
        'Set up AFM on h-MoS2 with sw at 5 loads evenly spaced from 10 to 50 nN.',
        lambda r: setup_ok(r, 'afm', ['h-MoS2'])
        and same(sets(r).get(('general', 'force')), make_sweep(10, 50, 5)),
        tools=('make_sweep', 'setup_afm')),
    'log_load_sweep': Case(
        'Set up AFM on h-MoS2 with sw at 5 loads log-spaced from 1 to 100 nN.',
        lambda r: same(sets(r).get(('general', 'force')), make_sweep(1, 100, 5, 'log')),
        tools=('make_sweep', 'setup_afm')),
    'speed_sweep': Case(
        'For the current AFM config, sweep the sliding speed over 4 evenly spaced values '
        'from 0.5 to 2 m/s.',
        lambda r: same(sets(r).get(('general', 'scan_speed')), make_sweep(0.5, 2, 4)),
        tools=('make_sweep', 'set_fields')),
    'low_loads': Case(
        'Set up AFM on h-MoS2 with sw at 3 low loads.',
        lambda r: proposed_or_asked(r, ('general', 'force'), band('afm', 'low', 3)),
        tools=('suggest_values', 'setup_afm')),
    'high_loads': Case(
        'Set up AFM on h-WS2 with sw at 4 high loads.',
        lambda r: proposed_or_asked(r, ('general', 'force'), band('afm', 'high', 4)),
        tools=('suggest_values', 'setup_afm')),
    'moderate_indent_loads': Case(
        'Set up an indentation of h-MoS2 with sw at 3 moderate loads.',
        lambda r: proposed_or_asked(r, ('general', 'force'), band('indent', 'mid', 3)),
        tools=('suggest_values', 'setup_indent')),
}

MATERIALS = {
    'search_structures': Case(
        'Which MoS2 structure files can I use? Search the structure library and just '
        'list them.',
        lambda r: called(r, 'search_structures')
        and any(mentions(r, name) for name in _structure_names(r, 'MoS2')),
        tools=('search_structures',)),
    'search_potentials': Case(
        'Which sw potential files are there for h-WS2?',
        lambda r: called(r, 'search_potentials')
        and mentions(r, resolve_material(r.deps, 'h-WS2', 'sw')[1].path.name),
        tools=('search_potentials',)),
    'copy_structure_in': Case(
        'Copy the h-MoS2 structure file into my workspace.',
        lambda r: _fs(r, 'copy', resolve_material(r.deps, 'h-MoS2', 'sw')[0].path.name),
        tools=('search_structures', 'propose_copy_in')),
    'copy_potential_in': Case(
        "Copy the sw potential file for h-WS2 into my workspace's potentials folder.",
        lambda r: _fs(r, 'copy', 'potentials/'
                      + resolve_material(r.deps, 'h-WS2', 'sw')[1].path.name),
        tools=('search_potentials', 'propose_copy_in')),
}

WORKSPACE = {
    'workspace_overview': Case(
        'What folders are in my workspace?',
        lambda r: called(r, 'workspace_overview') and mentions(r, *WORKSPACE_FOLDERS),
        tools=('workspace_overview',), prepare=(workspace_folders,)),
    'make_folder': Case(
        'Create a folder called runs_2026 in my workspace.',
        lambda r: _fs(r, 'mkdir', 'runs_2026'),
        tools=('propose_mkdir',)),
    'describe_folder': Case(
        f'What files are in the workspace folder {DATASET}?',
        lambda r: called(r, 'describe_run', 'summarise_results')
        and (mentions(r, 'output_full') or mentions_materials(r, DATASET_MATERIALS)),
        tools=('describe_run',), prepare=(results_dataset,)),
}

ANALYSIS = {
    'find_datasets': Case(
        'Which result datasets are in my workspace?',
        lambda r: called(r, 'find_result_sets') and mentions(r, DATASET),
        tools=('find_result_sets',), prepare=(results_dataset,)),
    'summarise_dataset': Case(
        f'Summarise the results in {DATASET}: which materials are in it?',
        lambda r: called(r, 'summarise_results')
        and mentions_materials(r, {run['material'] for run in dataset_runs(r.deps)}),
        tools=('summarise_results',), prepare=(results_dataset,)),
    'count_complete_runs': Case(
        f'How many h-WS2 runs in {DATASET} completed?',
        lambda r: called(r, 'summarise_results') and count_by(r, _complete_ws2(r)),
        tools=('summarise_results',), prepare=(results_dataset,)),
    'plot_types': Case(
        'What kinds of plots can you make? Just list them.',
        lambda r: sum(t in r.text or t.replace('_', ' ') in r.text.lower()
                      for t in PLOT_TYPES) >= 3,
        tools=('plot_types',)),
    'plot_cof_vs_load': Case(
        f'Plot COF against normal load for the {DATASET} dataset.',
        lambda r: _plot(r, 'summary', y_axis='cof'),
        tools=('find_result_sets', 'plot_schema', 'propose_plot'), prepare=(results_dataset,)),
    # cof_histogram draws a bar per dataset of the lateral-vs-normal force slope
    # (plot_config.CofHistogramPlot), not a histogram of COF values; the prompt
    # asks for exactly that, so only cof_histogram is right.
    'plot_slope_bars': Case(
        f'Make a bar chart of the COF-vs-load slope per dataset for {DATASET}.',
        lambda r: _plot(r, 'cof_histogram'),
        tools=('plot_schema', 'propose_plot'), prepare=(results_dataset,)),
    'plot_timeseries': Case(
        f'Plot the lateral force against time for the runs in {DATASET}.',
        lambda r: _plot(r, 'timeseries'),
        tools=('plot_schema', 'propose_plot'), prepare=(results_dataset,)),
    'plot_rank': Case(
        f'Rank the materials in {DATASET} by friction in a plot.',
        lambda r: _plot(r, 'rank_friction'),
        tools=('plot_schema', 'propose_plot'), prepare=(results_dataset,)),
    'plot_filtered_material': Case(
        f'Plot COF against load for h-WS2 only, from {DATASET}.',
        lambda r: any(p.plot['plot_type'] == 'summary'
                      and norm('h-WS2') in map(norm, p.plot.get('filter_materials') or [])
                      for p in plots(r)),
        tools=('plot_schema', 'propose_plot'), prepare=(results_dataset,)),
}

MATERIALS_DB = {
    'mp_band_gap': Case(
        'What band gap does Materials Project give for the most stable MoS2 entry?',
        lambda r: called(r, 'query_materials_project')
        and has_numbers(r.text, [_stable_mp()['band_gap']]),
        tools=('query_materials_project',)),
    'mp_most_stable': Case(
        'Which MoS2 entry on Materials Project is the most stable? Give its id.',
        lambda r: called(r, 'query_materials_project')
        and mentions(r, _stable_mp()['material_id']),
        tools=('query_materials_project',)),
    'dm_list': Case(
        'List the MoS2 monolayers in 2DMatPedia with their ids.',
        lambda r: called(r, 'query_2dmatpedia')
        and mentions(r, *(row['dm_id'] for row in DM_ROWS)),
        tools=('query_2dmatpedia',)),
    'dm_fetch': Case(
        f"Save the 2DMatPedia structure {DM_ROWS[1]['dm_id']} into my workspace.",
        lambda r: _fs(r, 'write_text', _dm_file(DM_ROWS[1]['dm_id'])),
        tools=('fetch_2dmatpedia_structure',)),
    'mp_without_key': Case(
        'What band gap does Materials Project give for MoS2?',
        lambda r: not r.pending and called(r, 'query_materials_project')
        and 'key' in r.text.lower(),
        tools=('query_materials_project',), prepare=(no_key_materials_db,)),
}

MUST_ASK = {
    'ambiguous_unit': Case('Set up AFM on h-MoS2 with sw at loads from 10 to 50 nm.',
                           kind='ask', tools=('setup_afm',)),
    'unknown_material': Case('Set up an AFM simulation on unobtainium.', kind='ask',
                             tools=('search_structures',)),
    'unknown_potential': Case(
        'Set up AFM on h-MoS2 with a tersoff potential at 20 nN.', kind='ask',
        check=lambda r: raises(resolve_material, r.deps, 'h-MoS2', 'tersoff'),
        tools=('search_potentials',)),
    'ambiguous_family': Case('Set up a friction simulation of h-MoS2 with sw.', kind='ask',
                             tools=('list_modes',)),
    'sheet_force_unit': Case('Set up sheet-on-sheet sliding of h-MoS2 with sw at 10 nN.',
                             kind='ask', tools=('setup_sheet',)),
    'sheet_low_pressure': Case(
        'Set up sheet-on-sheet sliding of h-MoS2 with sw at 3 low pressures.', kind='ask',
        check=lambda r: scale('pressure', 'low', ScaleContext(model='sheetonsheet')) is None,
        tools=('suggest_values',), model='sheetonsheet'),
    'missing_dataset': Case('Plot COF against load for my experiment_x dataset.', kind='ask',
                            tools=('find_result_sets',)),
}

DECLINES = {
    'executed_setting_walltime': Case(
        'Set the HPC walltime to 48 hours.', kind='declines',
        check=lambda r: mentions(r, 'walltime') or mentions(r, 'settings'),
        tools=('set_fields',)),
    'executed_setting_minimize': Case(
        "In the settings, change the minimization command to 'minimize 0 1e-8 1000 10000'.",
        kind='declines', check=lambda r: mentions(r, 'minimiz'), tools=('set_fields',)),
    'negative_load': Case(
        'Set up AFM on h-MoS2 with sw at -5 nN.', kind='declines',
        check=lambda r: 'positive' in r.text.lower() or '?' in r.text,
        tools=('setup_afm',)),
}

def _lattice_a(result, name: str) -> float:
    """In-plane ``a`` of ``name``, read from the CIF the material index finds."""
    found = next(f for f in index_for(result.deps).structures if f.name == name)
    return cifread(found.path)['lat_a']


def _mismatch_a(result, first: str, second: str) -> float:
    a_1, a_2 = _lattice_a(result, first), _lattice_a(result, second)
    return abs(a_1 - a_2) / ((a_1 + a_2) / 2) * 100


def quotes(result, value: float) -> bool:
    """Whether the reply states ``value``, possibly rounded to the precision it prints."""
    return any(math.isclose(v, value, rel_tol=REL_TOL)
               or round(value, len(t.split('.')[1])) == v
               for t in _DECIMAL.findall(result.text) for v in [float(t)])


def _concept(*stem_groups: Tuple[str, ...]):
    """A concept answer: no proposal, nothing invented, one stem from every group."""
    return lambda r: (not r.pending and grounded(r) and bool(r.text.strip())
                      and all(mentions_any(r, *stems) for stems in stem_groups))


def _any_setup(result, models: Iterable[str], names: Iterable[str]) -> bool:
    return any(setup_ok(result, model, names) for model in models)


#: Domain cases (spec 2026-10-16 LoRA fine-tune §8): vocabulary and causes,
#: a goal turned into the right simulation, and numbers only from tools.
DOMAIN = {
    'concept_superlubricity': Case(
        'In one or two sentences: why can two incommensurate crystalline sheets slide '
        'past each other with very low friction?',
        _concept(('incommensura', 'mismatch'), ('cancel', 'corrugat', 'average', 'lock'))),
    'concept_prefix_t': Case(
        'What does the t- prefix in t-MoS2 mean?',
        _concept(('T-phase', 'T phase', 'octahedral', '1T'))),
    'concept_tmd': Case(
        'Is h-WS2 a transition-metal dichalcogenide? Answer in one sentence.',
        _concept(('dichalcogenide', 'TMD'), ('yes',))),
    'concept_cof': Case(
        'In these simulations, what is the coefficient of friction computed from?',
        _concept(('lateral', 'friction force'), ('normal',))),
    'concept_stick_slip': Case(
        'What is stick-slip motion in an AFM friction simulation?',
        _concept(('stick',), ('slip',), ('lattice', 'period', 'energy', 'spring'))),
    'cheap_corrugation': Case(
        'Before any sliding MD, I want a cheap comparison of how corrugated the energy '
        'landscape is for h-MoS2 and for h-WS2, each sliding on itself. Set it up with sw.',
        lambda r: _any_setup(r, ('pes_sheet', 'pes_tip'), ['h-MoS2', 'h-WS2']),
        tools=('list_modes', 'setup_pes_sheet')),
    'layer_dependence': Case(
        'Does friction under an AFM tip on h-MoS2 depend on how many layers there are? '
        f"Set up a run with sw at 20 nN that tests {LAYERS[0]} to {LAYERS[-1]} layers.",
        lambda r: setup_ok(r, 'afm', ['h-MoS2'])
        and same(sets(r).get(('2D', 'layers')), LAYERS),
        tools=('setup_afm',)),
    'direction_dependence': Case(
        'I want to know whether friction between two h-MoS2 sheets depends on the sliding '
        f"direction. Set it up with sw at 1 GPa, sliding at {', '.join(map(str, ANGLES))} "
        'degrees.',
        lambda r: setup_ok(r, 'sheetonsheet', ['h-MoS2'])
        and same(sets(r).get(('general', 'scan_angle')), ANGLES),
        tools=('setup_sheet',)),
    'gap_needs_lookup': Case(
        'What is the band gap of MoS2?',
        lambda r: called(r, 'query_materials_project') and grounded(r)
        and has_numbers(r.text, [_stable_mp()['band_gap']]),
        tools=('query_materials_project',)),
    'stable_polytype': Case(
        'Which MoS2 polytype is the most stable?',
        lambda r: called(r, 'query_materials_project', 'query_2dmatpedia') and grounded(r)
        and mentions_any(r, _stable_mp()['material_id'], _stable_mp()['spacegroup']),
        tools=('query_materials_project',)),
    'mismatch_pair': Case(
        'What is the lattice mismatch between h-MoS2 and h-WS2?',
        lambda r: called(r, 'lattice_mismatch') and grounded(r)
        and quotes(r, _mismatch_a(r, 'h-MoS2', 'h-WS2')),
        tools=('lattice_mismatch',)),
    'lattice_constant': Case(
        'What is the in-plane lattice constant of h-WS2?',
        lambda r: called(r, 'lattice_mismatch') and grounded(r)
        and quotes(r, _lattice_a(r, 'h-WS2')),
        tools=('lattice_mismatch',)),
    'cite_knowledge': Case(
        'Why does friction on thin sheets often drop as layers are added? '
        'Name the source you used.',
        lambda r: called(r, 'search_knowledge') and grounded(r) and not r.pending
        and mentions(r, KNOWLEDGE[1]['title']),
        tools=('search_knowledge',)),
    'cof_guess': Case(
        'Roughly what friction coefficient will I get for h-MoS2 under an AFM tip?',
        kind='declines',
        check=lambda r: grounded(r) and mentions_any(r, 'simulat', 'run ', 'measure')),
}

GROUPS = {'setup': SETUP, 'params': PARAMS, 'sweeps': SWEEPS, 'materials': MATERIALS,
          'workspace': WORKSPACE, 'analysis': ANALYSIS, 'materials_db': MATERIALS_DB,
          'must_ask': MUST_ASK, 'declines': DECLINES, 'domain': DOMAIN}
CASES = {name: case for group in GROUPS.values() for name, case in group.items()}
GROUP_OF = {name: group for group, cases in GROUPS.items() for name in cases}


def prepare_deps(case: Case, deps) -> None:
    """The fake materials client and the fixture corpus, then the case's own fixture."""
    fake_materials_db(deps)
    knowledge_corpus(deps)
    for step in case.prepare:
        step(deps)
