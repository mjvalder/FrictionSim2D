"""Conversation categories for the generator: what each asks, and its slots.

A category's ``brief`` is the instruction the local teacher drafts templates
from (spec §2: template wording comes from the teacher or the user, never
from Claude), so it describes the request and its slots and gives no example
wording. ``kind`` is how the assistant must respond: ``act`` (call tools and
propose), ``ask`` (ask back, propose nothing) or ``declines`` (refuse).
The generator (Task 8) maps each category to the tool calls it expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Category:
    """One kind of user request."""

    name: str
    brief: str
    slots: Tuple[str, ...]
    kind: str = 'act'
    #: The template must name one of these (lower case), so its family is clear.
    must_mention: Tuple[str, ...] = ()
    #: ... and none of these, e.g. a must-ask request may not name a family.
    must_not: Tuple[str, ...] = ()


#: What each slot holds, for the teacher (the generator fills them from real data).
SLOTS: Dict[str, str] = {
    'material': 'a 2D material name such as the user would type it',
    'material_b': 'a second 2D material name',
    'formula': 'a chemical formula without phase prefix',
    'potential': 'an interatomic potential type name',
    'loads': 'a comma-separated list of normal loads, numbers only',
    'load': 'one normal load, a number only',
    'pressures': 'a comma-separated list of normal pressures, numbers only',
    'pressure': 'one normal pressure, a number only',
    'n': 'how many values, a small integer',
    'start': 'the first value of a range, a number only',
    'stop': 'the last value of a range, a number only',
    'band': 'a qualitative load level: low, moderate or high',
    'temperature': 'a temperature, a number only',
    'angles': 'a comma-separated list of angles in degrees, numbers only',
    'grid': 'the number of grid points per side, an integer',
    'parameter': 'the name of one simulation parameter in plain words',
    'value': 'a value for that parameter, a number only',
    'folder': 'a folder name',
    'dataset': 'the name of a results dataset folder',
    'property': 'a material property a database lists',
    'wrong_unit': 'a unit that does not fit the quantity asked for',
    'setting': 'an HPC or LAMMPS execution setting in plain words',
}

AFM = ('afm', 'tip')
SHEET = ('sheet', 'layer', 'bilayer')

CATEGORIES: Tuple[Category, ...] = (
    Category('afm_setup', 'Ask for an AFM tip-sliding friction simulation of one material '
             'with a given potential at several normal loads in nN.',
             ('material', 'potential', 'loads'), must_mention=AFM),
    Category('afm_setup_temperature', 'Ask for an AFM sliding simulation of one material '
             'with a given potential at one load in nN and a temperature in K.',
             ('material', 'potential', 'load', 'temperature'), must_mention=AFM),
    Category('afm_two_materials', 'Ask for AFM sliding simulations on two materials with '
             'the same potential type at several loads in nN.',
             ('material', 'material_b', 'potential', 'loads'), must_mention=AFM),
    Category('sheet_setup', 'Ask for a sheet-on-sheet sliding simulation of one material '
             'on itself with a given potential at several normal pressures in GPa.',
             ('material', 'potential', 'pressures'), must_mention=SHEET),
    Category('sheet_angles', 'Ask whether friction between two sheets of one material '
             'depends on sliding direction, and to set up sheet-on-sheet sliding at one '
             'normal pressure in GPa and several sliding angles in degrees.',
             ('material', 'potential', 'pressure', 'angles'), must_mention=SHEET),
    Category('indent_setup', 'Ask for an indentation simulation (press, hold, retract, no '
             'sliding) of one material at several loads in nN.',
             ('material', 'potential', 'loads'), must_mention=('indent',)),
    Category('pes_tip_setup', 'Ask for a static potential-energy-surface scan of an AFM '
             'tip over one material.', ('material', 'potential'), must_mention=('tip',)),
    Category('pes_sheet_setup', 'Ask for a static corrugation (potential energy surface) '
             'scan of one sheet of a material over another sheet of the same material, '
             'with a given number of grid points per side.',
             ('material', 'potential', 'grid'), must_mention=SHEET),
    Category('linear_sweep', 'Ask for AFM on one material at N loads evenly spaced between '
             'two values in nN.', ('material', 'potential', 'n', 'start', 'stop'),
             must_mention=AFM),
    Category('load_band', 'Ask for AFM on one material at N loads that all lie in one '
             'qualitative load level (low, moderate or high); the user gives no numbers '
             'and names only that one level.',
             ('material', 'potential', 'n', 'band'), must_mention=AFM),
    Category('param_change', 'Ask to change one parameter of the simulation currently open '
             'to a value.', ('parameter', 'value')),
    Category('find_parameter', 'Ask which configuration parameter controls something, '
             'explicitly without changing anything.', ('parameter',)),
    Category('search_structures', 'Ask which structure files exist for a formula.',
             ('formula',)),
    Category('search_potentials', 'Ask which potential files of a type exist for a '
             'material.', ('material', 'potential')),
    Category('copy_structure', "Ask to copy a material's structure file into the "
             'workspace.', ('material',)),
    Category('make_folder', 'Ask to create a folder in the workspace.', ('folder',)),
    Category('list_workspace', 'Ask what folders are in the workspace.', ()),
    Category('summarise_dataset', 'Ask which materials and loads a results dataset holds.',
             ('dataset',)),
    Category('plot_cof_load', 'Ask for a plot of friction coefficient against normal load '
             'for a results dataset.', ('dataset',)),
    Category('materials_project', 'Ask what Materials Project lists for a property of a '
             'formula.', ('formula', 'property')),
    Category('lattice_pair', 'Ask how well the lattices of two materials match, or their '
             'lattice mismatch.', ('material', 'material_b')),
    Category('lattice_one', 'Ask for the in-plane lattice constant of one material.',
             ('material',)),
    Category('ask_wrong_unit', 'Ask for AFM on a material at loads given in a unit that '
             'is not a force.', ('material', 'potential', 'loads', 'wrong_unit'), 'ask',
             must_mention=AFM),
    Category('ask_unknown_material', 'Ask for a friction simulation of one material, '
             'giving only its name and no other requirement (no potential, surface, '
             'lubricant or setting).', ('material',), 'ask'),
    Category('ask_which_family', 'Ask for "a friction simulation" of a material without '
             'saying whether a tip or a second sheet slides.', ('material', 'potential'),
             'ask', must_not=AFM + SHEET + ('indent', 'pes', 'potential energy surface')),
    Category('decline_execution_setting', 'Ask to change an execution setting such as '
             'the job walltime or the energy minimisation command.', ('setting',),
             'declines'),
)

BY_NAME = {category.name: category for category in CATEGORIES}
