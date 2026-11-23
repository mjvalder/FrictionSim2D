"""One axis-label function, over the observable registry.

`Observable.units` and `Observable.role_key` had no consumer in this package
at all: every label was hand-written per plot, per document. `label_for` is
the fallback a plot gets for free; an explicit `x_label`/`y_label` on the
spec still wins, because a paper figure sometimes needs a shorter axis than
the honest one.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

__all__ = ['DERIVED_LABELS', 'ROLE_LABELS', 'FAMILY_LABEL_OVERRIDES',
           'unit_for', 'display_name', 'label_for']

#: Columns this package derives rather than parses: (display name, unit).
DERIVED_LABELS: Dict[str, Tuple[str, Optional[str]]] = {
    'time': ('Timestep', 'step'),
    'lf': ('Lateral Force', 'nN'),
    'lateral_force': ('Lateral Force', 'nN'),
    'lf_as_was': ('Lateral Force (as written)', 'nN'),
    'lf_sx_sy': ('Lateral Force (reaction)', 'nN'),
    'lf_fx_fy': ('Lateral Force (drive)', 'nN'),
    'cof': ('Coefficient of Friction', None),
    'cof_slope': ('COF (slope of LF vs NF)', None),
    'force': ('Normal Load', 'nN'),
    'pressure': ('Pressure', 'GPa'),
    'load': ('Load', None),
    'angle': ('Scan Angle', 'deg'),
    'speed': ('Scan Speed', 'm/s'),
    'layer': ('Layers', None),
    'temperature': ('Temperature', 'K'),
    'tip_radius': ('Tip Radius', 'A'),
    'tip_sep': ('Tip-Sheet Separation', 'A'),
    'tip_pos': ('Tip Displacement', 'A'),
    'tipspeed': ('Tip Speed', 'm/s'),
}

#: (family, column) -> display name, overriding the role-derived one, for
#: where a role is served by a DIFFERENT physical quantity in one family and
#: the axis must say so.
#:
#: Empty, deliberately. An earlier draft put a caveat here on the flake's
#: `nf` and `cof` -- that the flake's normal force was the flake-sheet
#: adhesion rather than an applied load. Flake slides now apply a real load
#: (`fix force flake addforce`, pressure x the flake's projected area) and
#: `nf@sheet` claims the `flake` sim_type, so a flake COF is lateral force
#: over applied load exactly as sheet-on-sheet's is. The mechanism is kept
#: because the situation it covers is real and will recur.
FAMILY_LABEL_OVERRIDES: Dict[Tuple[str, str], str] = {}

#: role_key -> display name. Roles rather than names, so a family's own
#: channel name (v_xfrict vs v_fx_spr) never needs its own entry.
ROLE_LABELS: Dict[str, str] = {
    'normal_load': 'Normal Load',
    'lateral_force_x': 'Lateral Force x',
    'lateral_force_y': 'Lateral Force y',
    'raw_normal_load': 'Normal Load (raw)',
    'raw_lateral_force_x': 'Lateral Force x (raw)',
    'raw_lateral_force_y': 'Lateral Force y (raw)',
    'sheet_com_x': 'Sheet COM x', 'sheet_com_y': 'Sheet COM y',
    'sheet_com_z': 'Sheet COM z',
    'driven_com_x': 'Driven COM x', 'driven_com_y': 'Driven COM y',
    'driven_com_z': 'Driven COM z',
    'raw_driven_com_x': 'Driven COM x (raw)',
    'raw_driven_com_y': 'Driven COM y (raw)',
    'raw_driven_com_z': 'Driven COM z (raw)',
    'total_energy': 'Total Energy', 'potential_energy': 'Potential Energy',
    'kinetic_energy': 'Kinetic Energy',
    'thermostat_work_tip': 'Thermostat Work (tip)',
    'thermostat_work_substrate': 'Thermostat Work (substrate)',
    'thermostat_work_interior': 'Thermostat Work (interior)',
    'group_temperature': 'Group Temperature',
    'group_temperature_center': 'Group Temperature (center)',
    'group_temperature_layer_1': 'Group Temperature (layer 1)',
    'group_temperature_layer_n': 'Group Temperature (top layer)',
    'tip_sheet_interaction': 'Tip-Sheet Interaction Energy',
    'interlayer_interaction': 'Interlayer Interaction Energy',
    'interlayer_interaction_all_pairs': 'Interlayer Energy (all layer pairs)',
    'contact_count': 'Contact Count',
    'min_gap': 'Minimum Gap',
    'max_deflection': 'Maximum Deflection',
    'drive_force_x': 'Drive Force x', 'drive_force_y': 'Drive Force y',
    'reaction_force_x': 'Reaction Force x',
    'reaction_force_y': 'Reaction Force y',
    'reaction_force_z': 'Reaction Force z',
}

_LAYER_RE = re.compile(r'^(?P<stem>com|pe_sum|sigma_xx_2d|sigma_yy_2d)'
                       r'(?P<axis>[xyz])?_L(?P<n>\d+)$')

_LAYER_STEMS = {'com': 'COM', 'pe_sum': 'Group PE',
                'sigma_xx_2d': 'Membrane Stress xx',
                'sigma_yy_2d': 'Membrane Stress yy'}


def _observable(column: str, family: str):
    """The registry entry for `column` in `family`, or None.

    Per-layer channels beyond L2 are not in REGISTRY (it carries the L1/L2
    pair for the golden tests); `layer_channels` generates them for any layer
    count, and the count is read out of the column name itself.
    """
    from ..core.observables import REGISTRY, layer_channels  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    for ob in REGISTRY.values():
        if ob.name == column and family in ob.sim_types:
            return ob
    match = _LAYER_RE.match(column)
    if match:
        for ob in layer_channels(int(match.group('n'))):
            if ob.name == column and family in ob.sim_types:
                return ob
    return None


def unit_for(column: str, family: str = 'afm') -> Optional[str]:
    """The declared unit of `column` for `family`, or None."""
    if column in DERIVED_LABELS:
        return DERIVED_LABELS[column][1]
    ob = _observable(column, family)
    return ob.units if ob is not None else None


def display_name(column: str, family: str = 'afm') -> str:
    """The human-readable name of `column`, without its unit."""
    if (family, column) in FAMILY_LABEL_OVERRIDES:
        return FAMILY_LABEL_OVERRIDES[(family, column)]
    if column in DERIVED_LABELS:
        return DERIVED_LABELS[column][0]
    match = _LAYER_RE.match(column)
    if match and match.group('stem') in _LAYER_STEMS:
        stem = _LAYER_STEMS[match.group('stem')]
        axis = f" {match.group('axis')}" if match.group('axis') else ''
        return f"Layer {int(match.group('n'))} {stem}{axis}"
    ob = _observable(column, family)
    if ob is not None and ob.role_key in ROLE_LABELS:
        return ROLE_LABELS[ob.role_key]
    return ' '.join(part.capitalize() for part in column.split('_'))


def label_for(column: str, family: str = 'afm') -> str:
    """`'Normal Load (nN)'` -- the axis label for one column.

    Never raises: an unrecognised column comes back prettified, because a
    missing label must not stop a figure being drawn.
    """
    name = display_name(column, family)
    unit = unit_for(column, family)
    return f'{name} ({unit})' if unit else name
