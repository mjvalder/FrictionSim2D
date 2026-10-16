"""Descriptors such as 'low load' turned into measured ranges.

There is no contact model here (no Hertz scaling): the numbers are load
regimes measured on the reference AFM setup by the ML_taxonomy campaigns, and
they are returned as measured. A context outside the setup they were measured
on gets the same numbers plus a warning, never an extrapolation.

Stdlib only, and nothing from ``run``/``agent``/``gui``: the GUI assistant
imports this at startup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

#: A context value: one number, or every value of a swept field.
Value = Union[float, Sequence[float], None]


@dataclass(frozen=True)
class ScaleContext:
    """What the caller knows about the current setup (from its document).

    Field names carry their unit, as the agent's request models do.
    """
    # pylint: disable=invalid-name

    model: str
    tip_radius_A: Value = None
    driving_spring_N_m: Value = None
    temperature_K: Value = None


@dataclass(frozen=True)
class ScaledRange:
    """A measured range for one descriptor."""

    quantity: str
    unit: str
    low: float
    high: float
    working: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Regime:
    """One measured range and where it was measured."""

    models: Tuple[str, ...]
    quantity: str
    descriptor: str
    unit: str
    low: float
    high: float
    source: str


#: Validity of the reference AFM setup: (context field, low, high, unit, label).
REFERENCE_SETUP = 'R = 25 A Si tip, driving spring 8-50 N/m, 300 K'
AFM_VALIDITY = (('tip_radius_A', 20.0, 30.0, 'A', 'R'),
                ('driving_spring_N_m', 8.0, 50.0, 'N/m', 'spring'),
                ('temperature_K', 250.0, 350.0, 'K', 'T'))

_AFM_FAMILY = ('afm', 'indent', 'pes_tip')
_SETUP = f'reference AFM setup ({REFERENCE_SETUP})'

#: The measured regimes. Sheet-on-sheet pressure, speed and temperature have
#: none yet, so `scale` returns None for them.
REGIMES = (
    Regime(_AFM_FAMILY, 'load', 'low', 'nN', 10.0, 20.0,
           'ML_taxonomy taxlib/constants.py BASELINE_LOADS = (10, 20, 30) nN, '
           f'{_SETUP}; 2-5 nN give poor contact (user ruling 2026-10-16)'),
    Regime(_AFM_FAMILY, 'load', 'mid', 'nN', 20.0, 50.0,
           'ML_taxonomy runtax/constants.py MID_LOAD_CORE = [20, 50] nN, '
           f'measured over the full campaign, {_SETUP}'),
    Regime(_AFM_FAMILY, 'load', 'high', 'nN', 50.0, 100.0,
           'ML_taxonomy tip campaign (loads 2-100 nN): above MID_LOAD_CORE up to '
           f'the campaign maximum of 100 nN, {_SETUP}'),
)

_DESCRIPTORS = {
    'low': 'low', 'small': 'low', 'light': 'low', 'lower': 'low',
    'mid': 'mid', 'moderate': 'mid', 'medium': 'mid', 'middle': 'mid',
    'intermediate': 'mid',
    'high': 'high', 'large': 'high', 'heavy': 'high', 'higher': 'high',
}
_QUANTITIES = {'load': 'load', 'loads': 'load', 'force': 'load', 'forces': 'load',
               'normal load': 'load', 'normal force': 'load'}


def _descriptor(text: str) -> Optional[str]:
    """'low load' / 'Small' -> 'low'; None when no word is a known descriptor."""
    for word in text.lower().replace('-', ' ').split():
        if word in _DESCRIPTORS:
            return _DESCRIPTORS[word]
    return None


def _values(value: Value) -> Tuple[float, ...]:
    """The numbers in a context value; anything non-numeric is skipped."""
    items = value if isinstance(value, (list, tuple)) else (value,)
    numbers = []
    for item in items:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return tuple(numbers)


def _validity_warnings(context: ScaleContext) -> Tuple[str, ...]:
    warnings = []
    for field_name, low, high, unit, label in AFM_VALIDITY:
        outside = [v for v in _values(getattr(context, field_name)) if not low <= v <= high]
        if outside:
            shown = ', '.join(f'{v:g}' for v in outside)
            warnings.append(f'measured for an {REFERENCE_SETUP} ({label} {low:g}-{high:g} '
                            f'{unit}); untested for {label} = {shown} {unit}')
    return tuple(warnings)


def scale(quantity: str, descriptor: str, context: ScaleContext) -> Optional[ScaledRange]:
    """The measured range for ``descriptor quantity`` on ``context.model``, or None."""
    name = _QUANTITIES.get(quantity.strip().lower())
    level = _descriptor(descriptor)
    if name is None or level is None:
        return None
    for regime in REGIMES:
        if (context.model in regime.models and regime.quantity == name
                and regime.descriptor == level):
            working = (f'{level} {name} = {regime.low:g}-{regime.high:g} {regime.unit} '
                       f'from {regime.source}')
            return ScaledRange(name, regime.unit, regime.low, regime.high, working,
                               _validity_warnings(context))
    return None
