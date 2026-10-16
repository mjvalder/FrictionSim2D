"""Descriptors such as 'low load' turned into reference load bands.

These are reference bands from the ML_taxonomy work and the example campaign,
not a contact model and not measured regimes: each band's ``source`` says what
its numbers are. A context that differs from the reference setup gets the
same numbers plus a warning, never a rescaling.

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
    """A reference band for one descriptor."""

    quantity: str
    unit: str
    low: float
    high: float
    working: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Regime:
    """One reference band and what its numbers are."""

    models: Tuple[str, ...]
    quantity: str
    descriptor: str
    unit: str
    low: float
    high: float
    source: str


#: The reference AFM setup the bands come from.
REFERENCE_SETUP = 'R = 25 A Si tip at 300 K, driving spring 8 or 50 N/m'
#: (context field, reference values, tolerance, unit, what to call it).
REFERENCE_VALUES = (('tip_radius_A', (25.0,), 0.5, 'A', 'tip radius'),
                    ('temperature_K', (300.0,), 1.0, 'K', 'temperature'),
                    ('driving_spring_N_m', (8.0, 50.0), 0.5, 'N/m', 'driving spring'))

_AFM_FAMILY = ('afm', 'indent', 'pes_tip')

#: The reference bands. Sheet-on-sheet pressure, speed and temperature have
#: none yet, so `scale` returns None for them.
REGIMES = (
    Regime(_AFM_FAMILY, 'load', 'low', 'nN', 10.0, 20.0,
           'baseline loads chosen for good contact (user ruling 2026-10-16: 2-5 nN '
           'give poor contact) -- ML_taxonomy taxlib BASELINE_LOADS = (10, 20, 30); '
           'this band takes its lower part'),
    Regime(_AFM_FAMILY, 'load', 'mid', 'nN', 20.0, 50.0,
           'the mid-load core window the ML_taxonomy analysis fits its reference '
           'trends on (runtax MID_LOAD_CORE = [20, 50])'),
    Regime(_AFM_FAMILY, 'load', 'high', 'nN', 50.0, 100.0,
           "from the top of the mid window to the example campaign's maximum "
           '(100 nN); no dedicated constant'),
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


def _reference_warnings(context: ScaleContext) -> Tuple[str, ...]:
    warnings = []
    for field_name, reference, tolerance, unit, label in REFERENCE_VALUES:
        differs = [v for v in _values(getattr(context, field_name))
                   if all(abs(v - r) > tolerance for r in reference)]
        if differs:
            shown = ', '.join(f'{v:g}' for v in differs)
            warnings.append(f'reference setup is {REFERENCE_SETUP}; your {label} '
                            f'{shown} {unit} differs -- treat as a starting point')
    return tuple(warnings)


def scale(quantity: str, descriptor: str, context: ScaleContext) -> Optional[ScaledRange]:
    """The reference band for ``descriptor quantity`` on ``context.model``, or None."""
    name = _QUANTITIES.get(quantity.strip().lower())
    level = _descriptor(descriptor)
    if name is None or level is None:
        return None
    for regime in REGIMES:
        if (context.model in regime.models and regime.quantity == name
                and regime.descriptor == level):
            working = (f'{level} {name} = {regime.low:g}-{regime.high:g} {regime.unit}: '
                       f'{regime.source}')
            return ScaledRange(name, regime.unit, regime.low, regime.high, working,
                               _reference_warnings(context))
    return None
