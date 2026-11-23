"""Descriptors such as 'low load' turned into load bands for the current tip.

AFM-family bands are kept as a nominal pressure, F/R^2, calibrated so the
reference tip (R = 25 A) reproduces the reference load bands of the example
campaign; a different tip radius rescales the
load by (R/25)^2. This is not a contact model and not measured regimes: each
band's ``source`` says what its numbers are, and stiffness, sheet size and the
contact itself are ignored (a sensitivity study is in future-works). Other
differences from the reference setup (temperature, driving spring) keep the
numbers and add a warning.

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
#: Tip radius the nominal-pressure bands are calibrated on, in Angstrom.
REFERENCE_RADIUS_A = 25.0
#: 1 nN/A^2 expressed in GPa.
GPA_PER_NN_PER_A2 = 100.0
#: (context field, reference values, tolerance, unit, what to call it). The tip
#: radius is not here: it rescales the band instead of warning.
REFERENCE_VALUES = (('temperature_K', (300.0,), 1.0, 'K', 'temperature'),
                    ('driving_spring_N_m', (8.0, 50.0), 0.5, 'N/m', 'driving spring'))

_AFM_FAMILY = ('afm', 'indent', 'pes_tip')

def _nominal(load_nn: float) -> float:
    """Reference load -> nominal pressure F/R^2 in GPa at the reference radius."""
    return load_nn / REFERENCE_RADIUS_A ** 2 * GPA_PER_NN_PER_A2


#: The reference bands, as nominal pressure F/R^2 (GPa) calibrated on the
#: reference tip. Sheet-on-sheet pressure, speed and temperature have none yet,
#: so `scale` returns None for them.
REGIMES = (
    Regime(_AFM_FAMILY, 'load', 'low', 'GPa', _nominal(10.0), _nominal(20.0),
           'baseline loads chosen for good contact (user ruling 2026-10-16: 2-5 nN '
           'give poor contact) -- the baseline '
           'loads are 10, 20 and 30 nN; this band takes their lower part '
           '(10-20 nN at R = 25 A)'),
    Regime(_AFM_FAMILY, 'load', 'mid', 'GPa', _nominal(20.0), _nominal(50.0),
           'the mid-load core window the reference trends are fitted on '
           '(20-50 nN at R = 25 A)'),
    Regime(_AFM_FAMILY, 'load', 'high', 'GPa', _nominal(50.0), _nominal(100.0),
           "from the top of the mid window to the example campaign's maximum "
           '(100 nN at R = 25 A); no dedicated constant'),
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


def _load_band(regime: Regime, radius: float) -> Tuple[float, float]:
    """Nominal-pressure band -> load band (nN) for a tip of ``radius`` Angstrom."""
    factor = radius ** 2 / GPA_PER_NN_PER_A2
    return regime.low * factor, regime.high * factor


def scale(quantity: str, descriptor: str, context: ScaleContext) -> Optional[ScaledRange]:
    """The load band for ``descriptor quantity`` on ``context``'s tip, or None."""
    name = _QUANTITIES.get(quantity.strip().lower())
    level = _descriptor(descriptor)
    if name is None or level is None:
        return None
    for regime in REGIMES:
        if (context.model in regime.models and regime.quantity == name
                and regime.descriptor == level):
            warnings = list(_reference_warnings(context))
            radii = [r for r in _values(context.tip_radius_A) if r > 0]
            if not radii:
                radii = [REFERENCE_RADIUS_A]
                warnings.insert(0, f'no tip radius set; assumed the reference '
                                   f'{REFERENCE_RADIUS_A:g} A')
            low, high = _load_band(regime, radii[0])
            for other in radii[1:]:
                other_low, other_high = _load_band(regime, other)
                warnings.append(f'swept tip radius: R = {other:g} A would give '
                                f'{other_low:.3g}-{other_high:.3g} nN')
            working = (f'{level} {name} = {regime.low:.3g}-{regime.high:.3g} GPa nominal '
                       f'pressure (F/R^2) -> {low:.3g}-{high:.3g} nN at R = {radii[0]:g} A; '
                       f'simple tip-size scaling, no contact model. Band: {regime.source}')
            return ScaledRange(name, 'nN', low, high, working, tuple(warnings))
    return None
