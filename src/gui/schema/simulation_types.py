"""The simulation types the GUI offers.

The CLI exposes five ``run`` commands, but they are not five independent
things: they are two builder families, each with a mode. The families differ in
what they contain (a tip or a second sheet), in their directory layout (an
``L{n}`` level or not) and in their assembly grammar; the modes within a family
differ only in which run scripts follow the shared build.

Presenting it that way is both closer to the code and closer to how the
simulations are actually chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    'SimulationMode', 'SimulationFamily', 'FAMILIES', 'MODES',
    'modes_for_family', 'mode_for_model', 'family_for_model',
]


@dataclass(frozen=True)
class SimulationMode:
    """One way of running a family."""

    key: str
    #: CLI model name; what `run_simulations` is given.
    model: str
    label: str
    description: str
    #: Config sections this mode makes relevant, beyond the family's own.
    sections: Tuple[str, ...] = ()
    #: True when the build writes an approach + indentation `system.in`.
    has_system_phase: bool = True


@dataclass(frozen=True)
class SimulationFamily:
    """A builder family: a set of components and a directory layout."""

    key: str
    label: str
    description: str
    #: Config sections always present for this family.
    sections: Tuple[str, ...]
    modes: Tuple[SimulationMode, ...]
    #: Whether builds insert an `L{n}` level per sheet layer count.
    has_layer_levels: bool
    #: Whether a substrate-free config is accepted (only PES tip is).
    optional_sections: Tuple[str, ...] = ()

    def mode(self, key: str) -> Optional[SimulationMode]:
        return next((m for m in self.modes if m.key == key), None)


AFM = SimulationFamily(
    key='afm',
    label='AFM tip',
    description=(
        'A tip pressed onto a supported 2D sheet. The shared build writes an '
        'approach and indentation phase; the mode decides what follows.'
    ),
    sections=('2D', 'tip', 'sub', 'general'),
    optional_sections=('flake', 'lattice_match', 'lj_override'),
    has_layer_levels=True,
    modes=(
        SimulationMode(
            key='dynamic_indent', model='afm',
            label='Dynamic + indent',
            description='Approach and indent, then slide under load.',
        ),
        SimulationMode(
            key='indent', model='indent',
            label='Indent only',
            description='Approach and indent, then hold and retract; no sliding.',
            sections=('indent',),
        ),
        SimulationMode(
            key='pes', model='pes_tip',
            label='PES scan',
            description=(
                'Static potential-energy-surface scan of the tip over the sheet. '
                'Self-contained: no indentation phase.'
            ),
            sections=('pes',),
            has_system_phase=False,
        ),
    ),
)

SHEET_ON_SHEET = SimulationFamily(
    key='sheetonsheet',
    label='Sheet on sheet',
    description='Two stacked sheets sliding against each other, no tip.',
    sections=('2D', 'general'),
    optional_sections=('lattice_match', 'lj_override'),
    has_layer_levels=False,
    modes=(
        SimulationMode(
            key='dynamic', model='sheetonsheet',
            label='Dynamic (slide)',
            description='Slide the upper sheet under load.',
        ),
        SimulationMode(
            key='pes', model='pes_sheet',
            label='PES scan',
            description='Static corrugation scan of one sheet over the other.',
            sections=('pes',),
        ),
    ),
)

FAMILIES: Tuple[SimulationFamily, ...] = (AFM, SHEET_ON_SHEET)

#: Every (family, mode) pair, keyed by CLI model name.
MODES: Dict[str, Tuple[SimulationFamily, SimulationMode]] = {
    mode.model: (family, mode)
    for family in FAMILIES
    for mode in family.modes
}


def modes_for_family(family_key: str) -> List[SimulationMode]:
    """Modes offered by a family."""
    family = next((f for f in FAMILIES if f.key == family_key), None)
    return list(family.modes) if family else []


def mode_for_model(model: str) -> Optional[SimulationMode]:
    """The mode a CLI model name corresponds to."""
    entry = MODES.get(model)
    return entry[1] if entry else None


def family_for_model(model: str) -> Optional[SimulationFamily]:
    """The family a CLI model name belongs to."""
    entry = MODES.get(model)
    return entry[0] if entry else None


def sections_for(model: str) -> List[str]:
    """Config sections relevant to a model, required ones first."""
    entry = MODES.get(model)
    if not entry:
        return []
    family, mode = entry
    seen: List[str] = []
    for section in (*family.sections, *mode.sections, *family.optional_sections):
        if section not in seen:
            seen.append(section)
    return seen
