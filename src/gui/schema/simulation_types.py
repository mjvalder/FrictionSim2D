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
    #: Whether a two-material (``[2D-1]``/``[2D-2]``) config routes to a
    #: heterostructure builder for this mode. True for every mode currently
    #: offered -- ``HeteroAFMSimulation``, ``HeteroIndentSimulation``,
    #: ``HeteroPESTipSimulation``, ``HeteroPESSheetSimulation`` and
    #: ``HeteroSheetOnSheetSimulation`` cover all five -- so nothing is
    #: refused on account of it today. It is declared rather than assumed so
    #: that a mode added without a hetero build path refuses the second
    #: material outright instead of silently dropping it on the way to a
    #: builder that ignores `sheets`.
    supports_hetero: bool = True
    #: Narrows the family's ``outer_loop_choices`` for this mode specifically.
    #: ``None`` (the default) means "use the family's value". Set this when a
    #: mode's build path does not act on a value the family otherwise
    #: supports -- e.g. AFM's ``dynamic_indent`` mode writes one
    #: ``slide_a<angle>.in`` per angle, but ``indent`` never reads the angle
    #: variable and ``pes`` overrides the build entirely, so offering
    #: ``scan_angle`` there would configure a split that silently does
    #: nothing.
    outer_loop_choices: Optional[Tuple[str, ...]] = None


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
    #: ``[general]`` keys this family's templates never read. Offering them
    #: invites configuring something that silently has no effect.
    unsupported_general: Tuple[str, ...] = ()
    #: Values ``[general] outer_loop`` may take for this family.
    outer_loop_choices: Tuple[str, ...] = ()

    def mode(self, key: str) -> Optional[SimulationMode]:
        return next((m for m in self.modes if m.key == key), None)

    def outer_loop_choices_for(self, mode: SimulationMode) -> Tuple[str, ...]:
        """Effective ``outer_loop`` choices for one mode of this family.

        Falls back to the family's own value when the mode does not narrow it.
        """
        return (mode.outer_loop_choices if mode.outer_loop_choices is not None
                else self.outer_loop_choices)

    def supports(self, section: str, key: str) -> bool:
        """Whether a config key does anything for this family."""
        return not (section == 'general' and key in self.unsupported_general)


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
    # The AFM templates drive the tip with a normal load; pressure and the two
    # spring constants appear only in the sheet-on-sheet templates.
    unsupported_general=('pressure', 'bond_spring', 'driving_spring'),
    outer_loop_choices=('force',),
    modes=(
        SimulationMode(
            key='dynamic_indent', model='afm',
            label='Dynamic + indent',
            description='Approach and indent, then slide under load.',
            # Only mode in the family whose slide phase actually honors
            # outer_loop='scan_angle' (afm.py's _write_slide_scripts writes
            # one slide_a<angle>.in per angle here).
            outer_loop_choices=('force', 'scan_angle'),
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
            # PESTipSimulation.write_inputs overrides the run phase entirely
            # (a single self-contained lammps/slide.in) and never reads
            # outer_loop; offering 'force' here would configure a split that
            # silently does nothing. (count_expansion no longer over-reports
            # it -- it reads the builder's SPLIT_OUTER_LOOPS -- but a control
            # that configures nothing is still worth not offering.)
            outer_loop_choices=(),
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
    # There is no tip to load, so the normal-load field does nothing here;
    # the sheets are driven by pressure.
    unsupported_general=('force',),
    outer_loop_choices=('pressure', 'scan_speed'),
    modes=(
        SimulationMode(
            key='dynamic', model='sheetonsheet',
            label='Dynamic (slide)',
            description='Slide the upper sheet under load.',
            # Only mode in the family whose slide phase actually honors
            # outer_loop='scan_angle' (sheetonsheet.py writes one
            # slide_a<angle>.in per angle here); the PES scan below
            # overrides write_inputs entirely and never looks at outer_loop.
            outer_loop_choices=('pressure', 'scan_speed', 'scan_angle'),
        ),
        SimulationMode(
            key='pes', model='pes_sheet',
            label='PES scan',
            description='Static corrugation scan of one sheet over the other.',
            sections=('pes',),
            # PESSheetSimulation.write_inputs overrides the run phase entirely
            # (writes lammps/slide.in, or slide.in + slide_refine.in, directly)
            # and never reads outer_loop; offering pressure/scan_speed here
            # would configure a split that silently does nothing.
            outer_loop_choices=(),
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
