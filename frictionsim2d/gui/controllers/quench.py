"""The amorphous-block (quench) family, as the Builder sees it.

Pure Python, no Qt: what a quench document would make, and how a missing
amorphous block (``AmorphousMissing``) prefills one. The runs themselves are in
``quench_run``; nothing here starts a process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...core.config import GlobalSettings

__all__ = ['QUENCH_MODEL', 'CONDITION_KEYS', 'QuenchPlan', 'plan_quench',
           'prefill_quench_values', 'quench_steps']

#: The model name of the Amorphous block family (``run quench``).
QUENCH_MODEL = 'quench'

#: The six melt-quench conditions that key a block.
CONDITION_KEYS = ('melt_temp', 'quench_temp', 'quench_rate', 'timestep',
                  'melt_steps', 'equilibrate_steps')


@dataclass
class QuenchPlan:
    """What Generate would write for a quench document."""

    #: Members that get a job directory.
    members: List[Any] = field(default_factory=list)
    #: ``(member, library block)`` for members a block already satisfies.
    present: List[Tuple[Any, Path]] = field(default_factory=list)

    def notes(self) -> List[str]:
        """One line per member left out because the library has it."""
        return [
            f"[quench] {member.material} [{member.key.key8}]: already in the "
            f"library ({block.name}); no job is written for it"
            for member, block in self.present
        ]


def plan_quench(section: Mapping[str, Any], settings: GlobalSettings,
                base_dir: Optional[Path] = None) -> QuenchPlan:
    """The members of a ``[quench]`` section, split like ``run quench`` splits them.

    Raises:
        ValueError: an invalid section or member (missing potential or CIF, a
            non-positive rate), with the CLI's own message.
    """
    # pylint: disable-next=import-outside-toplevel
    from ...builders.construction.amorphous import expand_quench_config, satisfying_block
    plan = QuenchPlan()
    for member in expand_quench_config(section, settings, base_dir=base_dir):
        block = satisfying_block(member.key, member.slab)
        if block is None:
            plan.members.append(member)
        else:
            plan.present.append((member, block))
    return plan


def quench_steps(member: Any) -> int:
    """Total MD steps of one member (melt + quench + equilibrate)."""
    # pylint: disable-next=import-outside-toplevel
    from ...builders.construction.amorphous import quench_nsteps
    key = member.key
    return int(key.melt_steps) + quench_nsteps(key) + int(key.equilibrate_steps)


def _template(member: Mapping[str, Any], key: str) -> str:
    """``member[key]`` with the material name replaced by ``{mat}``."""
    return str(member[key]).replace(str(member['material']), '{mat}')


def prefill_quench_values(members: Sequence[Mapping[str, Any]]
                          ) -> Tuple[Dict[str, Any], List[Mapping[str, Any]]]:
    """A ``[quench]`` section for missing blocks, and the blocks it cannot hold.

    ``members`` are ``QuenchMember.as_dict()`` records. One section describes a
    product (materials x each listed condition), so it takes every member that
    shares the first one's conditions and potential/CIF layout and leaves the
    rest to a later round: nothing is made that was not asked for.

    Returns:
        ``(section values, left-out members)``.
    """
    if not members:
        return {}, []
    first = members[0]

    def shape(member: Mapping[str, Any]) -> tuple:
        return (member['pot_type'], _template(member, 'potential'), _template(member, 'cif'),
                *(member[name] for name in CONDITION_KEYS))

    group = [m for m in members if shape(m) == shape(first)]
    left_out = [m for m in members if shape(m) != shape(first)]
    materials = list(dict.fromkeys(str(m['material']) for m in group))
    if len(materials) == len(group) and len(materials) > 1:
        potential, cif = _template(first, 'potential'), _template(first, 'cif')
    else:
        # One material (or repeats of it): the literal paths are exact.
        potential, cif = str(first['potential']), str(first['cif'])
        group = group[:1] if len(materials) == 1 else group
    slab = [max(float(m['slab'][axis]) for m in group) for axis in range(3)]
    values: Dict[str, Any] = {
        'material': materials[0] if len(materials) == 1 else materials,
        'potential': potential,
        'pot_type': first['pot_type'],
        'cif': cif,
        'slab': slab,
    }
    for name in CONDITION_KEYS:
        values[name] = first[name]
    return values, left_out
