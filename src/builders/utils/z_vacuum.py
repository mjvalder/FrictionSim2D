"""Periodic-z vacuum: size the box from the atoms, refuse a deck that is too thin.

Every run script is ``boundary p p p``, so the top atoms see the periodic
image of the bottom ones across ``(zhi - zlo) - (zmax - zmin)`` of vacuum.
That gap must exceed the largest pair cutoff plus the neighbor skin, or the
stack interacts with itself. The box is therefore built from the actual atom
z-range of the data read::

    V   = cutoff + skin + settings.geometry.z_vacuum_margin
    zlo = zmin - V/2
    zhi = zmax + max(V/2, top_clearance) + top_travel

``top_travel`` is how far a run moves atoms upward past ``zmax`` (the indent
retract); it keeps the vacuum at ``V`` at the top of that travel.
``top_clearance`` is the room a deck needs above the atoms for something
that is not an interacting atom (the drive's virtual atom, created a fixed
height above the driven layer); it only ever adds vacuum.

:meth:`ZBox.check_deck` re-reads the box a rendered deck actually writes and
raises when its vacuum is below ``cutoff + skin``, so a template that drops
the computed bounds fails the build instead of the run.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

#: Column of z in a ``Atoms # <style>`` data section, per atom style.
_Z_COLUMN = {'atomic': 4, 'charge': 5, 'molecular': 5, 'bond': 5, 'angle': 5,
             'full': 6}

_REGION_BOX = re.compile(
    r'^\s*region\s+box\s+(?:block|prism)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)', re.M)
_CHANGE_BOX_Z = re.compile(
    r'^\s*change_box\s+all\b.*?\bz\s+final\s+(\S+)\s+(\S+)', re.M)


#: The sheet slides create the drive's virtual atom this far above the driven
#: layer's centre of mass (``create_atoms ... $(xcm(layer_N,z)+10)`` in
#: sheetonsheet/slide.lmp and hetero/slide.lmp).
SHEET_VIRTUAL_ATOM_HEIGHT = 10.0

#: Room kept above that virtual atom, so ``create_atoms single`` finds it
#: inside the box.
VIRTUAL_ATOM_ROOM = 1.0


def data_atom_z_range(path: Union[str, Path]) -> Tuple[float, float]:
    """``(zmin, zmax)`` of the atoms in a LAMMPS data file.

    The z column follows the ``Atoms # <style>`` comment (``atomic`` when
    there is none).
    """
    zmin = float('inf')
    zmax = float('-inf')
    col = None
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if col is None:
                if line.startswith('Atoms'):
                    style = line.split('#', 1)[1].strip() if '#' in line else 'atomic'
                    col = _Z_COLUMN.get(style.split()[0] if style else 'atomic', 4)
                continue
            if not line:
                if zmin <= zmax:
                    break
                continue
            parts = line.split()
            if not parts[0].lstrip('-').isdigit():
                break
            z = float(parts[col])
            zmin = min(zmin, z)
            zmax = max(zmax, z)
    if zmin > zmax:
        raise ValueError(f"{path}: no atoms in an Atoms section")
    return zmin, zmax


def shifted(zrange: Tuple[float, float], dz: float) -> Tuple[float, float]:
    """``zrange`` moved by ``dz`` (a ``read_data ... shift`` z)."""
    return zrange[0] + float(dz), zrange[1] + float(dz)


def union_z_range(ranges: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    """The z-range covering every ``(zmin, zmax)`` in ``ranges``."""
    ranges = list(ranges)
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


@dataclass(frozen=True)
class ZBox:
    """The periodic-z box of one run, from its atoms and its cutoff.

    Attributes:
        zmin, zmax: Atom z-range of the data the deck reads (Angstrom).
        cutoff: Largest pair cutoff acting across the boundary.
        skin: Neighbor skin (``settings.simulation.neighbor_list``).
        margin: ``settings.geometry.z_vacuum_margin``.
        top_travel: How far the run moves atoms above ``zmax``.
        top_clearance: Least room the deck needs above ``zmax``.
    """
    zmin: float
    zmax: float
    cutoff: float
    skin: float
    margin: float
    top_travel: float = 0.0
    top_clearance: float = 0.0

    @classmethod
    def for_settings(cls, settings, zrange: Tuple[float, float], cutoff: float, *,
                     top_travel: float = 0.0, top_clearance: float = 0.0) -> 'ZBox':
        """A :class:`ZBox` with the skin and margin read from ``settings``."""
        return cls(zmin=float(zrange[0]), zmax=float(zrange[1]), cutoff=float(cutoff),
                   skin=float(settings.simulation.neighbor_list),
                   margin=float(settings.geometry.z_vacuum_margin),
                   top_travel=float(top_travel), top_clearance=float(top_clearance))

    @property
    def vacuum(self) -> float:
        """The vacuum the box is built with: cutoff + skin + margin."""
        return self.cutoff + self.skin + self.margin

    @property
    def required(self) -> float:
        """The least vacuum a deck may have: cutoff + skin."""
        return self.cutoff + self.skin

    @property
    def zlo(self) -> float:
        """Box bottom."""
        return round(self.zmin - self.vacuum / 2.0, 6)

    @property
    def zhi(self) -> float:
        """Box top."""
        return round(self.zmax + max(self.vacuum / 2.0, self.top_clearance)
                     + self.top_travel, 6)

    def context(self) -> dict:
        """Template keys ``box_zlo`` / ``box_zhi``."""
        return {'box_zlo': f'{self.zlo:.6f}', 'box_zhi': f'{self.zhi:.6f}'}

    def check(self, zlo: float, zhi: float, where: str = 'deck') -> None:
        """Refuse a box ``[zlo, zhi]`` that cuts the atoms or is too thin.

        Raises:
            ValueError: The atoms do not fit, or the vacuum (less
                ``top_travel``) is below ``cutoff + skin``.
        """
        if zlo >= self.zmin or zhi <= self.zmax + self.top_clearance:
            raise ValueError(
                f"{where}: periodic z box [{zlo:.3f}, {zhi:.3f}] does not contain "
                f"the atoms [{self.zmin:.3f}, {self.zmax:.3f}]"
                + (f" plus {self.top_clearance:g} A above them"
                   if self.top_clearance else ""))
        vacuum = (zhi - zlo) - (self.zmax - self.zmin) - self.top_travel
        if vacuum < self.required - 1e-6:
            raise ValueError(
                f"{where}: periodic z vacuum {vacuum:.2f} A is below the pair "
                f"cutoff {self.cutoff:g} A + neighbor skin {self.skin:g} A = "
                f"{self.required:g} A, so the top atoms would interact with the "
                f"periodic image of the bottom ones (box [{zlo:.3f}, {zhi:.3f}], "
                f"atoms [{self.zmin:.3f}, {self.zmax:.3f}]"
                + (f", top travel {self.top_travel:g} A" if self.top_travel else "")
                + "). Use ZBox.zlo/zhi for the box.")

    def check_deck(self, text: str, where: str = 'deck') -> None:
        """:meth:`check` the z extent a rendered deck finally sets.

        The extent is the last ``change_box all ... z final`` if any, else
        ``region box``. A deck that defines no box (it reads a saved
        restart/data state) passes.
        """
        bounds: Optional[Tuple[str, str]] = None
        changed = _CHANGE_BOX_Z.findall(text)
        if changed:
            bounds = changed[-1]
        else:
            region = _REGION_BOX.findall(text)
            if region:
                bounds = region[-1]
        if bounds is None:
            return
        try:
            zlo, zhi = float(bounds[0]), float(bounds[1])
        except ValueError as exc:
            raise ValueError(f"{where}: unrendered z box bounds {bounds}") from exc
        self.check(zlo, zhi, where)
