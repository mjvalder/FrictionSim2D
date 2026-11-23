"""Builder-side glue for the interlayer-style choice (spec 2026-11-11 §5, §7).

Helpers every family shares:

* :func:`set_interlayer_from_config` hands the run's ``[general]``
  interlayer fields and each sheet's ``interlayer_g`` to
  :meth:`~frictionsim2d.potentials.manager.PotentialManager.set_interlayer`.
* :func:`interlayer_template_context` builds the slide-template keys
  (``needs_molecule_ids``, ``molecule_groups``, ``use_group_group``,
  ``interlayer_energy_substyle``).
* :func:`interlayer_provenance_block` is the provenance record's
  ``interlayer`` block.
* :func:`check_interlayer_box` / :func:`check_interlayer_box_from_data`
  refuse an ILP/KC build whose periodic cell is narrower than the
  interlayer cutoff plus the neighbor skin.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ...potentials.interlayer import ILP_CUTOFF, KC_FULL_CUTOFF, gaussian_g

#: Styles whose interlayer term ``compute group/group`` can evaluate (it
#: needs ``pair->single()``). Not ``lj_g``: ``kolmogorov/crespi/z`` has no
#: ``single()``, and LAMMPS aborts the run with "Pair hybrid sub-style does
#: not support single call" (measured, tests/integration/
#: test_interlayer_lammps.py), so LJ-G takes the reaction-force branch.
GROUP_GROUP_STYLES = ('lj',)

#: Interlayer cutoff (Angstrom) of the styles the periodic cell must exceed.
INTERLAYER_CUTOFFS: Dict[str, float] = {'ilp': ILP_CUTOFF, 'kc': KC_FULL_CUTOFF}


def interlayer_style(general: Any) -> str:
    """The run's interlayer style; ``lj`` for configs that predate the field."""
    return getattr(general, 'interlayer_style', 'lj') or 'lj'


def interlayer_comparison(general: Any) -> bool:
    """Whether the opt-in comparison mode (spec §7) is on."""
    return bool(getattr(general, 'interlayer_comparison', False))


def set_interlayer_from_config(pm: Any, general: Any,
                               components: Mapping[str, Any],
                               base_dir: Optional[Union[str, Path]] = None) -> None:
    """Configure ``pm``'s interlayer style from the run config.

    Args:
        pm: The :class:`~frictionsim2d.potentials.manager.PotentialManager`.
        general: The run's ``[general]`` config.
        components: ``{component name: sheet config}``; each config's
            ``interlayer_g`` (or the built-in table) gives that component's
            Gaussian strength.
        base_dir: Directory ``interlayer_file`` is resolved against. ``None``
            is the working directory -- the same place ``pot_path`` and
            ``cif_path`` resolve from when the CLI validates a config.
    """
    g_by_component = {}
    for name, cfg in components.items():
        g = gaussian_g(cfg.mat, getattr(cfg, 'interlayer_g', None))
        if g is not None:
            g_by_component[name] = g
    pm.set_interlayer(interlayer_style(general),
                      file=getattr(general, 'interlayer_file', None),
                      g_by_component=g_by_component, base_dir=base_dir)


def interlayer_template_context(general: Any, *, has_internal_lj: bool,
                                pm: Any,
                                molecule_groups: Sequence[Union[str, Dict[str, str]]],
                                ) -> Dict[str, Any]:
    """Slide-template keys for the interlayer choice.

    ``use_group_group`` is False for a potential with its own interlayer term,
    for ``lj_g``/``ilp``/``kc`` (``lj_g``'s KC/z has no ``pair->single()``),
    and whenever comparison mode is on, so every style of a comparison sweep
    reports the same reaction-force friction (spec §7).
    ``interlayer_energy_substyle`` is set only where ``compute pair`` isolates
    the interlayer term (``ilp``/``kc``); under ``lj_g``, and ``lj`` in
    comparison mode, the LJ part shares ``lj/cut`` with the tip/virtual pairs,
    so the energy is dropped there as it is for rebomos.
    """
    style = interlayer_style(general)
    use_group_group = (not has_internal_lj and style in GROUP_GROUP_STYLES
                       and not interlayer_comparison(general))
    needs_ids = bool(getattr(pm, 'needs_molecule_ids', False))
    substyle: Optional[str] = None
    if not use_group_group and style in ('ilp', 'kc'):
        substyle = getattr(pm, 'interlayer_substyle', None)
    groups: List[Union[str, Dict[str, str]]] = list(molecule_groups) if needs_ids else []
    return {
        'needs_molecule_ids': needs_ids,
        'molecule_groups': groups,
        'use_group_group': use_group_group,
        'interlayer_energy_substyle': substyle,
    }


def interlayer_provenance_block(pm: Any, general: Any) -> Optional[Dict[str, Any]]:
    """The provenance ``interlayer`` block, or None where it is not recorded.

    ``pm.interlayer_provenance()`` (style, parameter file + sha256, LJ-G g
    values with their source, generated files) plus ``comparison``. None --
    so the record carries no ``interlayer`` key -- for the default ``lj``
    with comparison mode off. (``provenance/config.json`` and ``config_hash``
    do include the four new default interlayer fields; only the LAMMPS
    scripts, data and settings stay byte-identical.) None without a manager
    to ask.
    """
    comparison = interlayer_comparison(general)
    if pm is None or (interlayer_style(general) == 'lj' and not comparison):
        return None
    block = dict(pm.interlayer_provenance())
    block['comparison'] = comparison
    return block


def periodic_widths(lx: float, ly: float, xy: float = 0.0) -> Tuple[float, float]:
    """Perpendicular widths of the in-plane periodic cell ``a=(lx,0)``, ``b=(xy,ly)``.

    The distance between the two faces spanned by ``b`` (``lx*ly/|b|``) and
    between the two spanned by ``a`` (``ly``). A tilted cell is narrower
    than its ``lx``: this is the length LAMMPS compares the ghost cutoff with.
    """
    return lx * ly / math.hypot(xy, ly), ly


def check_interlayer_box(general: Any, *, lx: float, ly: float, xy: float = 0.0,
                         skin: float) -> None:
    """Refuse an ``ilp``/``kc`` cell narrower than the cutoff plus the skin.

    The run scripts use ``comm_style tiled``, which cannot reach more than one
    periodic image, so every perpendicular in-plane width of the cell must
    exceed the interlayer cutoff plus the neighbor skin
    (``settings.simulation.neighbor_list``). ILP/KC also need every ghost
    atom's own neighbours for the layer normals. Other styles pass untouched.

    Args:
        general: The run's ``[general]`` config (``None``: plain LJ).
        lx, ly, xy: The periodic cell (Angstrom), ``xy`` the LAMMPS tilt.
        skin: The neighbor skin (Angstrom).

    Raises:
        ValueError: Naming the cell, its widths and the x/y that would pass.
    """
    style = interlayer_style(general)
    cutoff = INTERLAYER_CUTOFFS.get(style)
    if cutoff is None:
        return
    need = cutoff + skin
    width_x, width_y = periodic_widths(lx, ly, xy)
    if min(width_x, width_y) > need:
        return
    # Replicating the cell keeps xy/ly fixed, so width_x = lx / sqrt(1 + (xy/ly)^2).
    x_needed = need * math.hypot(xy, ly) / ly
    tilt = f", xy tilt {xy:.2f}" if xy else ""
    raise ValueError(
        f"interlayer_style = {style}: the periodic cell is {lx:.2f} x {ly:.2f} A"
        f"{tilt} (perpendicular widths {width_x:.2f} and {width_y:.2f} A), but "
        f"every width must exceed the {style} cutoff {cutoff:g} A + neighbor "
        f"skin {skin:g} A = {need:g} A (comm_style tiled reaches one periodic "
        f"image only). Use x > {x_needed:.1f} A and y > {need:.1f} A; the built "
        f"cell rounds down to whole unit cells, so allow one more.")


def read_data_cell(path: Union[str, Path]) -> Tuple[float, float, float]:
    """``(lx, ly, xy)`` from a LAMMPS data file's header (tilt 0 if absent)."""
    lx = ly = None
    xy = 0.0
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.split('#', 1)[0].strip()
            if line in ('Masses', 'Atoms') or line.startswith('Atoms'):
                break
            parts = line.split()
            if line.endswith('xlo xhi'):
                lx = float(parts[1]) - float(parts[0])
            elif line.endswith('ylo yhi'):
                ly = float(parts[1]) - float(parts[0])
            elif line.endswith('xy xz yz'):
                xy = float(parts[0])
    if lx is None or ly is None:
        raise ValueError(f"{path}: no xlo xhi / ylo yhi box lines in the header")
    return lx, ly, xy


def check_interlayer_box_from_data(general: Any, data_path: Union[str, Path], *,
                                   skin: float) -> None:
    """:func:`check_interlayer_box` on the periodic cell of a built data file."""
    if INTERLAYER_CUTOFFS.get(interlayer_style(general)) is None:
        return
    lx, ly, xy = read_data_cell(data_path)
    check_interlayer_box(general, lx=lx, ly=ly, xy=xy, skin=skin)
