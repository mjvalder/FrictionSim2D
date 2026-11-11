"""Builder-side glue for the interlayer-style choice (spec 2026-11-11 §5, §7).

Helpers every family shares:

* :func:`set_interlayer_from_config` hands the run's ``[general]``
  interlayer fields and each sheet's ``interlayer_g`` to
  :meth:`~src.potentials.manager.PotentialManager.set_interlayer`.
* :func:`interlayer_template_context` builds the slide-template keys
  (``needs_molecule_ids``, ``molecule_groups``, ``use_group_group``,
  ``interlayer_energy_substyle``).
* :func:`interlayer_provenance_block` is the provenance record's
  ``interlayer`` block.
"""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ...potentials.interlayer import gaussian_g

#: Styles whose interlayer term ``compute group/group`` can evaluate (it
#: needs ``pair->single()``). ``lj_g`` is provisional: Task 8 checks it
#: under ``kolmogorov/crespi/z``.
GROUP_GROUP_STYLES = ('lj', 'lj_g')


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
        pm: The :class:`~src.potentials.manager.PotentialManager`.
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
    for ``ilp``/``kc``, and whenever comparison mode is on, so every style of
    a comparison sweep reports the same reaction-force friction (spec §7).
    ``interlayer_energy_substyle`` is set only where ``compute pair`` isolates
    the interlayer term (``ilp``/``kc``); under ``lj``/``lj_g`` in comparison
    mode the LJ part shares ``lj/cut`` with the tip/virtual pairs, so the
    energy is dropped there as it is for rebomos.
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
    with comparison mode off, which keeps every lj ``provenance.json``
    byte-identical to one written before the interlayer choice existed; and
    None without a manager to ask.
    """
    comparison = interlayer_comparison(general)
    if pm is None or (interlayer_style(general) == 'lj' and not comparison):
        return None
    block = dict(pm.interlayer_provenance())
    block['comparison'] = comparison
    return block
