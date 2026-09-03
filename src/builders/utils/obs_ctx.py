"""The observable-registry context every builder passes to the Jinja macros.

Extracted from AFMSimulation._build_obs_ctx so the AFM and sheet families
share one implementation. A fourth hand-rolled copy would recreate, in
Python, exactly the drift this removes from Jinja.

Imports only src.core.observables -- never ase, directly or transitively.
tests/repo/test_import_weight.py pins that.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ...core.observables import (
    REGISTRY, TIER_ARGS, channels_for, layer_channels)

#: Output-schema marker stamped on every results file. Must stay in step with
#: DataReader._LF_SCHEMA_CURRENT; _parse_lf_schema reads the integer, so
#: adding keys (family=) is not a schema bump.
_LF_SCHEMA = 2


def build_obs_ctx(
    *,
    sim_type: str,
    n_layers: int,
    results_file_pattern: str,
    use_langevin: bool,
    enabled_tiers: Dict[str, bool],
    contact_cutoff: float,
    results_freq: int,
    layer_groups_available: bool = True,
    interface_layers: Optional[Tuple[int, int]] = None,
    extra_filter: Optional[Callable[[object], bool]] = None,
    optional_channels: Sequence[str] = (),
) -> Dict[str, object]:
    """Assemble the observable-registry context for the macro library.

    A single dict passed under the ``obs_ctx`` key -- the macros receive it
    explicitly (the Jinja env has no StrictUndefined, so a missing key would
    silently render empty; the contract test's varlist assertions are the
    guard).

    ``extra_filter`` drops channels the registry cannot decide about on its
    own -- notably the interlayer channels under a many-body potential, where
    the template's fallback branch never defines their variables.

    ``interface_layers`` is the ``(il_bot, il_top)`` pair the sliding
    interface sits at, substituted into ``{il_bot}``/``{il_top}`` the same
    way ``{contact_cutoff}`` and ``{n_layers}`` already are. The sheet
    builders pass ``(n_layers // 2, n_layers // 2 + 1)`` -- the identical
    rule ``sheetonsheet/slide.lmp`` uses for its own ``{% set il_bot %}``/
    ``{% set il_top %}``, so the interface has one definition shared between
    Python and Jinja rather than two that can drift.
    """
    static_layer_names = {f'com{ax}_L{lyr}'
                          for lyr in (1, 2) for ax in ('x', 'y', 'z')}
    # The peratom-tier analogue of static_layer_names: pe_sum_L{1,2} and the
    # sigma_{xx,yy}_2d_L{1,2} static bilayer pairs are swapped out for
    # layer_channels(n_layers)'s real-layer-count generation, exactly like
    # the window-tier com{x,y,z}_L{1,2} entries above.
    static_peratom_layer_names = {
        f'{prefix}_L{lyr}' for lyr in (1, 2)
        for prefix in ('pe_sum', 'sigma_xx_2d', 'sigma_yy_2d')
    }
    # Filtered to THIS family: layer_channels() generates com{x,y,z}_L{n}
    # for every family, but pe_sum_L{n}/sigma_{xx,yy}_2d_L{n} declare
    # sim_types=('sheetonsheet', 'hetero', 'flake') only. Without this
    # filter an afm/indent build would splice those sheet-only channels
    # into its own peratom_varlist, which rename_map('afm', 'peratom')
    # cannot name.
    generated_layer_channels = [
        ob for ob in layer_channels(n_layers) if sim_type in ob.sim_types
    ]
    win = [ob for ob in channels_for(sim_type, 'window')
           if ob.name not in static_layer_names]
    pa = [ob for ob in channels_for(sim_type, 'peratom')
          if ob.stream == 'results'
          and ob.name not in static_peratom_layer_names]
    if layer_groups_available:
        win += [ob for ob in generated_layer_channels if ob.tier == 'window']
        pa += [ob for ob in generated_layer_channels if ob.tier == 'peratom']
    if not use_langevin:
        win = [ob for ob in win if not ob.name.startswith('w_thermostat')]

    if not enabled_tiers.get('peratom', True):
        pa = []
    if extra_filter is not None:
        win = [ob for ob in win if extra_filter(ob)]
        pa = [ob for ob in pa if extra_filter(ob)]

    requested = set(optional_channels)
    # generated_layer_channels covers real per-layer names beyond the static
    # L1/L2 REGISTRY entries (e.g. pe_sum_L3 on a 4-layer stack) -- without
    # this, naming one in optional_channels would fail the unknown-name
    # check below even though build_obs_ctx knows exactly how to emit it.
    #
    # known is scoped to THIS family (sim_type in ob.sim_types), not the
    # whole REGISTRY: a name that belongs to another family (e.g.
    # 'n_contact_il' on an afm build) would otherwise be silently accepted
    # and emit nothing, because it is filtered out of win/pa by
    # channels_for(sim_type, ...) above -- the exact "typo means no
    # instrumentation" failure optional_channels exists to foreclose.
    known = ({ob.name for ob in REGISTRY.values() if sim_type in ob.sim_types}
             | {ob.name for ob in generated_layer_channels})
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"optional_channels: no such observable {sorted(unknown)}; "
            f"a typo here silently runs a pilot arm without the "
            f"instrumentation it was commissioned to measure"
        )
    win = [ob for ob in win if not ob.opt_in or ob.name in requested]
    pa = [ob for ob in pa if not ob.opt_in or ob.name in requested]

    def native(ob):
        return ob.rename_to or f'v_{ob.name}'

    cutoff = round(contact_cutoff, 4)

    def _sub(text: str) -> str:
        # {contact_cutoff}: n_contact/n_contact_il's coord/atom cutoff.
        # {n_layers}: the real top-layer group (layer_4, layer_6, ...)
        # behind the registry label 'layer_n' -- see temp_Ltop_3d.
        # {il_bot}/{il_top}: the real sliding-interface layer groups behind
        # n_contact_il's compute -- see interface_layers above. Applied to
        # both compute and variable text since a future channel may need
        # any of these in either field.
        text = (text.replace('{contact_cutoff}', str(cutoff))
                .replace('{n_layers}', str(n_layers)))
        if interface_layers is not None:
            il_bot, il_top = interface_layers
            text = (text.replace('{il_bot}', str(il_bot))
                    .replace('{il_top}', str(il_top)))
        return text

    compute_lines: List[str] = []
    variable_lines: List[str] = []
    for ob in [*win, *pa]:
        if ob.lammps_compute:
            text = _sub(ob.lammps_compute)
            # Split into individual LAMMPS statements (a compute block may
            # be multiple lines, e.g. n_contact's coord/atom + reduce pair)
            # so two Observables that share exactly one statement -- the
            # sigma_{xx,yy}_2d_L{i} pair's `stress/atom` compute -- dedupe
            # correctly even though their surrounding blocks differ.
            for stmt in text.split('\n'):
                if stmt not in compute_lines:
                    compute_lines.append(stmt)
        if ob.lammps_variable:
            variable_lines.append(_sub(ob.lammps_variable))

    window_varlist = ' '.join(native(ob) for ob in win)
    peratom_varlist = ' '.join(native(ob) for ob in pa)
    raw = [ob for ob in channels_for(sim_type, 'raw') if ob.stream == 'results']
    raw_varlist = ' '.join(native(ob) for ob in raw)

    return {
        'compute_lines': compute_lines,
        'variable_lines': variable_lines,
        'window_args': '{} {} {}'.format(
            TIER_ARGS['window'][0], TIER_ARGS['window'][1], results_freq),
        'emit_window': bool(enabled_tiers.get('window', True)) and bool(window_varlist),
        'emit_raw': bool(enabled_tiers.get('raw', True)) and bool(raw_varlist),
        'emit_peratom': bool(enabled_tiers.get('peratom', True)) and bool(peratom_varlist),
        'window_varlist': window_varlist,
        'raw_varlist': raw_varlist,
        'peratom_varlist': peratom_varlist,
        'contact_cutoff': cutoff,
        'results_file_pattern': results_file_pattern,
        'results_file_raw':
            results_file_pattern.replace('friction_', 'raw_'),
        'results_file_peratom':
            results_file_pattern.replace('friction_', 'peratom_'),
        'title1': f'# schema={_LF_SCHEMA} lf_units=nN family={sim_type}',
    }
