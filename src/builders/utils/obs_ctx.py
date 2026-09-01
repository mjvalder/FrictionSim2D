"""The observable-registry context every builder passes to the Jinja macros.

Extracted from AFMSimulation._build_obs_ctx so the AFM and sheet families
share one implementation. A fourth hand-rolled copy would recreate, in
Python, exactly the drift this removes from Jinja.

Imports only src.core.observables -- never ase, directly or transitively.
tests/test_import_weight.py pins that.
"""
from typing import Callable, Dict, List, Optional

from ...core.observables import TIER_ARGS, channels_for, layer_channels

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
    extra_filter: Optional[Callable[[object], bool]] = None,
) -> Dict[str, object]:
    """Assemble the observable-registry context for the macro library.

    A single dict passed under the ``obs_ctx`` key -- the macros receive it
    explicitly (the Jinja env has no StrictUndefined, so a missing key would
    silently render empty; the contract test's varlist assertions are the
    guard).

    ``extra_filter`` drops channels the registry cannot decide about on its
    own -- notably the interlayer channels under a many-body potential, where
    the template's fallback branch never defines their variables.
    """
    static_layer_names = {f'com{ax}_L{lyr}'
                          for lyr in (1, 2) for ax in ('x', 'y', 'z')}
    win = [ob for ob in channels_for(sim_type, 'window')
           if ob.name not in static_layer_names]
    if layer_groups_available:
        win += [ob for ob in layer_channels(n_layers) if ob.tier == 'window']
    if not use_langevin:
        win = [ob for ob in win if not ob.name.startswith('w_thermostat')]
    pa = [ob for ob in channels_for(sim_type, 'peratom')
          if ob.stream == 'results']

    if not enabled_tiers.get('peratom', True):
        pa = []
    if extra_filter is not None:
        win = [ob for ob in win if extra_filter(ob)]
        pa = [ob for ob in pa if extra_filter(ob)]

    def native(ob):
        return ob.rename_to or f'v_{ob.name}'

    cutoff = round(contact_cutoff, 4)
    compute_lines: List[str] = []
    variable_lines: List[str] = []
    for ob in [*win, *pa]:
        if ob.lammps_compute:
            line = ob.lammps_compute.replace('{contact_cutoff}', str(cutoff))
            if line not in compute_lines:
                compute_lines.append(line)
        if ob.lammps_variable:
            variable_lines.append(ob.lammps_variable)

    return {
        'compute_lines': compute_lines,
        'variable_lines': variable_lines,
        'window_args': '{} {} {}'.format(
            TIER_ARGS['window'][0], TIER_ARGS['window'][1], results_freq),
        'emit_window': bool(enabled_tiers.get('window', True)),
        'emit_raw': bool(enabled_tiers.get('raw', True)),
        'emit_peratom': bool(enabled_tiers.get('peratom', True)),
        'window_varlist': ' '.join(native(ob) for ob in win),
        'raw_varlist': ('v_fx_spr v_fy_spr v_fz_tip '
                        'v_comx_tip v_comy_tip v_comz_tip'),
        'peratom_varlist': ' '.join(native(ob) for ob in pa),
        'contact_cutoff': cutoff,
        'results_file_pattern': results_file_pattern,
        'results_file_raw':
            results_file_pattern.replace('friction_', 'raw_'),
        'results_file_peratom':
            results_file_pattern.replace('friction_', 'peratom_'),
        'title1': f'# schema={_LF_SCHEMA} lf_units=nN family={sim_type}',
    }
