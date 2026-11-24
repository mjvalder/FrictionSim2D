"""The registry of simulation output channels, declared once each.

A channel has two names that must agree: the LAMMPS variable a template writes
into its ``fix ave/time`` and the name the parser renames that column back to.

Sampling tiers (``fix ave/time`` args in ``TIER_ARGS``): ``window`` samples
every step and block-means over 1000, ``raw`` takes one instantaneous sample
per 100 steps, ``peratom`` runs the costly per-atom computes once per 1000.
``group/group`` rebuilds a neighbour list per call, so ``validate_registry``
bans it from the window tier.
"""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

TIERS = ('window', 'raw', 'peratom')
STREAMS = ('results', 'post')

#: Every schema a results file can follow. Distinct from `run.naming.MODELS`
#: (the CLI / top-directory axis): a run has a `model` (where it lives) and a
#: `family` (what schema its files follow).
FAMILIES: Tuple[str, ...] = (
    'afm', 'indent', 'sheetonsheet', 'hetero', 'flake', 'pes_tip', 'pes_sheet',
)

#: The AFM stack (tip on sheet) and the sheet-on-sheet stack: the two schemas
#: a family's results files follow.
AFM_LIKE: Tuple[str, ...] = ('afm', 'indent', 'pes_tip')
SHEET_LIKE: Tuple[str, ...] = ('sheetonsheet', 'hetero', 'flake', 'pes_sheet')

# fix ave/time argument triples (Nevery, Nrepeat, Nfreq), validated against LAMMPS
TIER_ARGS = {
    'window': (1, 1000, 1000),
    'raw': (100, 1, 100),
    'peratom': (1000, 1, 1000),
}

#: Length (steps) of the constant-load hold in ``afm/system_init.lmp``; the
#: AFM builder renders it into the template, so the script and the exported
#: config (`config.simulation.afm_hold_steps`) cannot drift.
AFM_HOLD_STEPS: int = 10000

#: Families whose `window` tier spans exactly the slide run (`fix fc_win` is
#: defined right before `run {{ run_steps }}` in the slide templates). Indent's
#: window covers the hold phase and PES writes a grid, so no expected length is
#: derived for them.
SLIDE_FAMILIES: Tuple[str, ...] = ('afm', 'sheetonsheet', 'hetero', 'flake')

#: eV/Å → nanoNewton (force conversion).  1 eV = 1.602176565e-19 J,
#: 1 Å = 1e-10 m  ⇒  1 eV/Å = 1.602176565 nN.
EV_A_TO_NN: float = 1.602176565

#: The static potential-energy-surface scan families: one grid evaluation per run.
PES_FAMILIES: Tuple[str, ...] = ('pes_tip', 'pes_sheet')

#: The grid a PES run writes under `results/`, and the optional refinement grid.
PES_BASE_NAME = 'pes_scan.csv'
PES_REFINE_NAME = 'pes_scan_refine.csv'

def tier_frequencies(config: Mapping[str, Any]) -> Dict[str, int]:
    """Output interval (steps) of each tier; `{}` if the window one is unknown."""
    try:
        every = int(config['settings']['output']['results_frequency'])
    except (KeyError, TypeError, ValueError, AttributeError):
        return {}
    if every <= 0:
        return {}
    return {'window': every,
            'raw': TIER_ARGS['raw'][2],
            'peratom': TIER_ARGS['peratom'][2]}


def expected_tier_rows(config: Mapping[str, Any], family: str) -> Dict[str, int]:
    """Rows each enabled tier writes over a finished slide, `{}` if unknown.

    `slide_run_steps // frequency`, from the run's own stored config (its
    `settings` block). `{}` for a non-slide family or any missing or
    non-positive value. An absent `output.channels` means every tier is on.
    """
    if family not in SLIDE_FAMILIES:
        return {}
    freqs = tier_frequencies(config)
    try:
        steps = int(config['settings']['simulation']['slide_run_steps'])
        channels = config['settings']['output'].get('channels')
    except (KeyError, TypeError, ValueError, AttributeError):
        return {}
    if not freqs or steps <= 0:
        return {}
    return {tier: steps // every for tier, every in freqs.items()
            if channels is None or channels.get(tier, True)}


@dataclass(frozen=True)
class Observable:
    """One output channel.

    ``rename_to`` is the native LAMMPS column name (``v_fz_tip``) the parser
    maps back to ``name``. No ``lammps_compute`` means the template owns the
    channel and emits it itself, often conditionally; the registry only has to
    parse it back.

    ``units`` is the declared physical unit of the emitted column, in LAMMPS
    ``metal`` terms ('nN', 'eV', 'eV/A', 'A', 'K', 'count'); nothing verifies
    it against what a template emits.
    """
    name: str
    sim_types: Tuple[str, ...]
    tier: str
    stream: str
    lammps_compute: Optional[str] = None
    lammps_variable: Optional[str] = None
    rename_to: Optional[str] = None
    units: Optional[str] = None
    key: Optional[str] = None          # registry key; defaults to name
    role_key: Optional[str] = None     # family-agnostic semantic identity
    opt_in: bool = False               # costs runtime; emitted only on request

    @property
    def registry_key(self) -> str:
        """Key in REGISTRY. Distinct from `name` only where two families
        share a name -- `lfx` is native `v_fx_spr` for AFM and `v_xfrict`
        for the sheet, and both must parse back to the column downstream
        consumers key on."""
        return self.key or self.name


REGISTRY: dict = {}

#: Results channels that carry no cross-family role worth naming; the parity
#: tests never see them.
ROLE_EXEMPT: dict = {
    'tip_min_z':
        'punch-through detection is specific to a rigid indenting body',
}


def _add(ob: Observable) -> None:
    if ob.stream == 'results' and not ob.role_key \
            and ob.name not in ROLE_EXEMPT:
        raise ValueError(
            f"{ob.registry_key}: results channels need a role_key (or an "
            f"explicit ROLE_EXEMPT entry) so parity is checkable"
        )
    if ob.registry_key in REGISTRY:
        raise ValueError(f"duplicate observable key {ob.registry_key}")
    for other in REGISTRY.values():
        if other.name != ob.name:
            continue
        clash = set(other.sim_types) & set(ob.sim_types)
        if clash:
            raise ValueError(
                f"{ob.name}: two observables may share a name only if their "
                f"sim_types are disjoint; overlaps with "
                f"{other.registry_key} on {sorted(clash)}"
            )
    REGISTRY[ob.registry_key] = ob


def _com_layer(lyr: int, ax: str) -> Observable:
    """Per-layer COM channel (window). Shared by the static REGISTRY entries
    and layer_channels() so the two can never drift apart."""
    return Observable(
        name=f'com{ax}_L{lyr}',
        sim_types=('afm', 'sheetonsheet', 'indent', 'hetero', 'flake'),
        tier='window', stream='results',
        lammps_compute=f'compute COM_L{lyr} layer_{lyr} com',
        lammps_variable=(f'variable com{ax}_L{lyr} equal '
                         f'c_COM_L{lyr}[{"xyz".index(ax) + 1}]'),
        rename_to=f'v_com{ax}_L{lyr}', units='A',
        role_key=f'layer_com_{ax}_L{lyr}')


# --- legacy window channels: the first nine columns, in the 2026 campaign's
# order. Real files carry full headers, so all nine need rename_to. ------------
_LEGACY_ROLE_KEYS = {
    'nf': 'normal_load',
    'lfx': 'lateral_force_x', 'lfy': 'lateral_force_y',
    'comx': 'sheet_com_x', 'comy': 'sheet_com_y', 'comz': 'sheet_com_z',
    'tipx': 'driven_com_x', 'tipy': 'driven_com_y', 'tipz': 'driven_com_z',
}
for _name, _native in [
    ('nf', 'v_fz_tip'), ('lfx', 'v_fx_spr'), ('lfy', 'v_fy_spr'),
    ('comx', 'v_comx'), ('comy', 'v_comy'), ('comz', 'v_comz'),
    ('tipx', 'v_comx_tip'), ('tipy', 'v_comy_tip'), ('tipz', 'v_comz_tip'),
]:
    _add(Observable(
        name=_name,
        sim_types=('afm', 'indent'),
        tier='window', stream='results',
        rename_to=_native,
        units='nN' if _name in ('nf', 'lfx', 'lfy') else 'A',
        role_key=_LEGACY_ROLE_KEYS[_name]))

# --- layer-resolved COM: bilayer entries; builders inject layer_channels(n_layers). ---
for _lyr in (1, 2):
    for _ax in ('x', 'y', 'z'):
        _add(_com_layer(_lyr, _ax))

# --- energies and temperatures ------------------------------------------------
_add(Observable(name='etotal', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                tier='window', stream='results',
                lammps_variable='variable etot equal etotal',
                rename_to='v_etot', units='eV',
                role_key='total_energy'))
_add(Observable(name='pe', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                tier='window', stream='results',
                lammps_variable='variable pe_now equal pe',
                rename_to='v_pe_now', units='eV',
                role_key='potential_energy'))
_add(Observable(name='ke', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                tier='window', stream='results',
                lammps_variable='variable ke_now equal ke',
                rename_to='v_ke_now', units='eV',
                role_key='kinetic_energy'))
# Cumulative thermostat work; needs `tally yes` on the langevin fixes.
_add(Observable(name='w_thermostat_tip', sim_types=('afm', 'indent'),
                tier='window', stream='results',
                lammps_variable='variable w_lang_tip equal f_lang_tip',
                rename_to='v_w_lang_tip', units='eV',
                role_key='thermostat_work_tip'))
_add(Observable(name='w_thermostat_sub', sim_types=('afm', 'indent'),
                tier='window', stream='results',
                lammps_variable='variable w_lang_sub equal f_lang_sub',
                rename_to='v_w_lang_sub', units='eV',
                role_key='thermostat_work_substrate'))
# The AFM sheet is pure NVE, so its 3D temperature measures frictional heating
# rather than tracking a setpoint.
_add(Observable(name='temp_sheet_3d', sim_types=('afm', 'indent'),
                tier='window', stream='results',
                lammps_compute='compute T_sheet 2D_all temp',
                lammps_variable='variable temp_sheet equal c_T_sheet',
                rename_to='v_temp_sheet', units='K',
                role_key='group_temperature'))
# Cumulative thermostat work for the sheet stack; needs `tally yes` on the
# langevin fix (build_obs_ctx drops it under nose-hoover). Its own role, not
# AFM's thermostat_work_tip: the sheet's thermostatted group is the stack
# interior, while the driven top layer is never thermostatted, so sharing a
# role would let a role-keyed consumer misread it as dissipation at the contact.
_add(Observable(name='w_thermostat_center',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                lammps_variable='variable w_lang_center equal f_lang',
                rename_to='v_w_lang_center', units='eV',
                role_key='thermostat_work_interior'))
# Group temperatures for the sheet stack: `center` (thermostatted interior),
# `layer_1` (fixed bottom band) and the driven top layer. The top layer's
# compute carries a `{n_layers}` placeholder that build_obs_ctx substitutes,
# like `{contact_cutoff}`. Diagnostics, not setpoint checks.
_add(Observable(name='temp_center_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                lammps_compute='compute T_center center temp',
                lammps_variable='variable temp_center_3d equal c_T_center',
                rename_to='v_temp_center_3d', units='K',
                role_key='group_temperature_center'))
_add(Observable(name='temp_L1_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                lammps_compute='compute T_layer_1 layer_1 temp',
                lammps_variable='variable temp_L1_3d equal c_T_layer_1',
                rename_to='v_temp_L1_3d', units='K',
                role_key='group_temperature_layer_1'))
_add(Observable(name='temp_Ltop_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                lammps_compute='compute T_Ltop layer_{n_layers} temp',
                lammps_variable='variable temp_Ltop_3d equal c_T_Ltop',
                rename_to='v_temp_Ltop_3d', units='K',
                role_key='group_temperature_layer_n'))
# Tip-sheet coupling is purely lj/cut, so group/group works here even under the
# hybrid pair style.
_add(Observable(name='E_int_tip_sheet', sim_types=('afm', 'indent'),
                tier='peratom', stream='results',
                lammps_compute='compute eint tip_all group/group 2D_all pair yes',
                lammps_variable='variable e_int equal c_eint',
                rename_to='v_e_int', units='eV',
                role_key='tip_sheet_interaction'))
# Template-owned (behind the slide template's has_internal_lj branch): the
# `compute group/group` energy of the shearing interface only (il_top against
# il_bot), for lj / lj_g runs.
_add(Observable(name='E_int_interlayer', sim_types=('sheetonsheet', 'hetero'),
                tier='peratom', stream='results',
                rename_to='v_e_int_il', units='eV',
                role_key='interlayer_interaction'))
# The ilp / kc counterpart: a DIFFERENT channel, since `compute pair` sums the
# interlayer term over every layer pair of the stack, not just the shearing
# interface (equal only on a bilayer). opt_in; the builders request it only
# when the template defines v_e_int_il_all.
_add(Observable(name='E_int_interlayer_all', sim_types=('sheetonsheet', 'hetero'),
                tier='peratom', stream='results', opt_in=True,
                rename_to='v_e_int_il_all', units='eV',
                role_key='interlayer_interaction_all_pairs'))
# --- sheet window channels, every native already emitted by
# sheetonsheet/slide.lmp's fc_win. The lateral channel keeps its group/group
# compute at Nevery=1 (it is the friction label itself); it is template-owned,
# so validate_registry's window-tier ban never sees it.
_add(Observable(name='lfx', key='lfx@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                rename_to='v_xfrict', units='nN',
                role_key='lateral_force_x'))
_add(Observable(name='lfy', key='lfy@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                rename_to='v_yfrict', units='nN',
                role_key='lateral_force_y'))
# 'flake' too: flake/slide.lmp loads the flake with `fix force flake addforce`
# and emits v_fz = f_force[3] in nN, so a flake COF is lateral force / normal load.
_add(Observable(name='nf', key='nf@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='window', stream='results',
                rename_to='v_fz', units='nN', role_key='normal_load'))
# nf and drive_fx/drive_fy come from f_force[1..3] on one fix aveforce; v_fz is
# named nf because every COF consumer expects it. The flake template defines
# only v_fz, so drive_fx/fy stay sheet/hetero.
for _ax, _native in (('x', 'v_fx'), ('y', 'v_fy')):
    _add(Observable(name=f'drive_f{_ax}', sim_types=('sheetonsheet', 'hetero'),
                    tier='window', stream='results',
                    rename_to=_native, units='nN',
                    role_key=f'drive_force_{_ax}'))
# drive{x,y,z} alias the top layer's COM, redundant with com{x,y,z}_L{n}; kept
# for the driven_com_* role that pairs with AFM's tipx/tipy/tipz.
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'drive{_ax}', sim_types=('sheetonsheet', 'hetero'),
                    tier='window', stream='results',
                    rename_to=f'v_com{_ax}_top', units='A',
                    role_key=f'driven_com_{_ax}'))
# The flake's driven COM is the flake's own, under the same driven_com_* role
# as the sheet's drive{x,y,z} and AFM's tip*. A distinct `key` is needed: the
# name is shared (sim_types are disjoint) but the registry key may not be.
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'drive{_ax}', key=f'drive{_ax}@flake',
                    sim_types=('flake',),
                    tier='window', stream='results',
                    rename_to=f'v_com{_ax}_flake', units='A',
                    role_key=f'driven_com_{_ax}'))
# Reaction force on the fixed bottom layer. react_x is also the legacy LF
# correction's LJ-branch discriminator (read_data._correct_legacy_lf_schema1).
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'react_{_ax}',
                    sim_types=('sheetonsheet', 'hetero', 'flake'),
                    tier='window', stream='results',
                    rename_to=f'v_s{_ax}', units='nN',
                    role_key=f'reaction_force_{_ax}'))

# --- sheet raw tier: registry-derived, so these drive emission as well as parsing.
for _name, _key, _native, _role, _unit in [
    ('raw_lfx', 'raw_lfx@sheet', 'v_xfrict', 'raw_lateral_force_x', 'nN'),
    ('raw_lfy', 'raw_lfy@sheet', 'v_yfrict', 'raw_lateral_force_y', 'nN'),
    ('raw_nf', 'raw_nf@sheet', 'v_fz', 'raw_normal_load', 'nN'),
    ('raw_drivex', None, 'v_comx_top', 'raw_driven_com_x', 'A'),
    ('raw_drivey', None, 'v_comy_top', 'raw_driven_com_y', 'A'),
]:
    _add(Observable(name=_name, key=_key, sim_types=('sheetonsheet', 'hetero'),
                    tier='raw', stream='results',
                    rename_to=_native, units=_unit, role_key=_role))

_PE_SUM_ROLE_KEYS = {
    'tip_all': 'group_pe_sum_driven',
    'sub_all': 'group_pe_sum_substrate',
    '2D_all': 'group_pe_sum_membrane',
}
for _grp in ('tip_all', 'sub_all', '2D_all'):
    _add(Observable(
        name=f'pe_sum_{_grp}', sim_types=('afm', 'indent'),
        tier='peratom', stream='results',
        lammps_compute=(f'compute peat_{_grp} {_grp} pe/atom\n'
                        f'compute pesum_{_grp} {_grp} reduce sum c_peat_{_grp}'),
        lammps_variable=f'variable pe_{_grp} equal c_pesum_{_grp}',
        rename_to=f'v_pe_{_grp}', units='eV',
        role_key=_PE_SUM_ROLE_KEYS[_grp]))

# --- contact ------------------------------------------------------------------
_add(Observable(name='n_contact', sim_types=('afm', 'indent'),
                tier='peratom', stream='results',
                lammps_compute=('compute coordc 2D_all coord/atom cutoff '
                                '{contact_cutoff} group tip_all\n'
                                'compute contacts 2D_all reduce sum c_coordc'),
                lammps_variable='variable n_contact equal c_contacts',
                rename_to='v_n_contact', units='count',
                role_key='contact_count'))
# In-plane virial sums; post normalises them to a membrane stress resultant
# (N/m). No sigma_zz: for a slab with traction-free faces, its through-thickness
# integral is not a stress resultant.
_add(Observable(name='sigma_xx_2d', sim_types=('afm', 'indent'),
                tier='peratom', stream='results',
                lammps_compute=('compute satom 2D_all stress/atom NULL\n'
                                'compute sxx 2D_all reduce sum c_satom[1]'),
                lammps_variable='variable sxx_2d equal c_sxx',
                rename_to='v_sxx_2d', units='eV/A',
                role_key='membrane_stress_xx'))
_add(Observable(name='sigma_yy_2d', sim_types=('afm', 'indent'),
                tier='peratom', stream='results',
                lammps_compute='compute syy 2D_all reduce sum c_satom[2]',
                lammps_variable='variable syy_2d equal c_syy',
                rename_to='v_syy_2d', units='eV/A',
                role_key='membrane_stress_yy'))
# post-stream: derived from retained artefacts, never a LAMMPS column
_add(Observable(name='deflect_max', sim_types=('afm', 'indent'),
                tier='peratom', stream='post', units='A',
                role_key='max_deflection'))
_add(Observable(name='gap_min', sim_types=('afm', 'indent'),
                tier='peratom', stream='post', units='A',
                role_key='min_gap'))


def _pe_sum_layer(lyr: int) -> Observable:
    """Per-layer group PE sum, opt-in; see :func:`_com_layer`."""
    return Observable(
        name=f'pe_sum_L{lyr}', sim_types=('sheetonsheet', 'hetero', 'flake'),
        tier='peratom', stream='results', opt_in=True,
        lammps_compute=(f'compute peat_L{lyr} layer_{lyr} pe/atom\n'
                        f'compute pesum_L{lyr} layer_{lyr} reduce sum '
                        f'c_peat_L{lyr}'),
        lammps_variable=f'variable pe_sum_L{lyr} equal c_pesum_L{lyr}',
        rename_to=f'v_pe_sum_L{lyr}', units='eV',
        role_key=f'group_pe_sum_L{lyr}')


def _sigma_layer(lyr: int, axis: str) -> Observable:
    """Per-layer in-plane stress sum (``axis`` 'xx' or 'yy'), opt-in."""
    return Observable(
        name=f'sigma_{axis}_2d_L{lyr}',
        sim_types=('sheetonsheet', 'hetero', 'flake'),
        tier='peratom', stream='results', opt_in=True,
        lammps_compute=(f'compute satom_L{lyr} layer_{lyr} stress/atom NULL\n'
                        f'compute s{axis}_L{lyr} layer_{lyr} reduce sum '
                        f'c_satom_L{lyr}[{1 if axis == "xx" else 2}]'),
        lammps_variable=f'variable sigma_{axis}_2d_L{lyr} equal c_s{axis}_L{lyr}',
        rename_to=f'v_sigma_{axis}_2d_L{lyr}', units='eV/A',
        role_key=f'membrane_stress_{axis}_L{lyr}')


# --- opt-in sheet peratom/post channels: gated by settings.output.optional_channels
# (a default build emits none), since they cost per-atom compute time. No
# sigma_zz_L{i}: for a slab with traction-free faces the through-thickness
# integral is not a stress resultant. The static L1/L2 entries serve the
# registry tests; layer_channels() generates the real layer count.
_add(_pe_sum_layer(1))
_add(_pe_sum_layer(2))
# Each sigma layer channel declares its own full `stress/atom` compute, since
# either may be requested alone; build_obs_ctx dedupes identical statements.
for _axis in ('xx', 'yy'):
    for _lyr in (1, 2):
        _add(_sigma_layer(_lyr, _axis))

# n_contact_il: contact count across the sliding interface; its compute carries
# {il_top}/{il_bot} placeholders that build_obs_ctx substitutes. Unavailable
# where there is no interlayer lj/cut sigma to derive a cutoff from, the same
# materials that lose E_int_interlayer.
_add(Observable(name='n_contact_il', sim_types=('sheetonsheet', 'hetero'),
                tier='peratom', stream='results', opt_in=True,
                lammps_compute=(
                    'compute coordil layer_{il_top} coord/atom cutoff '
                    '{contact_cutoff} group layer_{il_bot}\n'
                    'compute contactsil layer_{il_top} reduce sum c_coordil'),
                lammps_variable='variable n_contact_il equal c_contactsil',
                rename_to='v_n_contact_il', units='count',
                role_key='contact_count'))

# post-stream: derived in postprocessing, never a LAMMPS column.
_add(Observable(name='gap_min_il', sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='peratom', stream='post', opt_in=True,
                units='A', role_key='min_gap'))
_add(Observable(name='corrugation_max',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                tier='peratom', stream='post', opt_in=True,
                units='A', role_key='max_deflection'))

# --- raw trace (100 fs instantaneous samples). The native names are declared here
# so build_obs_ctx derives the raw varlist from the registry. ------------------
for _name, _native, _role, _unit in [
    ('raw_lfx', 'v_fx_spr', 'raw_lateral_force_x', 'nN'),
    ('raw_lfy', 'v_fy_spr', 'raw_lateral_force_y', 'nN'),
    ('raw_nf', 'v_fz_tip', 'raw_normal_load', 'nN'),
    ('raw_tipx', 'v_comx_tip', 'raw_driven_com_x', 'A'),
    ('raw_tipy', 'v_comy_tip', 'raw_driven_com_y', 'A'),
    ('raw_tipz', 'v_comz_tip', 'raw_driven_com_z', 'A'),
]:
    _add(Observable(name=_name, sim_types=('afm', 'indent'), tier='raw',
                    stream='results', rename_to=_native, units=_unit,
                    role_key=_role))

# --- punch-through ingredient: min tip z, in the SLIDING script --------------
_add(Observable(name='tip_min_z', sim_types=('afm',),
                tier='window', stream='results',
                lammps_compute='compute cz_tip_s tip_all reduce min z',
                lammps_variable='variable tip_min_z equal c_cz_tip_s',
                rename_to='v_tip_min_z', units='A'))


def channels_for(sim_type: str, tier: str) -> list:
    """Results-stream channels for one (sim_type, tier)."""
    return [ob for ob in REGISTRY.values()
            if sim_type in ob.sim_types and ob.tier == tier
            and ob.stream == 'results']


def channel_by_role(sim_type: str, role_key: str):
    """The channel serving `role_key` in `sim_type`, or None.

    Consumers that want to be family-agnostic (COF, the DB writer, the ML
    extractors) can key on roles instead of branching on sim_type.
    """
    for ob in REGISTRY.values():
        if sim_type in ob.sim_types and ob.role_key == role_key:
            return ob
    return None


def rename_map(sim_type: str, tier: str, n_layers: int = 2) -> dict:
    """Native column name -> channel name, for one tier and layer count.

    The raw file's header repeats the window file's native names, so those map
    to raw_* in the raw map only. Per-layer channels beyond the registry's L1/L2
    come from :func:`layer_channels`.
    """
    m = {}
    for ob in REGISTRY.values():
        if sim_type not in ob.sim_types:
            continue
        if ob.tier == tier and ob.rename_to:
            m[ob.rename_to] = ob.name
    for ob in layer_channels(max(int(n_layers), 0)):
        if sim_type not in ob.sim_types:
            continue
        if ob.tier == tier and ob.rename_to:
            m[ob.rename_to] = ob.name
    return m


def layer_channels(n_layers: int) -> list:
    """com{x,y,z}_L{n} plus the opt-in per-layer peratom channels, for any layer count.

    build_obs_ctx swaps these in for the registry's static L1/L2 entries.
    """
    out = []
    for lyr in range(1, n_layers + 1):
        for ax in ('x', 'y', 'z'):
            out.append(_com_layer(lyr, ax))
        out.append(_pe_sum_layer(lyr))
        out.append(_sigma_layer(lyr, 'xx'))
        out.append(_sigma_layer(lyr, 'yy'))
    return out


def validate_registry() -> None:
    """Raise on a malformed entry or a window-tier group/group compute."""
    for name, ob in REGISTRY.items():
        if ob.tier not in TIERS or ob.stream not in STREAMS:
            raise ValueError(f"malformed observable {name}")
        if ob.stream == 'results' and ob.tier == 'window' \
                and 'group/group' in (ob.lammps_compute or ''):
            raise ValueError(
                f"{name}: group/group may not run at Nevery=1")


validate_registry()
