"""Single source of truth for every simulation output channel.

Spec: docs/superpowers/specs/2026-07-29-data-quality-instrumentation-design.md §3.

Three consumers are generated from this module: the Jinja macro library
(templates/common/observables.lmp), the header-driven parser
(postprocessing/read_data.py), and the provenance schema (data/provenance.py).

Sampling tiers (spec §4): the `window` tier samples every step and block-means
over 1000 (as the 2026 campaign did); `raw` takes one instantaneous sample per
100 steps; `peratom` evaluates the expensive per-atom computes once per 1000
steps. `compute group/group` rebuilds a neighbour list per invocation and must
never run at Nevery=1 — validate_registry() enforces this.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

TIERS = ('window', 'raw', 'peratom')
STREAMS = ('results', 'provenance', 'post')
ROLES = ('descriptor', 'diagnostic', 'provenance')
PHASES = ('slide', 'hold', 'both')

# fix ave/time argument triples (Nevery, Nrepeat, Nfreq), validated against LAMMPS
TIER_ARGS = {
    'window': (1, 1000, 1000),
    'raw': (100, 1, 100),
    'peratom': (1000, 1, 1000),
}


@dataclass(frozen=True)
class Observable:
    """One output channel, declared once.

    ``rename_to`` is the LAMMPS-native column name (e.g. ``v_fz_tip``) that the
    parser maps back to ``name``. ``phase`` says which script emits the channel:
    system_init's constant-load hold is the only non-slide emission site.
    ``condition`` is an informational marker (e.g. ``is_sheet_lj``); presence or
    absence of conditional channels is template-owned and absorbed by
    header-driven parsing.
    """
    name: str
    units: str
    sim_types: Tuple[str, ...]
    group: str
    tier: str
    stream: str
    channel_role: str
    lammps_compute: Optional[str] = None
    lammps_variable: Optional[str] = None
    rename_to: Optional[str] = None
    phase: str = 'slide'
    condition: Optional[str] = None


REGISTRY: dict = {}


def _add(ob: Observable) -> None:
    if ob.name in REGISTRY:
        raise ValueError(f"duplicate observable {ob.name}")
    REGISTRY[ob.name] = ob


# --- legacy window channels (order preserved: the first nine window columns
# reproduce the 2026 campaign layout, spec §10 back-compat). EVERY entry needs
# rename_to: real 2026 files carry full headers (# TimeStep v_fz_tip ...
# v_comx v_comy v_comz ...), so the header path is the one that parses them —
# comx with rename_to=None would leave a v_comx column downstream. -----------
for _name, _native in [
    ('nf', 'v_fz_tip'), ('lfx', 'v_fx_spr'), ('lfy', 'v_fy_spr'),
    ('comx', 'v_comx'), ('comy', 'v_comy'), ('comz', 'v_comz'),
    ('tipx', 'v_comx_tip'), ('tipy', 'v_comy_tip'), ('tipz', 'v_comz_tip'),
]:
    _add(Observable(
        name=_name,
        units='nN' if _name in ('nf', 'lfx', 'lfy') else 'angstrom',
        sim_types=('afm', 'indent'),
        group='tip_fix' if _name.startswith(('nf', 'lf', 'tip')) else '2D_all',
        tier='window', stream='results', channel_role='descriptor',
        rename_to=_native))

# --- F1 layer-resolved COM: static bilayer entries for golden tests. Builders
# inject layer_channels(n_layers) for the material's real layer count. --------
for _lyr in (1, 2):
    for _ax in ('x', 'y', 'z'):
        _add(Observable(
            name=f'com{_ax}_L{_lyr}', units='angstrom',
            sim_types=('afm', 'sheetonsheet', 'indent'),
            group=f'layer_{_lyr}', tier='window', stream='results',
            channel_role='descriptor',
            lammps_compute=f'compute COM_L{_lyr} layer_{_lyr} com',
            lammps_variable=(f'variable com{_ax}_L{_lyr} equal '
                             f'c_COM_L{_lyr}[{"xyz".index(_ax) + 1}]'),
            rename_to=f'v_com{_ax}_L{_lyr}'))

# --- F2 energies and temperatures --------------------------------------------
_add(Observable(name='etotal', units='eV', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results', channel_role='diagnostic',
                lammps_variable='variable etot equal etotal', rename_to='v_etot'))
_add(Observable(name='pe', units='eV', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results', channel_role='diagnostic',
                lammps_variable='variable pe_now equal pe', rename_to='v_pe_now'))
_add(Observable(name='ke', units='eV', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results', channel_role='diagnostic',
                lammps_variable='variable ke_now equal ke', rename_to='v_ke_now'))
# Cumulative thermostat work: requires `tally yes` on the langevin fixes. A
# healthy run's thermostat work grows linearly (F2 detector #1).
_add(Observable(name='w_thermostat_tip', units='eV', sim_types=('afm', 'indent'),
                group='tip_thermo', tier='window', stream='results',
                channel_role='diagnostic',
                lammps_variable='variable w_lang_tip equal f_lang_tip',
                rename_to='v_w_lang_tip'))
_add(Observable(name='w_thermostat_sub', units='eV', sim_types=('afm', 'indent'),
                group='sub_thermo', tier='window', stream='results',
                channel_role='diagnostic',
                lammps_variable='variable w_lang_sub equal f_lang_sub',
                rename_to='v_w_lang_sub'))
# The AFM sheet is pure NVE (no setpoint): its 3D temperature is a
# frictional-heating observable, not a controlled parameter (spec §2.3).
_add(Observable(name='temp_sheet_3d', units='K', sim_types=('afm', 'indent'),
                group='2D_all', tier='window', stream='results',
                channel_role='diagnostic',
                lammps_compute='compute T_sheet 2D_all temp',
                lammps_variable='variable temp_sheet equal c_T_sheet',
                rename_to='v_temp_sheet'))
# True tip-sheet interaction energy: tip<->sheet coupling is purely lj/cut, so
# group/group works under the hybrid pair style (spec §5 F2, verified).
_add(Observable(name='E_int_tip_sheet', units='eV', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='results',
                channel_role='descriptor',
                lammps_compute='compute eint tip_all group/group 2D_all pair yes',
                lammps_variable='variable e_int equal c_eint',
                rename_to='v_e_int'))
# Template-owned conditional channel: the compute lives in sheetonsheet's slide
# template behind the existing has_internal_lj branch with real per-material
# group names. No lammps_compute here.
_add(Observable(name='E_int_interlayer', units='eV', sim_types=('sheetonsheet',),
                group='layer_n', tier='peratom', stream='results',
                channel_role='descriptor', condition='is_sheet_lj',
                rename_to='v_e_int_il'))
# Sheet friction label: compute frict ... group/group referenced by the window
# fix at Nevery=1 — a deliberate, documented exemption from the §4 rule (the
# label itself, cost already paid campaign-wide). Template-owned: no
# lammps_compute here, so validate_registry's group/group rule is not violated.
_add(Observable(name='xfrict', units='eV/angstrom', sim_types=('sheetonsheet',),
                group='layer_n', tier='window', stream='results',
                channel_role='descriptor', rename_to='v_xfrict'))
for _grp in ('tip_all', 'sub_all', '2D_all'):
    _add(Observable(
        name=f'pe_sum_{_grp}', units='eV', sim_types=('afm', 'indent'),
        group=_grp, tier='peratom', stream='results', channel_role='diagnostic',
        lammps_compute=(f'compute peat_{_grp} {_grp} pe/atom\n'
                        f'compute pesum_{_grp} {_grp} reduce sum c_peat_{_grp}'),
        lammps_variable=f'variable pe_{_grp} equal c_pesum_{_grp}',
        rename_to=f'v_pe_{_grp}'))

# --- F3 contact ---------------------------------------------------------------
_add(Observable(name='n_contact', units='count', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                channel_role='descriptor',
                lammps_compute=('compute coordc 2D_all coord/atom cutoff '
                                '{contact_cutoff} group tip_all\n'
                                'compute contacts 2D_all reduce sum c_coordc'),
                lammps_variable='variable n_contact equal c_contacts',
                rename_to='v_n_contact'))
# In-plane virial sums, normalised to a membrane stress resultant (N/m) in
# post. sigma_zz is deliberately absent: a through-thickness integral of
# sigma_zz for a slab with traction-free faces is not a stress resultant.
_add(Observable(name='sigma_xx_2d', units='raw_virial', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                channel_role='descriptor',
                lammps_compute=('compute satom 2D_all stress/atom NULL\n'
                                'compute sxx 2D_all reduce sum c_satom[1]'),
                lammps_variable='variable sxx_2d equal c_sxx',
                rename_to='v_sxx_2d'))
_add(Observable(name='sigma_yy_2d', units='raw_virial', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                channel_role='descriptor',
                lammps_compute='compute syy 2D_all reduce sum c_satom[2]',
                lammps_variable='variable syy_2d equal c_syy',
                rename_to='v_syy_2d'))
# post-stream channels: derived from retained artefacts, never a LAMMPS column
_add(Observable(name='deflect_max', units='angstrom', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='post',
                channel_role='descriptor'))
_add(Observable(name='gap_min', units='angstrom', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='post',
                channel_role='descriptor'))

# --- F4 raw trace (100 fs instantaneous samples) -------------------------------
for _name in ('raw_lfx', 'raw_lfy', 'raw_nf', 'raw_tipx', 'raw_tipy', 'raw_tipz'):
    _add(Observable(
        name=_name,
        units='nN' if 'f' in _name.split('_')[1][:2] else 'angstrom',
        sim_types=('afm',), group='tip_fix', tier='raw', stream='results',
        channel_role='descriptor'))

# --- hold diagnostics (spec §6) — phase='hold': emitted ONLY by system_init.
# channel_role='diagnostic' is load-bearing: production-run hold channels are
# forbidden as ML features (they share the run with the COF label). ------------
for _name in ('hold_pen', 'hold_n_contact', 'hold_fz'):
    _add(Observable(name=_name,
                    units='angstrom' if _name == 'hold_pen' else 'mixed',
                    sim_types=('afm',), group='tip_fix', tier='window',
                    stream='results', channel_role='diagnostic', phase='hold'))

# --- punch-through ingredient (spec §7): min tip z in the SLIDING script -------
_add(Observable(name='tip_min_z', units='angstrom', sim_types=('afm',),
                group='tip_all', tier='window', stream='results',
                channel_role='diagnostic',
                lammps_compute='compute cz_tip_s tip_all reduce min z',
                lammps_variable='variable tip_min_z equal c_cz_tip_s',
                rename_to='v_tip_min_z'))


def channels_for(sim_type: str, tier: str, phase: str = 'slide') -> list:
    """Results-stream channels for one (sim_type, tier, phase)."""
    return [ob for ob in REGISTRY.values()
            if sim_type in ob.sim_types and ob.tier == tier
            and ob.stream == 'results'
            and ob.phase in (phase, 'both')]


def expected_columns(sim_type: str, tier: str, phase: str = 'slide') -> list:
    return [ob.name for ob in channels_for(sim_type, tier, phase)]


def rename_map(sim_type: str, tier: str) -> dict:
    """Per-tier rename maps.

    The raw file's header carries the SAME native names (v_fx_spr...) as the
    window file, but they must map to raw_lfx... only in the raw map. The
    parser picks the tier from the filename prefix (friction_/fc_ave_ ->
    window, raw_ -> raw, peratom_ -> peratom).
    """
    m = {}
    for ob in REGISTRY.values():
        if sim_type not in ob.sim_types:
            continue
        if ob.tier == tier and ob.rename_to:
            m[ob.rename_to] = ob.name
    if tier == 'raw':
        m.update({'v_fx_spr': 'raw_lfx', 'v_fy_spr': 'raw_lfy',
                  'v_fz_tip': 'raw_nf', 'v_comx_tip': 'raw_tipx',
                  'v_comy_tip': 'raw_tipy', 'v_comz_tip': 'raw_tipz'})
    return m


def layer_channels(n_layers: int) -> list:
    """Build-time injection of com{x,y,z}_L{n} for arbitrary layer counts.

    The static REGISTRY carries only the bilayer entries (golden tests);
    builders call this and extend their channel lists per material.
    """
    out = []
    for lyr in range(1, n_layers + 1):
        for ax in ('x', 'y', 'z'):
            out.append(Observable(
                name=f'com{ax}_L{lyr}', units='angstrom',
                sim_types=('afm', 'sheetonsheet', 'indent'),
                group=f'layer_{lyr}', tier='window', stream='results',
                channel_role='descriptor',
                lammps_compute=f'compute COM_L{lyr} layer_{lyr} com',
                lammps_variable=(f'variable com{ax}_L{lyr} equal '
                                 f'c_COM_L{lyr}[{"xyz".index(ax) + 1}]'),
                rename_to=f'v_com{ax}_L{lyr}'))
    return out


def validate_registry() -> None:
    for name, ob in REGISTRY.items():
        if ob.tier not in TIERS or ob.stream not in STREAMS \
                or ob.channel_role not in ROLES:
            raise ValueError(f"malformed observable {name}")
        if ob.phase not in PHASES:
            raise ValueError(f"{name}: bad phase {ob.phase}")
        if ob.group == 'tip':
            raise ValueError(f"{name}: bare 'tip' group (spec §1.2)")
        if ob.stream == 'results' and ob.tier == 'window' \
                and 'group/group' in (ob.lammps_compute or ''):
            raise ValueError(
                f"{name}: group/group may not run at Nevery=1 (spec §4)")
