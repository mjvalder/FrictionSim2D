"""The registry of simulation output channels, declared once each.

Every channel has two names that must agree: the LAMMPS variable a template
writes into its ``fix ave/time``, and the name the parser renames that column
back to. Declaring both here keeps the builders (which read the varlists) and
``postprocessing/read_data.py`` (which reads the rename maps) from drifting.

Sampling tiers, with their ``fix ave/time`` args in ``TIER_ARGS``: ``window``
samples every step and block-means over 1000 (the 2026 campaign layout),
``raw`` takes one instantaneous sample per 100 steps, ``peratom`` runs the
costly per-atom computes once per 1000. ``group/group`` rebuilds a neighbour
list per call, so ``validate_registry`` bans it from the window tier.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

TIERS = ('window', 'raw', 'peratom')
STREAMS = ('results', 'post')

# fix ave/time argument triples (Nevery, Nrepeat, Nfreq), validated against LAMMPS
TIER_ARGS = {
    'window': (1, 1000, 1000),
    'raw': (100, 1, 100),
    'peratom': (1000, 1, 1000),
}


@dataclass(frozen=True)
class Observable:
    """One output channel.

    ``rename_to`` is the native LAMMPS column name (``v_fz_tip``) the parser
    maps back to ``name``. No ``lammps_compute`` means the template owns the
    channel and emits it itself, often conditionally; the registry only has to
    parse it back.

    ``units`` is the physical unit of the emitted column, in LAMMPS ``metal``
    terms: 'nN', 'eV', 'eV/A', 'A', 'K', 'count'. It is not decorative -- the
    sheet family shipped a lateral force in eV/A against a normal load in nN
    for the life of the project, producing a COF a factor 1.602176565 too
    small, and a coherence test over this field is what catches that class of
    defect.
    """
    name: str
    sim_types: Tuple[str, ...]
    group: str
    tier: str
    stream: str
    lammps_compute: Optional[str] = None
    lammps_variable: Optional[str] = None
    rename_to: Optional[str] = None
    units: Optional[str] = None


REGISTRY: dict = {}


def _add(ob: Observable) -> None:
    if ob.name in REGISTRY:
        raise ValueError(f"duplicate observable {ob.name}")
    REGISTRY[ob.name] = ob


# --- legacy window channels: the first nine columns, in the 2026 campaign's
# order. All nine need rename_to, because real 2026 files carry full headers and
# a missing one leaves e.g. a v_comx column downstream. -----------------------
for _name, _native in [
    ('nf', 'v_fz_tip'), ('lfx', 'v_fx_spr'), ('lfy', 'v_fy_spr'),
    ('comx', 'v_comx'), ('comy', 'v_comy'), ('comz', 'v_comz'),
    ('tipx', 'v_comx_tip'), ('tipy', 'v_comy_tip'), ('tipz', 'v_comz_tip'),
]:
    _add(Observable(
        name=_name,
        sim_types=('afm', 'indent'),
        group='tip_fix' if _name.startswith(('nf', 'lf', 'tip')) else '2D_all',
        tier='window', stream='results',
        rename_to=_native,
        units='nN' if _name in ('nf', 'lfx', 'lfy') else 'A'))

# --- layer-resolved COM: bilayer entries for the golden tests. Builders inject
# layer_channels(n_layers) for the material's real layer count. ---------------
for _lyr in (1, 2):
    for _ax in ('x', 'y', 'z'):
        _add(Observable(
            name=f'com{_ax}_L{_lyr}',
            sim_types=('afm', 'sheetonsheet', 'indent'),
            group=f'layer_{_lyr}', tier='window', stream='results',
            lammps_compute=f'compute COM_L{_lyr} layer_{_lyr} com',
            lammps_variable=(f'variable com{_ax}_L{_lyr} equal '
                             f'c_COM_L{_lyr}[{"xyz".index(_ax) + 1}]'),
            rename_to=f'v_com{_ax}_L{_lyr}', units='A'))

# --- energies and temperatures ------------------------------------------------
_add(Observable(name='etotal', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results',
                lammps_variable='variable etot equal etotal',
                rename_to='v_etot', units='eV'))
_add(Observable(name='pe', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results',
                lammps_variable='variable pe_now equal pe',
                rename_to='v_pe_now', units='eV'))
_add(Observable(name='ke', sim_types=('afm', 'sheetonsheet', 'indent'),
                group='all', tier='window', stream='results',
                lammps_variable='variable ke_now equal ke',
                rename_to='v_ke_now', units='eV'))
# Cumulative thermostat work; needs `tally yes` on the langevin fixes. Grows
# linearly in a healthy run.
_add(Observable(name='w_thermostat_tip', sim_types=('afm', 'indent'),
                group='tip_thermo', tier='window', stream='results',
                lammps_variable='variable w_lang_tip equal f_lang_tip',
                rename_to='v_w_lang_tip', units='eV'))
_add(Observable(name='w_thermostat_sub', sim_types=('afm', 'indent'),
                group='sub_thermo', tier='window', stream='results',
                lammps_variable='variable w_lang_sub equal f_lang_sub',
                rename_to='v_w_lang_sub', units='eV'))
# The AFM sheet is pure NVE, so its 3D temperature measures frictional heating
# rather than tracking a setpoint.
_add(Observable(name='temp_sheet_3d', sim_types=('afm', 'indent'),
                group='2D_all', tier='window', stream='results',
                lammps_compute='compute T_sheet 2D_all temp',
                lammps_variable='variable temp_sheet equal c_T_sheet',
                rename_to='v_temp_sheet', units='K'))
# Tip-sheet coupling is purely lj/cut, so group/group works here even under the
# hybrid pair style.
_add(Observable(name='E_int_tip_sheet', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='results',
                lammps_compute='compute eint tip_all group/group 2D_all pair yes',
                lammps_variable='variable e_int equal c_eint',
                rename_to='v_e_int', units='eV'))
# Template-owned: the compute sits in sheetonsheet's slide template, behind its
# has_internal_lj branch, with real per-material group names.
_add(Observable(name='E_int_interlayer', sim_types=('sheetonsheet',),
                group='layer_n', tier='peratom', stream='results',
                rename_to='v_e_int_il', units='eV'))
# Sheet friction label: a group/group read by the window fix at Nevery=1, a
# deliberate exemption from the tier rule (cost already paid campaign-wide).
# Template-owned, so validate_registry never sees it.
#
# rename_to is deliberately None. read_data merges this map OVER
# _SHEET_COLUMN_RENAME, so declaring 'v_xfrict' here shadowed that map's
# 'v_xfrict' -> 'lfx' and the frame lost lfx entirely -- silently killing COF
# for every sheet run (read_data.py's COF guard needs lfx, lfy AND nf).
# The real fix is to name this channel 'lfx' for the sheet family, which needs
# the registry to allow one name across disjoint sim_types; until then the
# parser's own sheet map owns the rename.
_add(Observable(name='xfrict', sim_types=('sheetonsheet',),
                group='layer_n', tier='window', stream='results',
                rename_to=None, units='nN'))
for _grp in ('tip_all', 'sub_all', '2D_all'):
    _add(Observable(
        name=f'pe_sum_{_grp}', sim_types=('afm', 'indent'),
        group=_grp, tier='peratom', stream='results',
        lammps_compute=(f'compute peat_{_grp} {_grp} pe/atom\n'
                        f'compute pesum_{_grp} {_grp} reduce sum c_peat_{_grp}'),
        lammps_variable=f'variable pe_{_grp} equal c_pesum_{_grp}',
        rename_to=f'v_pe_{_grp}', units='eV'))

# --- contact ------------------------------------------------------------------
_add(Observable(name='n_contact', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                lammps_compute=('compute coordc 2D_all coord/atom cutoff '
                                '{contact_cutoff} group tip_all\n'
                                'compute contacts 2D_all reduce sum c_coordc'),
                lammps_variable='variable n_contact equal c_contacts',
                rename_to='v_n_contact', units='count'))
# In-plane virial sums; post normalises them to a membrane stress resultant
# (N/m). No sigma_zz: for a slab with traction-free faces, its through-thickness
# integral is not a stress resultant.
_add(Observable(name='sigma_xx_2d', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                lammps_compute=('compute satom 2D_all stress/atom NULL\n'
                                'compute sxx 2D_all reduce sum c_satom[1]'),
                lammps_variable='variable sxx_2d equal c_sxx',
                rename_to='v_sxx_2d', units='eV/A'))
_add(Observable(name='sigma_yy_2d', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                lammps_compute='compute syy 2D_all reduce sum c_satom[2]',
                lammps_variable='variable syy_2d equal c_syy',
                rename_to='v_syy_2d', units='eV/A'))
# post-stream: derived from retained artefacts, never a LAMMPS column
_add(Observable(name='deflect_max', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='post', units='A'))
_add(Observable(name='gap_min', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='post', units='A'))

# --- raw trace (100 fs instantaneous samples) ---------------------------------
for _name in ('raw_lfx', 'raw_lfy', 'raw_nf', 'raw_tipx', 'raw_tipy', 'raw_tipz'):
    _add(Observable(
        name=_name,
        sim_types=('afm',), group='tip_fix', tier='raw', stream='results',
        units='nN' if _name in ('raw_lfx', 'raw_lfy', 'raw_nf') else 'A'))

# --- punch-through ingredient: min tip z, in the SLIDING script --------------
_add(Observable(name='tip_min_z', sim_types=('afm',),
                group='tip_all', tier='window', stream='results',
                lammps_compute='compute cz_tip_s tip_all reduce min z',
                lammps_variable='variable tip_min_z equal c_cz_tip_s',
                rename_to='v_tip_min_z', units='A'))


def channels_for(sim_type: str, tier: str) -> list:
    """Results-stream channels for one (sim_type, tier)."""
    return [ob for ob in REGISTRY.values()
            if sim_type in ob.sim_types and ob.tier == tier
            and ob.stream == 'results']


def expected_columns(sim_type: str, tier: str) -> list:
    """The names alone, from :func:`channels_for`."""
    return [ob.name for ob in channels_for(sim_type, tier)]


def rename_map(sim_type: str, tier: str) -> dict:
    """Native column name -> channel name, for one tier.

    The raw file's header repeats the window file's native names (v_fx_spr...),
    so those map to raw_lfx... in the raw map only. The parser picks the tier
    from the filename prefix: friction_/fc_ave_, raw_, peratom_.
    """
    m = {}
    for ob in REGISTRY.values():
        if sim_type not in ob.sim_types:
            continue
        if ob.tier == tier and ob.rename_to:
            m[ob.rename_to] = ob.name
    if tier == 'raw' and sim_type in ('afm', 'indent'):
        m.update({'v_fx_spr': 'raw_lfx', 'v_fy_spr': 'raw_lfy',
                  'v_fz_tip': 'raw_nf', 'v_comx_tip': 'raw_tipx',
                  'v_comy_tip': 'raw_tipy', 'v_comz_tip': 'raw_tipz'})
    return m


def layer_channels(n_layers: int) -> list:
    """com{x,y,z}_L{n} for any layer count, injected at build time.

    REGISTRY holds only the bilayer entries, for the golden tests; builders call
    this and extend their channel lists per material.
    """
    out = []
    for lyr in range(1, n_layers + 1):
        for ax in ('x', 'y', 'z'):
            out.append(Observable(
                name=f'com{ax}_L{lyr}',
                sim_types=('afm', 'sheetonsheet', 'indent'),
                group=f'layer_{lyr}', tier='window', stream='results',
                lammps_compute=f'compute COM_L{lyr} layer_{lyr} com',
                lammps_variable=(f'variable com{ax}_L{lyr} equal '
                                 f'c_COM_L{lyr}[{"xyz".index(ax) + 1}]'),
                rename_to=f'v_com{ax}_L{lyr}', units='A'))
    return out


def validate_registry() -> None:
    """Raise on a malformed entry, a bare 'tip' group, or window-tier group/group."""
    for name, ob in REGISTRY.items():
        if ob.tier not in TIERS or ob.stream not in STREAMS:
            raise ValueError(f"malformed observable {name}")
        if ob.group == 'tip':
            raise ValueError(f"{name}: bare 'tip' group")
        if ob.stream == 'results' and ob.tier == 'window' \
                and 'group/group' in (ob.lammps_compute or ''):
            raise ValueError(
                f"{name}: group/group may not run at Nevery=1")
