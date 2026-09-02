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
    terms: 'nN', 'eV', 'eV/A', 'A', 'K', 'count'. The sheet family shipped a
    lateral force in eV/A against a normal load in nN for the life of the
    project, producing a COF a factor 1.602176565 too small.

    Note what this field does NOT do: it records a DECLARED unit and nothing
    verifies it against what a template actually emits. The coherence test
    over it (``test_afm_cof_inputs_share_force_units``) covers AFM only; the
    sheet family gets its own ``test_sheet_cof_inputs_share_force_units``
    now that it has ``lfx@sheet``/``nf@sheet`` here. Neither could have
    caught the original sheet defect, since it was in the TEMPLATE, not the
    registry -- the test that does is
    ``test_lj_branch_lateral_force_scaled_to_nN``, over the templates.
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

#: Results channels that carry no cross-family role worth naming. Distinct
#: from PARITY_EXEMPT (Task 7): a name here is invisible to the parity tests
#: entirely, while a PARITY_EXEMPT entry concedes ONE family's absence from a
#: role other families do carry.
ROLE_EXEMPT: dict = {
    'tip_min_z':
        'punch-through detection is specific to a rigid indenting body',
}


#: Ordered family pairs the parity contract checks: every results channel
#: with a role_key in `source` must have a counterpart in `target`.
#:
#: afm -> sheetonsheet: these are the two full families, but only this
#: direction is live yet. hetero/flake -> sheetonsheet one way only: they are
#: genuinely reduced variants, and demanding the reverse would need an
#: exemption table large enough to stop being read. 'indent' inherits AFM's
#: declarations and is not separately checked.
#:
#: `('afm', 'sheetonsheet')` is added in Task 18, once the sheet family
#: declares every role AFM does. Turning it on earlier would need ~17
#: exemptions with untrue reasons.
PARITY_PAIRS = (
    ('sheetonsheet', 'afm'),
    ('hetero', 'sheetonsheet'),
    ('flake', 'sheetonsheet'),
)

#: (sim_type, role_key) -> why this family cannot or should not carry it.
#: A reviewed decision with a reason, not a suppression.
PARITY_EXEMPT: dict = {
    ('sheetonsheet', 'tip_sheet_interaction'):
        'no tip body exists in a sheet-on-sheet model',
    ('sheetonsheet', 'sheet_com_x'):
        'AFM tracks one sheet COM; the sheet family resolves COM per layer',
    ('sheetonsheet', 'sheet_com_y'):
        'AFM tracks one sheet COM; the sheet family resolves COM per layer',
    ('sheetonsheet', 'sheet_com_z'):
        'AFM tracks one sheet COM; the sheet family resolves COM per layer',
    ('sheetonsheet', 'group_pe_sum_driven'):
        'the driven body is a layer of the stack, covered by group_pe_sum',
    ('sheetonsheet', 'group_pe_sum_substrate'):
        'the substrate is a layer of the stack, covered by group_pe_sum',
    ('sheetonsheet', 'group_pe_sum_membrane'):
        'the membrane is a layer of the stack, covered by group_pe_sum',
    ('sheetonsheet', 'thermostat_work_substrate'):
        'the sheet stack has one thermostatted group, not tip plus substrate',
    ('afm', 'interlayer_interaction'):
        'AFM measures the tip-sheet interaction, a different pair of bodies',
    ('afm', 'drive_force_x'):
        'the AFM tip is driven by a spring, not a constant applied force',
    ('afm', 'drive_force_y'):
        'the AFM tip is driven by a spring, not a constant applied force',
    ('afm', 'reaction_force_x'):
        'no fixed bottom layer exists in the AFM model to react against',
    ('afm', 'reaction_force_y'):
        'no fixed bottom layer exists in the AFM model to react against',
    ('afm', 'reaction_force_z'):
        'no fixed bottom layer exists in the AFM model to react against',
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


# --- legacy window channels: the first nine columns, in the 2026 campaign's
# order. All nine need rename_to, because real 2026 files carry full headers and
# a missing one leaves e.g. a v_comx column downstream. -----------------------
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
        group='tip_fix' if _name.startswith(('nf', 'lf', 'tip')) else '2D_all',
        tier='window', stream='results',
        rename_to=_native,
        units='nN' if _name in ('nf', 'lfx', 'lfy') else 'A',
        role_key=_LEGACY_ROLE_KEYS[_name]))

# --- layer-resolved COM: bilayer entries for the golden tests. Builders inject
# layer_channels(n_layers) for the material's real layer count. ---------------
for _lyr in (1, 2):
    for _ax in ('x', 'y', 'z'):
        _add(Observable(
            name=f'com{_ax}_L{_lyr}',
            sim_types=('afm', 'sheetonsheet', 'indent', 'hetero', 'flake'),
            group=f'layer_{_lyr}', tier='window', stream='results',
            lammps_compute=f'compute COM_L{_lyr} layer_{_lyr} com',
            lammps_variable=(f'variable com{_ax}_L{_lyr} equal '
                             f'c_COM_L{_lyr}[{"xyz".index(_ax) + 1}]'),
            rename_to=f'v_com{_ax}_L{_lyr}', units='A',
            role_key=f'layer_com_{_ax}_L{_lyr}'))

# --- energies and temperatures ------------------------------------------------
_add(Observable(name='etotal', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                group='all', tier='window', stream='results',
                lammps_variable='variable etot equal etotal',
                rename_to='v_etot', units='eV',
                role_key='total_energy'))
_add(Observable(name='pe', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                group='all', tier='window', stream='results',
                lammps_variable='variable pe_now equal pe',
                rename_to='v_pe_now', units='eV',
                role_key='potential_energy'))
_add(Observable(name='ke', sim_types=('afm', 'sheetonsheet', 'indent', 'hetero'),
                group='all', tier='window', stream='results',
                lammps_variable='variable ke_now equal ke',
                rename_to='v_ke_now', units='eV',
                role_key='kinetic_energy'))
# Cumulative thermostat work; needs `tally yes` on the langevin fixes. Grows
# linearly in a healthy run.
_add(Observable(name='w_thermostat_tip', sim_types=('afm', 'indent'),
                group='tip_thermo', tier='window', stream='results',
                lammps_variable='variable w_lang_tip equal f_lang_tip',
                rename_to='v_w_lang_tip', units='eV',
                role_key='thermostat_work_driven'))
_add(Observable(name='w_thermostat_sub', sim_types=('afm', 'indent'),
                group='sub_thermo', tier='window', stream='results',
                lammps_variable='variable w_lang_sub equal f_lang_sub',
                rename_to='v_w_lang_sub', units='eV',
                role_key='thermostat_work_substrate'))
# The AFM sheet is pure NVE, so its 3D temperature measures frictional heating
# rather than tracking a setpoint.
_add(Observable(name='temp_sheet_3d', sim_types=('afm', 'indent'),
                group='2D_all', tier='window', stream='results',
                lammps_compute='compute T_sheet 2D_all temp',
                lammps_variable='variable temp_sheet equal c_T_sheet',
                rename_to='v_temp_sheet', units='K',
                role_key='group_temperature'))
# Tip-sheet coupling is purely lj/cut, so group/group works here even under the
# hybrid pair style.
_add(Observable(name='E_int_tip_sheet', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='results',
                lammps_compute='compute eint tip_all group/group 2D_all pair yes',
                lammps_variable='variable e_int equal c_eint',
                rename_to='v_e_int', units='eV',
                role_key='tip_sheet_interaction'))
# Template-owned: the compute sits in sheetonsheet's slide template, behind its
# has_internal_lj branch, with real per-material group names.
_add(Observable(name='E_int_interlayer', sim_types=('sheetonsheet', 'hetero'),
                group='layer_n', tier='peratom', stream='results',
                rename_to='v_e_int_il', units='eV',
                role_key='interlayer_interaction'))
# --- sheet window channels: every native below is already emitted by
# sheetonsheet/slide.lmp's fc_win. Declaring them is what lets the registry
# own the renames -- DataReader._SHEET_COLUMN_RENAME is deleted in Task 10.
#
# The lateral channel keeps its group/group compute at Nevery=1: it is the
# friction label itself, its cost is paid in every production sheet run, and
# moving it would change the label's averaging. Template-owned (no
# lammps_compute), so validate_registry's window-tier ban never sees it.
_add(Observable(name='lfx', key='lfx@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='window', stream='results',
                rename_to='v_xfrict', units='nN',
                role_key='lateral_force_x'))
_add(Observable(name='lfy', key='lfy@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='window', stream='results',
                rename_to='v_yfrict', units='nN',
                role_key='lateral_force_y'))
_add(Observable(name='nf', key='nf@sheet', sim_types=('sheetonsheet', 'hetero'),
                group='layer_n', tier='window', stream='results',
                rename_to='v_fz', units='nN', role_key='normal_load'))
# nf and drive_fx/drive_fy are inconsistent siblings: all three come from
# f_force[1..3] on one fix aveforce, but v_fz is named nf because every
# downstream COF consumer expects it. Back-compatibility beats symmetry.
for _ax, _native in (('x', 'v_fx'), ('y', 'v_fy')):
    _add(Observable(name=f'drive_f{_ax}', sim_types=('sheetonsheet', 'hetero'),
                    group='layer_n', tier='window', stream='results',
                    rename_to=_native, units='nN',
                    role_key=f'drive_force_{_ax}'))
# drive{x,y,z} alias the top layer's COM and are therefore redundant with
# com{x,y,z}_L{n}. Kept because they carry the driven_com_* role that pairs
# with AFM's tipx/tipy/tipz, which is what lets COF and the ML extractors be
# written once for both families. A known overlap, not an oversight.
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'drive{_ax}', sim_types=('sheetonsheet', 'hetero'),
                    group='layer_n', tier='window', stream='results',
                    rename_to=f'v_com{_ax}_top', units='A',
                    role_key=f'driven_com_{_ax}'))
# The flake model has one mobile body, so the driven COM is the flake's, not
# a top layer's. Same role_key as the sheet's drive{x,y,z} and AFM's tip*,
# which is what lets a COF consumer read all three families' driven body
# through channel_by_role. A distinct `key` is required: the bare name
# 'drivex' is already taken by the sheet/hetero entry above, and their
# sim_types are disjoint (flake vs sheetonsheet/hetero) so the name may be
# reused, but the registry key may not.
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'drive{_ax}', key=f'drive{_ax}@flake',
                    sim_types=('flake',), group='flake',
                    tier='window', stream='results',
                    rename_to=f'v_com{_ax}_flake', units='A',
                    role_key=f'driven_com_{_ax}'))
# Reaction force on the fixed bottom layer. react_x doubles as the frozen
# legacy correction's LJ-branch discriminator -- see Task 11 and
# read_data._correct_legacy_lf_schema1.
for _ax in ('x', 'y', 'z'):
    _add(Observable(name=f'react_{_ax}',
                    sim_types=('sheetonsheet', 'hetero', 'flake'),
                    group='layer_1', tier='window', stream='results',
                    rename_to=f'v_s{_ax}', units='nN',
                    role_key=f'reaction_force_{_ax}'))

# --- sheet raw tier: registry-derived, so these drive emission as well as
# parsing (the AFM raw varlist is still a literal; Task 13 fixes that).
for _name, _key, _native, _role, _unit in [
    ('raw_lfx', 'raw_lfx@sheet', 'v_xfrict', 'raw_lateral_force_x', 'nN'),
    ('raw_lfy', 'raw_lfy@sheet', 'v_yfrict', 'raw_lateral_force_y', 'nN'),
    ('raw_nf', 'raw_nf@sheet', 'v_fz', 'raw_normal_load', 'nN'),
    ('raw_drivex', None, 'v_comx_top', 'raw_driven_com_x', 'A'),
    ('raw_drivey', None, 'v_comy_top', 'raw_driven_com_y', 'A'),
]:
    _add(Observable(name=_name, key=_key, sim_types=('sheetonsheet', 'hetero'),
                    group='layer_n', tier='raw', stream='results',
                    rename_to=_native, units=_unit, role_key=_role))

_PE_SUM_ROLE_KEYS = {
    'tip_all': 'group_pe_sum_driven',
    'sub_all': 'group_pe_sum_substrate',
    '2D_all': 'group_pe_sum_membrane',
}
for _grp in ('tip_all', 'sub_all', '2D_all'):
    _add(Observable(
        name=f'pe_sum_{_grp}', sim_types=('afm', 'indent'),
        group=_grp, tier='peratom', stream='results',
        lammps_compute=(f'compute peat_{_grp} {_grp} pe/atom\n'
                        f'compute pesum_{_grp} {_grp} reduce sum c_peat_{_grp}'),
        lammps_variable=f'variable pe_{_grp} equal c_pesum_{_grp}',
        rename_to=f'v_pe_{_grp}', units='eV',
        role_key=_PE_SUM_ROLE_KEYS[_grp]))

# --- contact ------------------------------------------------------------------
_add(Observable(name='n_contact', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
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
                group='2D_all', tier='peratom', stream='results',
                lammps_compute=('compute satom 2D_all stress/atom NULL\n'
                                'compute sxx 2D_all reduce sum c_satom[1]'),
                lammps_variable='variable sxx_2d equal c_sxx',
                rename_to='v_sxx_2d', units='eV/A',
                role_key='membrane_stress_xx'))
_add(Observable(name='sigma_yy_2d', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='results',
                lammps_compute='compute syy 2D_all reduce sum c_satom[2]',
                lammps_variable='variable syy_2d equal c_syy',
                rename_to='v_syy_2d', units='eV/A',
                role_key='membrane_stress_yy'))
# post-stream: derived from retained artefacts, never a LAMMPS column
_add(Observable(name='deflect_max', sim_types=('afm', 'indent'),
                group='2D_all', tier='peratom', stream='post', units='A',
                role_key='max_deflection'))
_add(Observable(name='gap_min', sim_types=('afm', 'indent'),
                group='tip_all', tier='peratom', stream='post', units='A',
                role_key='min_gap'))

# --- raw trace (100 fs instantaneous samples) ---------------------------------
_RAW_ROLE_KEYS = {
    'raw_lfx': 'raw_lateral_force_x', 'raw_lfy': 'raw_lateral_force_y',
    'raw_nf': 'raw_normal_load',
    'raw_tipx': 'raw_driven_com_x', 'raw_tipy': 'raw_driven_com_y',
    'raw_tipz': 'raw_driven_com_z',
}
#: Native names these six parse back from. Declared here (rather than left to
#: rename_map's old unconditional block) so build_obs_ctx can derive the raw
#: varlist from the registry instead of carrying its own hardcoded copy.
_RAW_RENAME_TO = {
    'raw_lfx': 'v_fx_spr', 'raw_lfy': 'v_fy_spr', 'raw_nf': 'v_fz_tip',
    'raw_tipx': 'v_comx_tip', 'raw_tipy': 'v_comy_tip', 'raw_tipz': 'v_comz_tip',
}
for _name in ('raw_lfx', 'raw_lfy', 'raw_nf', 'raw_tipx', 'raw_tipy', 'raw_tipz'):
    _add(Observable(
        name=_name,
        sim_types=('afm', 'indent'), group='tip_fix', tier='raw', stream='results',
        rename_to=_RAW_RENAME_TO[_name],
        units='nN' if _name in ('raw_lfx', 'raw_lfy', 'raw_nf') else 'A',
        role_key=_RAW_ROLE_KEYS[_name]))

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


def channel_by_role(sim_type: str, role_key: str):
    """The channel serving `role_key` in `sim_type`, or None.

    Consumers that want to be family-agnostic (COF, the DB writer, the ML
    extractors) can key on roles instead of branching on sim_type.
    """
    for ob in REGISTRY.values():
        if sim_type in ob.sim_types and ob.role_key == role_key:
            return ob
    return None


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
                sim_types=('afm', 'sheetonsheet', 'indent', 'hetero', 'flake'),
                group=f'layer_{lyr}', tier='window', stream='results',
                lammps_compute=f'compute COM_L{lyr} layer_{lyr} com',
                lammps_variable=(f'variable com{ax}_L{lyr} equal '
                                 f'c_COM_L{lyr}[{"xyz".index(ax) + 1}]'),
                rename_to=f'v_com{ax}_L{lyr}', units='A',
                role_key=f'layer_com_{ax}_L{lyr}'))
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
