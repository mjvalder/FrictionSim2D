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

#: Every schema a results file can follow. The ONE vocabulary: node types,
#: `read_data._KNOWN_FAMILIES` and `hpc/manifest.py` all import this rather
#: than repeating a subset of it. Distinct from `run.naming.MODELS`, which is
#: the CLI / top-directory axis: a run has both a `model` (where it lives) and
#: a `family` (what schema its files follow).
FAMILIES: Tuple[str, ...] = (
    'afm', 'indent', 'sheetonsheet', 'hetero', 'flake', 'pes_tip', 'pes_sheet',
)

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
#: sheetonsheet <-> afm: the two full families, checked in both directions.
#: Every sheet channel with an AFM counterpart now exists, so both readings
#: of the contract hold honestly, each against its own reasoned
#: PARITY_EXEMPT entries. hetero/flake -> sheetonsheet one way only: they are
#: genuinely reduced variants, and demanding the reverse would need an
#: exemption table large enough to stop being read. 'indent' inherits AFM's
#: declarations and is not separately checked.
PARITY_PAIRS = (
    ('sheetonsheet', 'afm'),
    ('afm', 'sheetonsheet'),
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
        'the driven body is a layer of the stack, covered by group_pe_sum_L{i}',
    ('sheetonsheet', 'group_pe_sum_substrate'):
        'the substrate is a layer of the stack, covered by group_pe_sum_L{i}',
    ('sheetonsheet', 'group_pe_sum_membrane'):
        'the membrane is a layer of the stack, covered by group_pe_sum_L{i}',
    ('sheetonsheet', 'thermostat_work_substrate'):
        'the sheet stack has one thermostatted group, not tip plus substrate',
    ('afm', 'group_temperature_center'):
        'AFM has one 2D sheet with one temperature; the sheet stack resolves '
        'temperature per body (interior, fixed bottom, driven top)',
    ('afm', 'group_temperature_layer_1'):
        'AFM has one 2D sheet with one temperature; the sheet stack resolves '
        'temperature per body (interior, fixed bottom, driven top)',
    ('afm', 'group_temperature_layer_n'):
        'AFM has one 2D sheet with one temperature; the sheet stack resolves '
        'temperature per body (interior, fixed bottom, driven top)',
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
    ('afm', 'group_pe_sum_L1'):
        'AFM resolves group PE by body (tip/substrate/membrane), not by '
        'sheet layer; the sheet stack has no equivalent whole-body groups',
    ('afm', 'group_pe_sum_L2'):
        'AFM resolves group PE by body (tip/substrate/membrane), not by '
        'sheet layer; the sheet stack has no equivalent whole-body groups',
    ('afm', 'membrane_stress_xx_L1'):
        'AFM already covers whole-sheet membrane stress (sigma_xx_2d); the '
        'sheet stack resolves it per layer instead',
    ('afm', 'membrane_stress_xx_L2'):
        'AFM already covers whole-sheet membrane stress (sigma_xx_2d); the '
        'sheet stack resolves it per layer instead',
    ('afm', 'membrane_stress_yy_L1'):
        'AFM already covers whole-sheet membrane stress (sigma_yy_2d); the '
        'sheet stack resolves it per layer instead',
    ('afm', 'membrane_stress_yy_L2'):
        'AFM already covers whole-sheet membrane stress (sigma_yy_2d); the '
        'sheet stack resolves it per layer instead',
    ('sheetonsheet', 'group_temperature'):
        'AFM has a single 2D sheet with one temperature to report; the '
        'sheet stack resolves temperature per body (center, layer_1, the '
        'top layer) and carries no aggregate equivalent',
    ('sheetonsheet', 'thermostat_work_tip'):
        'the sheet has no tip body; its thermostatted group is the stack '
        'interior, not the driven contact',
    ('afm', 'thermostat_work_interior'):
        'the AFM tip is both thermostatted and driven, so its dissipation '
        'is reported at the contact, not at a separate interior body',
    ('sheetonsheet', 'membrane_stress_xx'):
        'AFM sums the in-plane virial over one membrane; the sheet stack '
        'resolves it per layer (membrane_stress_xx_L{i})',
    ('sheetonsheet', 'membrane_stress_yy'):
        'AFM sums the in-plane virial over one membrane; the sheet stack '
        'resolves it per layer (membrane_stress_yy_L{i})',
    ('sheetonsheet', 'raw_driven_com_z'):
        'the sheet family could carry this role -- v_comz_top already '
        'exists and raw_drivex/raw_drivey already prove the mechanism -- '
        'but it is deliberately NOT declared: doing so would widen every '
        'sheet raw results file by a column, and this conversion is '
        'specified to add none. A scope decision, not an incapacity.',
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
                role_key='thermostat_work_tip'))
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
# Cumulative thermostat work for the sheet stack. `fix lang center langevin
# ... tally yes` is already emitted by all three sheet templates, so LAMMPS is
# already accumulating this scalar and throwing it away. Filtered out on
# non-Langevin runs by build_obs_ctx's w_thermostat prefix rule -- f_lang
# does not exist under nose-hoover. role_key is its OWN role
# (thermostat_work_interior), not AFM's w_thermostat_tip's
# (thermostat_work_tip): the sheet's thermostatted group is the stack
# INTERIOR (`center`), while the driven top layer runs undamped `rigid/nve`
# and is never thermostatted -- unlike AFM, where the tip IS both
# thermostatted and driven. Sharing one role would let a role-keyed consumer
# (COF, the DB writer) read both as "dissipation at the driven contact",
# which is true for AFM and false for the sheet. Each role carries a
# PARITY_EXEMPT entry against the other family instead.
_add(Observable(name='w_thermostat_center',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='center', tier='window', stream='results',
                lammps_variable='variable w_lang_center equal f_lang',
                rename_to='v_w_lang_center', units='eV',
                role_key='thermostat_work_interior'))
# Group temperatures for the sheet stack: `center` (thermostatted interior),
# `layer_1` (the fixed bottom band, `setforce 0 0 0`) and the driven top
# layer (`rigid/nve`). `center` and `layer_1` are literal group names in all
# three templates. The top layer is NOT: `layer_n` here is a registry LABEL
# meaning "the top layer", and the real LAMMPS group is `layer_{n_layers}`
# (`layer_4`, `layer_6`, ...) -- so temp_Ltop_3d's compute carries a
# `{n_layers}` placeholder that build_obs_ctx substitutes at render time, the
# same mechanism n_contact uses for `{contact_cutoff}`. `group='layer_n'`
# stays a label (validate_registry only rejects the bare string 'tip').
#
# Diagnostics with empirically banded ranges, not setpoint checks -- see the
# task brief. Not filtered by thermostat type: `compute ... temp` works under
# either langevin or nose-hoover.
_add(Observable(name='temp_center_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='center', tier='window', stream='results',
                lammps_compute='compute T_center center temp',
                lammps_variable='variable temp_center_3d equal c_T_center',
                rename_to='v_temp_center_3d', units='K',
                role_key='group_temperature_center'))
_add(Observable(name='temp_L1_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_1', tier='window', stream='results',
                lammps_compute='compute T_layer_1 layer_1 temp',
                lammps_variable='variable temp_L1_3d equal c_T_layer_1',
                rename_to='v_temp_L1_3d', units='K',
                role_key='group_temperature_layer_1'))
_add(Observable(name='temp_Ltop_3d', sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='window', stream='results',
                lammps_compute='compute T_Ltop layer_{n_layers} temp',
                lammps_variable='variable temp_Ltop_3d equal c_T_Ltop',
                rename_to='v_temp_Ltop_3d', units='K',
                role_key='group_temperature_layer_n'))
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
# own the renames -- DataReader._SHEET_COLUMN_RENAME was deleted once it did.
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
# 'flake' too: flake/slide.lmp loads the flake with `fix force flake addforce`
# (pressure x the flake's projected area) and emits v_fz = f_force[3] in nN,
# so a flake run's COF is lateral force / normal load like the sheet's.
# `group` is a label (the sheet's driven top layer); for flake the loaded
# body is the flake group.
_add(Observable(name='nf', key='nf@sheet',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='window', stream='results',
                rename_to='v_fz', units='nN', role_key='normal_load'))
# nf and drive_fx/drive_fy are inconsistent siblings: all three come from
# f_force[1..3] on one fix aveforce, but v_fz is named nf because every
# downstream COF consumer expects it. Back-compatibility beats symmetry.
# The flake template defines only v_fz, so drive_fx/fy stay sheet/hetero.
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
# parsing (the AFM raw varlist is likewise derived from channels_for('afm',
# 'raw'), not a hardcoded literal).
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


def _pe_sum_layer(lyr: int) -> Observable:
    """Per-layer group PE sum, opt-in. Shared by the static REGISTRY entries
    (lyr in (1, 2), for the golden tests) and layer_channels() (the real
    layer count) so the two can never drift apart."""
    return Observable(
        name=f'pe_sum_L{lyr}', sim_types=('sheetonsheet', 'hetero', 'flake'),
        group=f'layer_{lyr}', tier='peratom', stream='results', opt_in=True,
        lammps_compute=(f'compute peat_L{lyr} layer_{lyr} pe/atom\n'
                        f'compute pesum_L{lyr} layer_{lyr} reduce sum '
                        f'c_peat_L{lyr}'),
        lammps_variable=f'variable pe_sum_L{lyr} equal c_pesum_L{lyr}',
        rename_to=f'v_pe_sum_L{lyr}', units='eV',
        role_key=f'group_pe_sum_L{lyr}')


def _sigma_xx_layer(lyr: int) -> Observable:
    """Per-layer in-plane stress sum (xx), opt-in. See _pe_sum_layer."""
    return Observable(
        name=f'sigma_xx_2d_L{lyr}',
        sim_types=('sheetonsheet', 'hetero', 'flake'),
        group=f'layer_{lyr}', tier='peratom', stream='results', opt_in=True,
        lammps_compute=(f'compute satom_L{lyr} layer_{lyr} stress/atom NULL\n'
                        f'compute sxx_L{lyr} layer_{lyr} reduce sum '
                        f'c_satom_L{lyr}[1]'),
        lammps_variable=f'variable sigma_xx_2d_L{lyr} equal c_sxx_L{lyr}',
        rename_to=f'v_sigma_xx_2d_L{lyr}', units='eV/A',
        role_key=f'membrane_stress_xx_L{lyr}')


def _sigma_yy_layer(lyr: int) -> Observable:
    """Per-layer in-plane stress sum (yy), opt-in. See _pe_sum_layer."""
    return Observable(
        name=f'sigma_yy_2d_L{lyr}',
        sim_types=('sheetonsheet', 'hetero', 'flake'),
        group=f'layer_{lyr}', tier='peratom', stream='results', opt_in=True,
        lammps_compute=(f'compute satom_L{lyr} layer_{lyr} stress/atom NULL\n'
                        f'compute syy_L{lyr} layer_{lyr} reduce sum '
                        f'c_satom_L{lyr}[2]'),
        lammps_variable=f'variable sigma_yy_2d_L{lyr} equal c_syy_L{lyr}',
        rename_to=f'v_sigma_yy_2d_L{lyr}', units='eV/A',
        role_key=f'membrane_stress_yy_L{lyr}')


# --- opt-in sheet peratom/post channels (Task 18) ------------------------------
# Every channel below ships DISABLED: opt_in=True, gated by
# settings.output.optional_channels (a default build emits none of them --
# see test_default_build_emits_no_opt_in_channel). They cost measurable
# per-atom compute time and stay off until a pilot measures the overhead.
#
# No sigma_zz_L{i}: same reasoning as sigma_xx_2d/sigma_yy_2d above -- for a
# slab with traction-free faces, the through-thickness integral of sigma_zz
# is not a stress resultant.
#
# pe_sum_L{i}/sigma_xx_2d_L{i}/sigma_yy_2d_L{i} follow the existing
# per-layer convention (com{x,y,z}_L{1,2} below / layer_channels()): static
# L1/L2 entries here for the golden/registry tests, plus a generator
# (_pe_sum_layer/_sigma_xx_layer/_sigma_yy_layer) reused by layer_channels()
# for the real layer count. build_obs_ctx swaps the static pair for the
# generated ones the same way it already does for com{x,y,z}_L{1,2}.
_add(_pe_sum_layer(1))
_add(_pe_sum_layer(2))
# sigma_xx_2d_L{i} and sigma_yy_2d_L{i} each declare their OWN full copy of
# the shared `stress/atom` compute (unlike AFM's sigma_xx_2d/sigma_yy_2d
# pair, where sigma_yy_2d omits it and relies on sigma_xx_2d having already
# defined c_satom): either may be requested alone via optional_channels, so
# neither can depend on the other being present. build_obs_ctx dedupes
# identical LAMMPS statement lines, so requesting both still emits
# `compute satom_L{i} ...` exactly once.
_add(_sigma_xx_layer(1))
_add(_sigma_xx_layer(2))
_add(_sigma_yy_layer(1))
_add(_sigma_yy_layer(2))

# n_contact_il: contact count across the real sliding interface (il_top vs
# il_bot). 4f0ed92 moved the sliding interface to the middle of the stack,
# so layer_{n-1}/layer_{n-2} no longer names it in general -- the compute
# below carries {il_top}/{il_bot} placeholders build_obs_ctx substitutes
# from its interface_layers argument, the same mechanism {contact_cutoff}
# and {n_layers} already use. Pairs with AFM's n_contact under role
# 'contact_count', so no PARITY_EXEMPT entry is needed. Where is_sheet_lj()
# is false there is no interlayer lj/cut sigma to derive a cutoff from, so
# this channel is unavailable for exactly the materials that also lose
# E_int_interlayer -- the builder's existing extra_filter already drops
# both.
_add(Observable(name='n_contact_il', sim_types=('sheetonsheet', 'hetero'),
                group='layer_n', tier='peratom', stream='results', opt_in=True,
                lammps_compute=(
                    'compute coordil layer_{il_top} coord/atom cutoff '
                    '{contact_cutoff} group layer_{il_bot}\n'
                    'compute contactsil layer_{il_top} reduce sum c_coordil'),
                lammps_variable='variable n_contact_il equal c_contactsil',
                rename_to='v_n_contact_il', units='count',
                role_key='contact_count'))

# post-stream: derived in postprocessing from retained artefacts, never a
# LAMMPS column (test_post_stream_channels_have_no_compute enforces no
# lammps_compute/rename_to on these). Both pair cleanly with an existing AFM
# role, so neither needs a PARITY_EXEMPT entry.
_add(Observable(name='gap_min_il', sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='peratom', stream='post', opt_in=True,
                units='A', role_key='min_gap'))
_add(Observable(name='corrugation_max',
                sim_types=('sheetonsheet', 'hetero', 'flake'),
                group='layer_n', tier='peratom', stream='post', opt_in=True,
                units='A', role_key='max_deflection'))

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


def rename_map(sim_type: str, tier: str, n_layers: int = 2) -> dict:
    """Native column name -> channel name, for one tier and layer count.

    The raw file's header repeats the window file's native names (v_fx_spr...),
    so those map to raw_lfx... in the raw map only. The parser picks the tier
    from the filename prefix: friction_/fc_ave_, raw_, peratom_.

    REGISTRY carries only the L1/L2 per-layer entries (they exist for the
    golden tests); :func:`layer_channels` is the generator builders already
    use to EMIT the real layer count, and it is consulted here so the parser
    can name back exactly what was emitted. Without it a 5-layer run's L3..L5
    columns reached the export under their native ``v_comz_L3`` names,
    invisible to every consumer that keys on channel names.
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
    """com{x,y,z}_L{n} (window) plus the opt-in per-layer peratom channels
    (pe_sum_L{n}, sigma_xx_2d_L{n}, sigma_yy_2d_L{n}), for any layer count,
    injected at build time.

    REGISTRY holds only the bilayer entries, for the golden tests; builders call
    this and extend their channel lists per material. build_obs_ctx routes the
    window-tier entries into its window list and the peratom-tier ones into
    its peratom list, swapping out the corresponding static L1/L2 REGISTRY
    entries either way -- see its Ruling 1 comment.
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
        out.append(_pe_sum_layer(lyr))
        out.append(_sigma_xx_layer(lyr))
        out.append(_sigma_yy_layer(lyr))
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
