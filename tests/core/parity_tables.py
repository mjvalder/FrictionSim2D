"""Role-parity contract tables for tests/core/test_observables.py."""

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
    ('afm', 'interlayer_interaction_all_pairs'):
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
