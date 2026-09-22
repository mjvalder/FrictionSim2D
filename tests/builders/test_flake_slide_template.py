"""What the flake slide template may and may not emit as a normal force.

A flake slide now APPLIES a normal load -- `fix force flake addforce` presses
the flake onto the sheet with the configured pressure times the flake's own
projected area -- so `v_fz` is the same physical quantity as the sheet's and
`nf@sheet` claims the `flake` sim_type. This file pins that, and pins the two
wrong sources the channel must never come from.

**The invention these tests exist to prevent.** Before the load block landed a
flake run had no normal force at all, and the obvious-looking substitute was
`variable fz equal abs(v_sz)`, where `v_sz = f_fsbot[3]*ev_a_to_nn` is the
frozen bottom band's pre-reset net z-force. With no applied load its mean is
zero, so `abs()` rectifies the thermal fluctuation to `sigma*sqrt(2/pi)` -- a
strictly positive number set by TEMPERATURE and bottom-layer atom count, not by
friction. `mean_cof` would then be `mean(|F_lat|)/mean(|noise|)`: finite,
plausible, scaling as roughly 1/sqrt(T), and landing in the same column as
genuine AFM and sheet COFs. Two aggravating details, so the shape of the
mistake is recognisable elsewhere: `abs()` inside LAMMPS destroys the sign at
emission, unrecoverably without re-running; and on the `v_sx` branch
`xfrict IS v_sx`, so the "COF" degenerates to the direction cosine of one noise
vector.
"""
from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[2]
            / 'src' / 'templates' / 'flake' / 'slide.lmp')


def _source() -> str:
    return TEMPLATE.read_text(encoding='utf-8')


def test_the_template_emits_v_fz_from_the_applied_load():
    """`fz` is `f_force[3]`, the group force on the `addforce` fix, scaled to
    nN like the lateral channels it is the denominator of."""
    source = _source()

    assert 'fix             force flake addforce' in source
    assert 'variable        fz equal f_force[3]*{{ ev_a_to_nn }}' in source


def test_the_normal_force_is_never_the_frozen_bottom_bands_reaction():
    """`f_fsbot[3]` is the setforce-frozen bottom band's reaction. Nothing
    presses on it through the stack, so its mean is zero and any COF built on
    it is rectified thermal noise."""
    for line in _source().splitlines():
        if line.lstrip().startswith('variable') and ' fz ' in f' {line} ':
            assert 'fsbot' not in line, line


def test_no_emitted_normal_force_is_rectified_with_abs():
    """The sign of a normal force is physical (attractive vs repulsive) and
    `abs()` inside LAMMPS destroys it at emission, unrecoverably."""
    for line in _source().splitlines():
        stripped = line.lstrip()
        if stripped.startswith('variable') and ('fz' in stripped
                                                or 'frict' in stripped):
            assert 'abs(' not in stripped, line


def test_the_load_is_the_flake_area_not_the_box_area():
    """A flake covers a fraction of the box; pressure times the BOX area would
    over-load it by that fraction and make the COF area-dependent."""
    source = _source()

    assert 'variable        A equal {{ flake_area }}' in source
    assert 'variable        f equal -v_p*v_A/(count(flake))' in source
