"""ILP/KC box-size guard (src/builders/utils/interlayer.py).

The run scripts use ``comm_style tiled``, which reaches one periodic image
only, so under ``ilp``/``kc`` every perpendicular in-plane width of the cell
must exceed the interlayer cutoff (16 A) plus the neighbor skin.
"""
import math
from types import SimpleNamespace

import pytest

from src.builders.utils.interlayer import (check_interlayer_box,
                                           check_interlayer_box_from_data,
                                           periodic_widths, read_data_cell)
from src.potentials.interlayer import ILP_CUTOFF, KC_FULL_CUTOFF

SKIN = 0.3


def _general(style):
    return SimpleNamespace(interlayer_style=style)


def test_orthogonal_widths_are_lx_and_ly():
    assert periodic_widths(20.0, 30.0) == (20.0, 30.0)


def test_tilt_narrows_the_x_width():
    # Hexagonal 60-degree cell: a=(L,0), b=(L/2, L*sqrt(3)/2) -> both widths L*sqrt(3)/2.
    length = 20.0
    wx, wy = periodic_widths(length, length * math.sqrt(3) / 2, length / 2)
    assert wx == pytest.approx(length * math.sqrt(3) / 2)
    assert wy == pytest.approx(length * math.sqrt(3) / 2)


@pytest.mark.parametrize("style,cutoff", [("ilp", ILP_CUTOFF), ("kc", KC_FULL_CUTOFF)])
def test_too_small_raises_with_box_and_fix(style, cutoff):
    with pytest.raises(ValueError) as exc:
        check_interlayer_box(_general(style), lx=12.0, ly=30.0, skin=SKIN)
    msg = str(exc.value)
    assert "12.00 x 30.00" in msg
    assert f"{cutoff + SKIN:g}" in msg
    assert f"x > {cutoff + SKIN:.1f}" in msg and f"y > {cutoff + SKIN:.1f}" in msg


def test_exactly_cutoff_plus_skin_is_refused():
    need = ILP_CUTOFF + SKIN
    with pytest.raises(ValueError):
        check_interlayer_box(_general("ilp"), lx=need, ly=40.0, skin=SKIN)
    check_interlayer_box(_general("ilp"), lx=need + 0.01, ly=need + 0.01, skin=SKIN)


def test_tilted_cell_refused_although_lx_passes():
    # lx = 18 > 16.3, but the tilt makes the x width 18*17/hypot(9,17) = 15.9.
    with pytest.raises(ValueError) as exc:
        check_interlayer_box(_general("ilp"), lx=18.0, ly=17.0, xy=9.0, skin=SKIN)
    msg = str(exc.value)
    assert "xy tilt 9.00" in msg
    # The suggested x clears the tilted width.
    x_needed = float(msg.split("x > ")[1].split(" A")[0])
    assert periodic_widths(x_needed + 0.1, 17.0, 9.0)[0] > ILP_CUTOFF + SKIN


@pytest.mark.parametrize("style", ["lj", "lj_g", None])
def test_other_styles_pass(style):
    general = None if style is None else _general(style)
    check_interlayer_box(general, lx=5.0, ly=5.0, skin=SKIN)


def test_skin_counts():
    check_interlayer_box(_general("ilp"), lx=16.5, ly=16.5, skin=0.3)
    with pytest.raises(ValueError):
        check_interlayer_box(_general("ilp"), lx=16.5, ly=16.5, skin=1.0)


def test_read_data_cell(tmp_path):
    data = tmp_path / "cell.lmp"
    data.write_text("# header\n\n3 atoms\n1 atom types\n\n"
                    "0.0 19.1 xlo xhi\n-1.0 21.0 ylo yhi\n0 30 zlo zhi\n"
                    "4.5 0.0 0.0 xy xz yz\n\nMasses\n\n1 95.95\n\n"
                    "Atoms # atomic\n\n1 1 0 0 0\n", encoding="utf-8")
    assert read_data_cell(data) == pytest.approx((19.1, 22.0, 4.5))
    with pytest.raises(ValueError, match="19.10 x 22.00"):
        check_interlayer_box_from_data(_general("kc"), data, skin=3.0)
    check_interlayer_box_from_data(_general("ilp"), data, skin=SKIN)


def test_lj_does_not_read_the_file(tmp_path):
    check_interlayer_box_from_data(_general("lj"), tmp_path / "missing.lmp", skin=SKIN)


def test_sheetonsheet_build_refuses_a_small_ilp_cell(tmp_path, monkeypatch):
    """Builder level: a 12 x 12 ilp sheet-on-sheet build stops before writing scripts."""
    from tests.builders.test_sheetonsheet_builder import _build_example_sheet
    with pytest.raises(ValueError, match="interlayer_style = ilp: the periodic cell"):
        _build_example_sheet(tmp_path, monkeypatch,
                             general={"interlayer_style": "ilp"},
                             sheet={"x": 12, "y": 12, "lat_c": 6.15})
    assert not (tmp_path / "out" / "lammps" / "slide.in").exists()
