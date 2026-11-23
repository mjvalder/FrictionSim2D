"""ilp/tmd chalcogen sub-layer split (interlayer-potential-choice Task 4b).

ilp/tmd builds a TMD atom's normal from neighbours with the same molecule ID
AND the same atom type, so the two chalcogen sub-layers of a layer need their
own types. A potential labelling the chalcogen once (``Mo S``) gets it split
per layer under ``ilp``; one that already splits it (``Mo S1 S2``) does not.

The builds below run the in-process stacking (``run 0`` through the LAMMPS
Python module, not ``lmp_serial``), which under ilp/tmd aborts with "too many
neighbors for TMD atoms" unless the split is in the data file.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from frictionsim2d.builders.construction import components
from frictionsim2d.core.config import GlobalSettings, SheetConfig
from frictionsim2d.potentials.manager import PotentialManager

REPO = Path(__file__).resolve().parents[2]
MAT = REPO / "examples" / "materials"
SW = REPO / "examples" / "potentials" / "sw"
SNAPSHOT = Path(__file__).parent / "data" / "sheet_lj_jiang3_snapshot"


def _sheet(pot: str, layers: int = 2) -> SheetConfig:
    return SheetConfig(mat="h-MoS2", pot_type="sw", pot_path=str(SW / pot),
                       cif_path=str(MAT / "h-MoS2.cif"), x=12.0, y=12.0,
                       layers=[layers], lat_c=6.15)


def _general(style: str):
    return SimpleNamespace(interlayer_style=style, interlayer_file=None,
                           interlayer_comparison=False)


def _pm(sheet: SheetConfig, style: str, n_layers: int = 2):
    pm = PotentialManager(GlobalSettings(), use_langevin=False)
    if style != "lj":
        pm.set_interlayer(style)
    pm.register_component("sheet", sheet, n_layers=n_layers)
    pm.add_self_interaction("sheet")
    pm.add_interlayer(["sheet"])
    return pm


def _read_atoms(path: Path):
    """[(type, z)] from a LAMMPS atomic data file."""
    rows, in_atoms = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if s.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms and s and s[0].isalpha():
            break
        if in_atoms and s:
            p = s.split()
            rows.append((int(p[1]), float(p[4])))
    return rows


def test_ilp_splits_single_chalcogen_in_the_potential_manager():
    pm = _pm(_sheet("mos2_jiang3.sw"), "ilp")
    for layer in range(2):
        ids = [int(t) for t in pm.types.get_layer_group_string("sheet", layer).split()]
        assert len(ids) == 3
    sw = [ln for ln in pm.self_interaction_commands if " sw " in f" {ln} "]
    assert len(sw) == 2
    for line in sw:
        tokens = line.split("#")[0].split()
        assert sorted(t for t in tokens[6:] if t != "NULL") == ["Mo", "S", "S"]
    emap = pm.types.build_element_map(["sheet"])
    for layer in range(2):
        ids = [int(t) for t in pm.types.get_layer_group_string("sheet", layer).split()]
        assert sorted(emap[t - 1] for t in ids) == ["Mo", "S", "S"]


def test_ilp_splits_single_chalcogen(tmp_path):
    """A real 2-layer jiang3 build under ilp: 3 types per layer, top S above Mo."""
    sheet = _sheet("mos2_jiang3.sw")
    settings = GlobalSettings()
    path, _dims, _lat_c = components.build_sheet(
        sheet, tmp_path, stack_if_multi=True, settings=settings,
        n_layers_override=2, interlayer_general=_general("ilp"))
    pm = _pm(sheet, "ilp")
    atoms = _read_atoms(path)
    assert {t for t, _ in atoms} == set(range(1, 7))
    for layer in range(2):
        ids = [int(t) for t in pm.types.get_layer_group_string("sheet", layer).split()]
        by = {t: [z for tt, z in atoms if tt == t] for t in ids}
        mo, s_a, s_b = (by[t] for t in ids)      # element-first: Mo, S(top), S(bottom)
        mo_z = sum(mo) / len(mo)
        assert min(s_a) > mo_z > max(s_b)
    # The split types share one label, so sw sees `Mo S S` and ilp/tmd `Mo S S`.
    assert [t.pot_label for t in pm.types].count("S") == 4


def test_ilp_sevik_not_double_split():
    sevik = _sheet("MoS2_sevik.sw")
    assert len(_pm(sevik, "ilp").types) == len(_pm(sevik, "lj").types)


@pytest.mark.parametrize("style", ["lj_g", "kc"])
def test_other_styles_do_not_split(style):
    jiang = _sheet("mos2_jiang3.sw")
    plain = PotentialManager(GlobalSettings(), use_langevin=False)
    plain.register_component("sheet", jiang, n_layers=2)
    other = PotentialManager(GlobalSettings(), use_langevin=False)
    other.set_interlayer(style)
    other.register_component("sheet", jiang, n_layers=2)
    assert len(other.types) == len(plain.types) == 4


def test_set_interlayer_after_a_splitting_registration_is_refused():
    pm = PotentialManager(GlobalSettings(), use_langevin=False)
    pm.register_component("sheet", _sheet("mos2_jiang3.sw"), n_layers=2)
    with pytest.raises(ValueError, match="before register_component"):
        pm.set_interlayer("ilp")


def test_lj_unchanged_with_jiang(tmp_path, monkeypatch):
    """lj + jiang3: every lammps/ file and the stacked data equal the pre-split snapshot."""
    from tests.builders.test_sheetonsheet_builder import (
        _build_example_sheet, _normalised_lammps_files)
    out = _build_example_sheet(
        tmp_path, monkeypatch,
        sheet={"x": 12, "y": 12, "lat_c": 6.15,
               "pot_path": "examples/potentials/sw/mos2_jiang3.sw"})
    got = _normalised_lammps_files(out)
    got.update({f"build/{p.name}": p.read_text(encoding="utf-8")
                for p in (out / "build").glob("*.lmp")})
    want = {str(p.relative_to(SNAPSHOT)): p.read_text(encoding="utf-8")
            for p in SNAPSHOT.rglob("*") if p.is_file()}
    assert sorted(got) == sorted(want)
    for name, text in want.items():
        assert got[name] == text, name
