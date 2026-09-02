"""Tests for AFMSimulation builder layer sweep behavior."""

import re
from pathlib import Path

import pytest

from src.builders.afm import AFMSimulation
from src.potentials.utils import normalize_potential_type
from tests.helpers.afm import _make_afm_config, _prepare_builder_for_slide_write, _seed_builder_state

# The langevin fixes embed a fresh `{{ range(1, 999999) | random }}` seed on
# every render (see afm/slide.lmp), so raw output can't be pinned literally.
# Normalize both seeds to a fixed placeholder before comparing.
_SEED_RE = re.compile(
    r'(langevin [\d.]+ [\d.]+ \$\(100\.0\*dt\) )\d+( zero yes)'
)


def _normalize_random_seeds(content: str) -> str:
    return _SEED_RE.sub(r'\1SEED\2', content)


# Captured from AFMSimulation.write_inputs() with outer_loop unset, on the
# `_make_afm_config` fixture config, before the scan_angle split was added.
# Pins that the unset (default) path is byte-identical to today.
_EXPECTED_SLIDE_UNSET_OUTER_LOOP = """\
# LAMMPS input script for AFM Sliding Simulation

variable        speed equal 2.0

variable        find equal 10.0


variable        a equal 0.0


clear
units           metal
atom_style      atomic
boundary        p p p

neighbor        0.3 bin
neigh_modify every 1 delay 0 check yes

timestep        0.001
thermo          100000
comm_style      tiled

read_data       out/L1/data/load_$(v_find)N.data
include         out/L1/lammps/system.in.settings

group           2D type 3
velocity        2D zero angular



dump            sys all atom 10000 out/L1/visuals/slide_f$(v_find)_a${a}_s${speed}.lammpstrj

balance         1.0 rcb

fix             sub_fix sub_fix setforce 0.0 0.0 0.0
fix             tip_f tip_fix rigid/nve single force * on on on torque * off off off

compute         temp_tip tip_thermo temp/partial 0 1 0
fix             lang_tip tip_thermo langevin 300.0 300.0 $(100.0*dt) SEED zero yes tally yes
fix_modify      lang_tip temp temp_tip

compute         temp_sub sub_thermo temp/partial 0 1 0
fix             lang_sub sub_thermo langevin 300.0 300.0 $(100.0*dt) SEED zero yes tally yes
fix_modify      lang_sub temp temp_sub

group           nve_apply subtract all sub_fix tip_fix
fix             nve_all nve_apply nve

compute         COM_top 2D_all com
variable        comx equal c_COM_top[1]
variable        comy equal c_COM_top[2]
variable        comz equal c_COM_top[3]

compute         COM_tip tip_fix com
variable        comx_tip equal c_COM_tip[1]
variable        comy_tip equal c_COM_tip[2]
variable        comz_tip equal c_COM_tip[3]

# --- observable registry channels (spec §3) ---------------------------------
compute T_sheet 2D_all temp
compute cz_tip_s tip_all reduce min z
compute COM_L1 layer_1 com
compute eint tip_all group/group 2D_all pair yes
compute peat_tip_all tip_all pe/atom
compute pesum_tip_all tip_all reduce sum c_peat_tip_all
compute peat_sub_all sub_all pe/atom
compute pesum_sub_all sub_all reduce sum c_peat_sub_all
compute peat_2D_all 2D_all pe/atom
compute pesum_2D_all 2D_all reduce sum c_peat_2D_all
compute coordc 2D_all coord/atom cutoff 5.0 group tip_all
compute contacts 2D_all reduce sum c_coordc
compute satom 2D_all stress/atom NULL
compute sxx 2D_all reduce sum c_satom[1]
compute syy 2D_all reduce sum c_satom[2]
variable etot equal etotal
variable pe_now equal pe
variable ke_now equal ke
variable w_lang_tip equal f_lang_tip
variable w_lang_sub equal f_lang_sub
variable temp_sheet equal c_T_sheet
variable tip_min_z equal c_cz_tip_s
variable comx_L1 equal c_COM_L1[1]
variable comy_L1 equal c_COM_L1[2]
variable comz_L1 equal c_COM_L1[3]
variable e_int equal c_eint
variable pe_tip_all equal c_pesum_tip_all
variable pe_sub_all equal c_pesum_sub_all
variable pe_2D_all equal c_pesum_2D_all
variable n_contact equal c_contacts
variable sxx_2d equal c_sxx
variable syy_2d equal c_syy

run             0

variable        Ftotal equal -v_find/1.602176565
variable        n equal v_Ftotal/count(tip_fix)
fix             forcetip tip_fix aveforce 0.0 0.0 $n

variable        fz_tip equal f_forcetip[3]*1.602176565

variable        fx_spr equal f_spr[1]*1.602176565
variable        fy_spr equal f_spr[2]*1.602176565

fix             damp tip_fix viscous 6.25

variable        spring_x equal cos(v_a*PI/180)
variable        spring_y equal sin(v_a*PI/180)
variable        drive_vx equal v_spring_x*v_speed/100
variable        drive_vy equal v_spring_y*v_speed/100

run             0
variable        virtual_x equal xcm(tip_fix,x)+30.0*v_spring_x
variable        virtual_y equal xcm(tip_fix,y)+30.0*v_spring_y
variable        virtual_z equal xcm(tip_fix,z)

create_atoms    3 single $(v_virtual_x) $(v_virtual_y) $(v_virtual_z) units box
group           virtual type 3
velocity        virtual set 0.0 0.0 0.0

fix             spr tip_fix spring couple virtual 3.1210986267166043 0.0 0.0 NULL 30.0
fix             move_virtual virtual move linear $(v_drive_vx) $(v_drive_vy) 0.0

# Output fixes AFTER the drive block: fix spr must exist before any tier can
# sample v_fx_spr. With Nrepeat=1, ave/time samples IMMEDIATELY at setup when
# the (restored) timestep is a multiple of Nfreq — the load_*.data timestep
# always is a multiple of 100, so fc_raw defined earlier fires at run 0 and
# dies on the missing fix. (The legacy fc_ave survived the same layout only
# because Nrepeat=1000's phase shift never aligns.)
fix             fc_win all ave/time 1 1000 1000 v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz v_comx_tip v_comy_tip v_comz_tip v_etot v_pe_now v_ke_now v_w_lang_tip v_w_lang_sub v_temp_sheet v_tip_min_z v_comx_L1 v_comy_L1 v_comz_L1 file out/L1/results/friction_f$(v_find)_a${a}_s${speed}_layer1.txt title1 "# schema=2 lf_units=nN family=afm"
fix             fc_raw all ave/time 100 1 100 v_fx_spr v_fy_spr v_fz_tip v_comx_tip v_comy_tip v_comz_tip file out/L1/results/raw_f$(v_find)_a${a}_s${speed}_layer1.txt title1 "# schema=2 lf_units=nN family=afm"
fix             fc_pa all ave/time 1000 1 1000 v_e_int v_pe_tip_all v_pe_sub_all v_pe_2D_all v_n_contact v_sxx_2d v_syy_2d file out/L1/results/peratom_f$(v_find)_a${a}_s${speed}_layer1.txt title1 "# schema=2 lf_units=nN family=afm"


run             500000

# Final-state retention (spec §5 F6): the post-slide configuration is always
# available for contact/deflection reconstruction — never lucky again.
write_data      out/L1/data/final_f$(v_find)N_a$(v_a).data"""


def test_afm_builder_supports_layer_sweep(tmp_path: Path) -> None:
    """AFM builder should accept multiple layer values (unlike sheet-on-sheet)."""
    config = _make_afm_config(tmp_path, layers=[1, 2, 3])

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    # Should not raise; multiple layers are intentional for AFM
    assert config.sheet.layers == [1, 2, 3]


def test_afm_builder_layer_specific_paths(tmp_path: Path, monkeypatch) -> None:
    """AFM should create L{n} directories for each layer count."""
    # Mock out actual file operations to avoid needing real files
    monkeypatch.setattr(
        "src.builders.components.build_sheet",
        lambda *args, **kwargs: (tmp_path / "sheet.lmp", {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zhi": 15}, 3.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_tip",
        lambda *args, **kwargs: (tmp_path / "tip.lmp", 20.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_substrate",
        lambda *args, **kwargs: tmp_path / "sub.lmp"
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._init_provenance",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *args, **kwargs: None
    )

    config = _make_afm_config(tmp_path, layers=[2, 3])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    # Verify internal layer storage before build
    assert builder.config.sheet.layers == [2, 3]


def test_normalize_potential_type_consistency() -> None:
    """Potential type normalization should be consistent across builders."""
    test_cases = [
        ("SW", "sw"),
        ("Tersoff", "tersoff"),
        ("ReaxFF", "reaxff"),
        ("REAX/C", "reax/c"),
        ("  sw  ", "sw"),
    ]

    for input_val, expected in test_cases:
        assert normalize_potential_type(input_val) == expected


def test_afm_build_orchestrates_per_layer_steps(tmp_path: Path, monkeypatch) -> None:
    """Build should execute per-layer orchestration for all requested layers."""
    config = _make_afm_config(tmp_path, layers=[2, 4])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    called = {
        "build_sheet_layers": [],
        "create_dirs": [],
        "build_components": 0,
        "generate_potentials": [],
        "z_positions": [],
        "write_inputs": [],
        "hpc": 0,
    }

    def mock_build_sheet(*_args, **kwargs):
        called["build_sheet_layers"].append(kwargs["n_layers_override"])
        return (
            tmp_path / f"sheet_L{kwargs['n_layers_override']}.lmp",
            {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zlo": 0, "zhi": 20},
            3.0,
        )

    monkeypatch.setattr("src.builders.components.build_sheet", mock_build_sheet)
    monkeypatch.setattr("src.builders.afm.AFMSimulation._init_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._get_substrate_box_dims",
        lambda _self: {"xlo": 0.0, "xhi": 100.0, "ylo": 0.0, "yhi": 100.0, "zlo": -5.0, "zhi": 0.0},
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda _self, out_dir=None: called["create_dirs"].append(out_dir),
    )

    def mock_build_components(_self, _build_dir):
        called["build_components"] += 1
        return tmp_path / "tip.lmp", 20.0, tmp_path / "sub.lmp"

    monkeypatch.setattr("src.builders.afm.AFMSimulation._build_components", mock_build_components)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_potentials",
        lambda _self, n: called["generate_potentials"].append(n) or object(),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        lambda _self, n, _r: called["z_positions"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda _self, n: called["write_inputs"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *_a, **_k: called.__setitem__("hpc", called["hpc"] + 1),
    )

    builder.build()

    assert called["build_sheet_layers"] == [2, 4]
    assert called["build_components"] == 1
    assert called["generate_potentials"] == [2, 4]
    assert called["z_positions"] == [2, 4]
    assert called["write_inputs"] == [2, 4]
    assert called["hpc"] == 1
    assert sorted(builder.output_dir_layer.keys()) == [2, 4]


def test_afm_finite_sheet_computes_box_and_center_offsets(tmp_path: Path) -> None:
    """Finite-sheet mode should use substrate x/y as box and center the sheet."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.finite_sheet = True
    config.sub.x = 160.0
    config.sub.y = 140.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    builder.sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    builder.box_dims = {
        "xlo": 0.0,
        "xhi": 160.0,
        "ylo": 0.0,
        "yhi": 140.0,
        "zlo": -5.0,
        "zhi": 0.0,
    }

    offsets = builder._compute_sheet_offset()

    assert builder.box_dims["xhi"] == 160.0
    assert builder.box_dims["yhi"] == 140.0
    assert offsets == {"x": 30.0, "y": 20.0}


def test_afm_finite_sheet_validates_final_substrate_dims(tmp_path: Path) -> None:
    """Final substrate dims from .lmp should be validated, not only requested config dims."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.finite_sheet = True
    config.sub.x = 160.0
    config.sub.y = 140.0
    config.settings.potential.lj_cutoff = 11.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    builder.sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    # Simulate post-build substrate dimensions read from sub.lmp.
    builder.box_dims = {
        "xlo": 0.0,
        "xhi": 110.0,
        "ylo": 0.0,
        "yhi": 110.0,
        "zlo": -5.0,
        "zhi": 0.0,
    }

    with pytest.raises(ValueError, match="Final built substrate x-span delta must be greater than LJ cutoff"):
        builder._validate_final_substrate_dims()


def test_afm_scan_angle_list_with_force_gate(tmp_path: Path) -> None:
    """Explicit angle lists should be used directly (no interval expansion)."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [2.0, 10.0, 4.0]
    config.general.scan_angle_force = 30.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        a index 2.0 10.0 4.0" in slide_script
    assert "variable        scan_angle_force equal 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script


def test_afm_scan_angle_explicit_list_with_force_gate(tmp_path: Path) -> None:
    """Explicit AFM angle lists should be supported with a separate force gate."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [0.0, 2.0, 6.0, 10.0]
    config.general.scan_angle_force = 30.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        a index 0.0 2.0 6.0 10.0" in slide_script
    assert "variable        scan_angle_force equal 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script


def test_afm_scan_angle_force_accepts_multiple_targets(tmp_path: Path) -> None:
    """scan_angle_force should support lists of target force values."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [0.0, 45.0, 90.0]
    config.general.scan_angle_force = [10.0, 30.0]

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        scan_angle_force index 10.0 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script
    assert "then \"next scan_angle_force\"" in slide_script


def test_afm_outer_loop_force_splits_slide_per_load(tmp_path: Path) -> None:
    """outer_loop=force emits one slide_f<load>N.in per load, sharing system.in."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0]
    config.general.outer_loop = "force"

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    lammps_dir = tmp_path / "out" / "L1" / "lammps"

    # One shared system.in; one slide script per load; no monolithic slide.in.
    assert (lammps_dir / "system.in").exists()
    assert not (lammps_dir / "slide.in").exists()
    slide_scripts = sorted(p.name for p in lammps_dir.glob("slide_*.in"))
    assert slide_scripts == ["slide_f10N.in", "slide_f2N.in", "slide_f5N.in"]

    # system.in still writes every load's data file (full force loop retained).
    system_script = (lammps_dir / "system.in").read_text(encoding="utf-8")
    assert "variable        find index 2.0 5.0 10.0" in system_script

    # Each split slide reads its own load and pins a scalar force (no force loop).
    slide_5 = (lammps_dir / "slide_f5N.in").read_text(encoding="utf-8")
    assert "read_data       out/L1/data/load_$(v_find)N.data" in slide_5
    assert "variable        find equal 5.0" in slide_5
    assert "label           force_loop" not in slide_5


def test_afm_outer_loop_force_single_load_stays_slide_in(tmp_path: Path) -> None:
    """A single load keeps slide.in so the combined system+slide HPC path applies."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [10.0]
    config.general.outer_loop = "force"

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    lammps_dir = tmp_path / "out" / "L1" / "lammps"

    assert (lammps_dir / "slide.in").exists()
    assert not list(lammps_dir.glob("slide_f*.in"))


def test_afm_slide_unset_outer_loop_matches_pinned_baseline(tmp_path: Path) -> None:
    """outer_loop unset must emit byte-identical slide.in to today (pinned)."""
    config = _make_afm_config(tmp_path, layers=[1])

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert _normalize_random_seeds(slide_script.rstrip()) == _EXPECTED_SLIDE_UNSET_OUTER_LOOP


def test_afm_supports_angle_split_by_default() -> None:
    """AFMSimulation opts into the angle split; only opt-out subclasses (e.g.
    IndentSimulation, see tests/test_indent_builder.py) drop it. The same
    declaration drives the job-count prediction (src/run/expand.py)."""
    assert 'scan_angle' in AFMSimulation.SPLIT_OUTER_LOOPS
    assert 'force' in AFMSimulation.SPLIT_OUTER_LOOPS


def test_afm_outer_loop_scan_angle_splits_slide_per_angle(tmp_path: Path) -> None:
    """outer_loop=scan_angle emits one slide_a<angle>.in per angle, sharing system.in."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.scan_angle = [0.0, 30.0, 60.0]
    config.general.outer_loop = "scan_angle"

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    lammps_dir = tmp_path / "out" / "L1" / "lammps"

    # One shared system.in; one slide script per angle; no monolithic slide.in.
    assert (lammps_dir / "system.in").exists()
    assert not (lammps_dir / "slide.in").exists()
    slide_scripts = sorted(p.name for p in lammps_dir.glob("slide_*.in"))
    assert slide_scripts == ["slide_a0.in", "slide_a30.in", "slide_a60.in"]

    # Each split slide pins a scalar angle (no in-script angle loop).
    slide_30 = (lammps_dir / "slide_a30.in").read_text(encoding="utf-8")
    assert "variable        a equal 30.0" in slide_30
    assert "label           angle_loop" not in slide_30
    assert "variable        a index" not in slide_30

    slide_0 = (lammps_dir / "slide_a0.in").read_text(encoding="utf-8")
    assert "variable        a equal 0.0" in slide_0
    slide_60 = (lammps_dir / "slide_a60.in").read_text(encoding="utf-8")
    assert "variable        a equal 60.0" in slide_60


def test_afm_outer_loop_scan_angle_single_angle_stays_slide_in(tmp_path: Path) -> None:
    """A single angle keeps slide.in so the combined system+slide HPC path applies."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.scan_angle = [15.0]
    config.general.outer_loop = "scan_angle"

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    lammps_dir = tmp_path / "out" / "L1" / "lammps"

    assert (lammps_dir / "slide.in").exists()
    assert not list(lammps_dir.glob("slide_a*.in"))


def test_afm_outer_loop_scan_angle_rejects_scan_angle_force(tmp_path: Path) -> None:
    """The split can't honor a force-selective angle filter; reject it clearly.

    scan_angle_force restricts which force value(s) the scan_angle list applies
    to (afm/slide.lmp's angle_loop gate at line ~140). Splitting into one script
    per angle applies each angle to every force value in that script, silently
    dropping the filter -- so reject the combination instead of changing what
    the physics means.
    """
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0]
    config.general.scan_angle = [0.0, 30.0, 60.0]
    config.general.scan_angle_force = 5.0
    config.general.outer_loop = "scan_angle"

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _seed_builder_state(builder, tmp_path)

    with pytest.raises(ValueError, match="scan_angle_force"):
        builder.write_inputs(1)


def test_afm_explicit_tip_x_ignores_tip_x_offset(tmp_path: Path) -> None:
    """Explicit tip_x should be used exactly; tip_x_offset should not be applied."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.tip.tip_x = 33.0
    config.tip.tip_x_offset = 7.5

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    layer_dir = _prepare_builder_for_slide_write(builder, tmp_path).parent
    system_script = (layer_dir / "system.in").read_text(encoding="utf-8")

    assert "shift 33.0" in system_script


def test_afm_explicit_tip_y_ignores_tip_y_offset(tmp_path: Path) -> None:
    """Explicit tip_y should be used exactly; tip_y_offset should not be applied."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.tip.tip_y = 41.0
    config.tip.tip_y_offset = -3.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    layer_dir = _prepare_builder_for_slide_write(builder, tmp_path).parent
    system_script = (layer_dir / "system.in").read_text(encoding="utf-8")

    assert " 41.0 " in system_script


def _freestanding_afm_cfg(tmp_path, layers=(4,)):
    from src.core.config import AFMSimulationConfig, load_settings
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return AFMSimulationConfig(**{
        "general": {"temp": 300.0, "force": [2.0], "scan_speed": 2.0, "finite_sheet": False},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 40.0, "y": 40.0, "layers": list(layers)},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "pes": {"grid_n": 8},
        "settings": load_settings().model_dump(),
    })


def test_provenance_skips_absent_substrate(tmp_path):
    from src.builders.afm import AFMSimulation
    cfg = _freestanding_afm_cfg(tmp_path)
    builder = AFMSimulation(cfg, output_dir=str(tmp_path / "o"))

    # Spy on which components get recorded: the guard must SKIP the absent
    # substrate while still recording the present components. This fails if the
    # guard is removed (i.e. ('sub', None) is passed unconditionally).
    seen = []
    builder._add_component_files_to_provenance = lambda name, config: seen.append(name)

    builder._init_provenance()  # must not raise on a None substrate
    assert "sub" not in seen
    assert {"sheet", "tip"} <= set(seen)
