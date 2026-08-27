"""Tests for SheetOnSheetSimulation builder behavior."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.builders.sheetonsheet import SheetOnSheetSimulation
from src.core.config import SheetOnSheetSimulationConfig, load_settings

# slide.lmp embeds two fresh `{{ range(1, 999999) | random }}` seeds on every
# render (velocity create + langevin fix), so raw output can't be pinned
# literally. Normalize both seeds to a fixed placeholder before comparing.
_VELOCITY_SEED_RE = re.compile(
    r'(velocity        center create [\d.]+ )\d+'
)
_LANGEVIN_SEED_RE = re.compile(
    r'(langevin [\d.]+ [\d.]+ \$\(100\.0\*dt\) )\d+( zero yes)'
)


def _normalize_random_seeds(content: str) -> str:
    content = _VELOCITY_SEED_RE.sub(r'\1SEED', content)
    content = _LANGEVIN_SEED_RE.sub(r'\1SEED\2', content)
    return content


# Captured from SheetOnSheetSimulation.write_inputs() with outer_loop unset,
# on the `_make_sheetonsheet_config` fixture config (pressure=1.0), before
# the scan_angle split was added. Pins that the unset (default) path is
# byte-identical to today.
_EXPECTED_SLIDE_UNSET_OUTER_LOOP = """\
# LAMMPS Input Script for Sheet-vs-Sheet Sliding
# 3-layer model, constraint_mode: none

variable        speed equal 2.0

variable        pressure equal 1.0

variable        a equal 0.0


clear
units           metal
atom_style      atomic
boundary        p p p

neighbor        0.3 bin
neigh_modify every 1 delay 0 check yes

comm_style      tiled

region          box block 0.0 100.0 0.0 100.0 -15.0 30.0 units box
create_box      3 box

read_data       out/build/sheet.lmp add append group bot

include         out/lammps/system.in.settings



dump            sys all atom 10000 out/visuals/slide_p${pressure}_a${a}_s${speed}.lammpstrj

balance         1.0 rcb

min_style       cg
minimize 1e-4 1e-8 1000000 1000000

# Thermostat on middle layers (2 through 2)
group           center union layer_2
velocity        center create 300.0 SEED
fix             lang center langevin 300.0 300.0 $(100.0*dt) SEED zero yes tally yes
fix             nve_center center nve

timestep        0.001
thermo          100000

# Top layer (layer 3): driven
fix             fstage_top layer_3 rigid/nve single force * on on on torque * off off off
# Bottom layer (layer 1): fixed
fix             fsbot layer_1 setforce 0.0 0.0 0.0
velocity        layer_1 set 0.0 0.0 0.0 units box

# COM computes for key layers
compute         COM_top layer_3 com
variable        comx_top equal c_COM_top[1]
variable        comy_top equal c_COM_top[2]
variable        comz_top equal c_COM_top[3]

compute         COM_ctop layer_2 com
variable        comx_ctop equal c_COM_ctop[1]
variable        comy_ctop equal c_COM_ctop[2]
variable        comz_ctop equal c_COM_ctop[3]

compute         COM_cbot layer_2 com
variable        comx_cbot equal c_COM_cbot[1]
variable        comy_cbot equal c_COM_cbot[2]
variable        comz_cbot equal c_COM_cbot[3]

compute         COM_bot layer_1 com
variable        comx_bot equal c_COM_bot[1]
variable        comy_bot equal c_COM_bot[2]
variable        comz_bot equal c_COM_bot[3]

# --- observable registry channels (spec §3): layer-resolved COM for every
# layer, plus global energies. Lateral layer COMs are the realized-registry
# observable (spec §12 — registry is measured, not imposed).
compute         COM_L1 layer_1 com
variable        comx_L1 equal c_COM_L1[1]
variable        comy_L1 equal c_COM_L1[2]
variable        comz_L1 equal c_COM_L1[3]
compute         COM_L2 layer_2 com
variable        comx_L2 equal c_COM_L2[1]
variable        comy_L2 equal c_COM_L2[2]
variable        comz_L2 equal c_COM_L2[3]
compute         COM_L3 layer_3 com
variable        comx_L3 equal c_COM_L3[1]
variable        comy_L3 equal c_COM_L3[2]
variable        comz_L3 equal c_COM_L3[3]
variable        etot equal etotal
variable        pe_now equal pe
variable        ke_now equal ke

# Apply Uniform Pressure
# 1 eV/Å³ = 160.2176565 GPa
variable        A equal (100.0-0.0)*(100.0-0.0)
variable        p equal v_pressure/160.2176565
variable        f equal -v_p*v_A/(count(layer_3))

fix             force layer_3 aveforce 0.0 0.0 $f

variable        fx equal f_force[1]*1.602176565
variable        fy equal f_force[2]*1.602176565
variable        fz equal f_force[3]*1.602176565
variable        sx equal f_fsbot[1]*1.602176565
variable        sy equal f_fsbot[2]*1.602176565
variable        sz equal f_fsbot[3]*1.602176565

compute         frict layer_2 group/group layer_1
# compute group/group's VECTOR is a force in eV/A; the many-body branch above
# reports nN. Scale so both branches -- and v_fz, the load this is divided by --
# are nN. Unscaled here for the life of the project: see
# docs/notes/2026-08-27-sheet-cof-unit-correction.md.
variable        xfrict equal c_frict[1]*1.602176565
variable        yfrict equal c_frict[2]*1.602176565
# The group/group scalar IS the interlayer interaction energy of the shearing
# interface (spec §5 F2, conditional on is_sheet_lj) — free alongside frict.
variable        e_int_il equal c_frict

run             10000

# --- Constant-pressure compression hold (spec §6.1): the sheet's penetration
# analogue, closing the KEY_RESULTS §5 parity gap. d(P) is defined at the
# SHEARING interface (layer_2/layer_1); the
# other gaps are reported alongside. channel_role=diagnostic.
variable        d_shear equal v_comz_L2-v_comz_L1
variable        gap_L1_L2 equal v_comz_L2-v_comz_L1
variable        gap_L2_L3 equal v_comz_L3-v_comz_L2
fix             hold_ave all ave/time 1 1000 1000 v_d_shear v_gap_L1_L2 v_gap_L2_L3 v_fz file out/data/hold_p$(v_pressure)GPa.txt
run             20000
unfix           hold_ave


variable        speed_x equal cos(v_a*PI/180)
variable        speed_y equal sin(v_a*PI/180)

create_atoms    3 single $(xcm(layer_3,x)) $(xcm(layer_3,y)) $(xcm(layer_3,z)+10) units box
group           virtual type 3
velocity        virtual set 0.0 0.0 0.0

fix             spr layer_3 spring couple virtual 3.12075467163009 0.0 0.0 NULL 0
fix             move_virtual virtual move linear $(v_speed*v_speed_x/100) $(v_speed*v_speed_y/100) 0.0

# Three sampling tiers (spec §4). fc_win keeps the legacy 16-column prefix.
# NB v_xfrict's group/group compute at Nevery=1 is the documented spec-§4
# exemption: it is the label itself and its cost was already paid campaign-wide.
fix             fc_win all ave/time 1 1000 1000 v_xfrict v_yfrict v_sx v_sy v_sz v_fx v_fy v_fz v_comx_top v_comy_top v_comx_ctop v_comy_ctop v_comz_ctop v_comx_cbot v_comy_cbot v_comz_cbot v_comx_L1 v_comy_L1 v_comz_L1 v_comx_L2 v_comy_L2 v_comz_L2 v_comx_L3 v_comy_L3 v_comz_L3 v_etot v_pe_now v_ke_now file out/results/friction_p${pressure}_a${a}_s${speed}
fix             fc_raw all ave/time 100 1 100 v_xfrict v_yfrict v_fz v_comx_top v_comy_top file out/results/raw_p${pressure}_a${a}_s${speed}
fix             fc_pa all ave/time 1000 1 1000 v_e_int_il file out/results/peratom_p${pressure}_a${a}_s${speed}

run             500000

# Final-state retention (spec §5 F6)
write_data      out/data/final_p$(v_pressure)GPa_a$(v_a).data"""


def _make_sheetonsheet_config(tmp_path: Path, layers: list[int]) -> SheetOnSheetSimulationConfig:
    """Create a minimal valid sheet-on-sheet config for builder tests."""
    pot_path = tmp_path / "dummy.sw"
    cif_path = tmp_path / "dummy.cif"
    pot_path.write_text("# dummy potential", encoding="utf-8")
    cif_path.write_text("# dummy cif", encoding="utf-8")

    settings = load_settings().model_dump()

    return SheetOnSheetSimulationConfig(
        **{
            "general": {
                "temp": 300.0,
            },
            "2D": {
                "mat": "h-MoS2",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "x": 20.0,
                "y": 20.0,
                "layers": layers,
            },
            "lj_override": {},
            "settings": settings,
        }
    )


def test_n_layers_uses_single_explicit_value(tmp_path: Path) -> None:
    """Builder should use the single configured layer count."""
    config = _make_sheetonsheet_config(tmp_path, layers=[4])

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    assert builder.n_layers == 4


def test_n_layers_raises_for_multiple_layer_values(tmp_path: Path) -> None:
    """Builder should fail fast instead of silently collapsing layer lists."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3, 5, 4])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    with pytest.raises(ValueError, match="requires exactly one value in 2D.layers"):
        _ = builder.n_layers


def test_build_raises_when_less_than_three_layers(tmp_path: Path) -> None:
    """Sheet-on-sheet builder intentionally rejects <3-layer configurations."""
    config = _make_sheetonsheet_config(tmp_path, layers=[2])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    with pytest.raises(ValueError, match="requires at least 3 layers"):
        builder.build()


def test_sheetonsheet_build_orchestrates_single_layer_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Build should execute the expected high-level flow for one layer value."""
    config = _make_sheetonsheet_config(tmp_path, layers=[4])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    called = {
        "init": 0,
        "hpc": 0,
        "write_inputs": 0,
        "generate_potentials": 0,
        "build_sheet_kwargs": None,
    }

    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._create_directories", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._init_provenance",
        lambda *_a, **_k: called.__setitem__("init", called["init"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_hpc_scripts",
        lambda *_a, **_k: called.__setitem__("hpc", called["hpc"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation.write_inputs",
        lambda *_a, **_k: called.__setitem__("write_inputs", called["write_inputs"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_potentials",
        lambda *_a, **_k: called.__setitem__("generate_potentials", called["generate_potentials"] + 1) or object(),
    )

    def mock_build_sheet(*_args, **kwargs):
        called["build_sheet_kwargs"] = kwargs
        return (
            tmp_path / "sheet.lmp",
            {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zlo": 0, "zhi": 20},
            3.0,
        )

    monkeypatch.setattr("src.builders.components.build_sheet", mock_build_sheet)

    builder.build()

    assert called["init"] == 1
    assert called["generate_potentials"] == 1
    assert called["write_inputs"] == 1
    assert called["hpc"] == 1
    assert called["build_sheet_kwargs"]["n_layers_override"] == 4
    assert builder.structure_paths["sheet"].name == "sheet.lmp"
    assert builder.lat_c == 3.0
    assert builder.z_positions["layer_1"] == 0.0
    assert builder.z_positions["layer_4"] == 9.0


def test_sheetonsheet_rejects_internal_lj_with_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Internal-LJ potentials should be rejected for constrained modes."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.sheet.pot_type = "reaxff"
    config.settings.simulation.constraint_mode = "atom_bonds"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._create_directories", lambda *_a, **_k: None)
    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._init_provenance", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="does not support external LJ"):
        builder.build()


def test_sheetonsheet_collect_simulation_paths(tmp_path: Path) -> None:
    """Collect path should report a single simulation when lammps dir exists."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    out_dir = tmp_path / "out"
    builder = SheetOnSheetSimulation(config, output_dir=str(out_dir))

    assert builder._collect_simulation_paths() == []

    (out_dir / "lammps").mkdir(parents=True, exist_ok=True)
    assert builder._collect_simulation_paths() == ["."]


def _seed_builder_state(builder: SheetOnSheetSimulation, tmp_path: Path) -> Path:
    """Seed builder state so write_inputs can render slide.in in isolation.

    Bypasses build()/_generate_potentials() by hand-populating the attributes
    write_inputs() reads: structure_paths, lat_c, sheet_dims, pm, groups.
    """
    output_dir = tmp_path / "out"
    builder.output_dir = output_dir
    builder.relative_run_dir = Path("out")

    sheet_path = tmp_path / "sheet.lmp"
    sheet_path.write_text("# dummy\n", encoding="utf-8")
    builder.structure_paths["sheet"] = sheet_path

    builder.lat_c = 3.0
    builder.sheet_dims = {
        "xlo": 0.0, "xhi": 100.0, "ylo": 0.0, "yhi": 100.0, "zhi": 15.0,
    }
    builder.pm = SimpleNamespace(types=[1, 2, 3], is_sheet_lj=lambda _pt: True)
    builder.groups = {
        "layer_1": "1",
        "layer_2": "2",
        "layer_3": "3",
        "center": "2",
        "all_types": "1 2 3",
    }
    return output_dir


def _prepare_builder_for_slide_write(builder: SheetOnSheetSimulation, tmp_path: Path) -> Path:
    """Seed builder state and render slide.in."""
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    return output_dir / "lammps" / "slide.in"


def test_sheetonsheet_slide_unset_outer_loop_matches_pinned_baseline(tmp_path: Path) -> None:
    """outer_loop unset must emit byte-identical slide.in to today (pinned)."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = 1.0

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert _normalize_random_seeds(slide_script.rstrip()) == _EXPECTED_SLIDE_UNSET_OUTER_LOOP


def test_sheetonsheet_outer_loop_scan_angle_splits_slide_per_angle(tmp_path: Path) -> None:
    """outer_loop=scan_angle emits one slide_a<angle>.in per angle."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = 1.0
    config.general.scan_angle = [0.0, 30.0, 60.0]
    config.general.outer_loop = "scan_angle"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    lammps_dir = output_dir / "lammps"

    # No monolithic slide.in; one per angle instead.
    assert not (lammps_dir / "slide.in").exists()
    slide_scripts = sorted(p.name for p in lammps_dir.glob("slide_*.in"))
    assert slide_scripts == ["slide_a0.in", "slide_a30.in", "slide_a60.in"]

    # Each split slide pins a scalar angle (no in-script angle loop), while
    # the pressure axis keeps its full config (still a single value here).
    slide_30 = (lammps_dir / "slide_a30.in").read_text(encoding="utf-8")
    assert "variable        a equal 30.0" in slide_30
    assert "label           angle_loop" not in slide_30
    assert "variable        a index" not in slide_30

    slide_0 = (lammps_dir / "slide_a0.in").read_text(encoding="utf-8")
    assert "variable        a equal 0.0" in slide_0
    slide_60 = (lammps_dir / "slide_a60.in").read_text(encoding="utf-8")
    assert "variable        a equal 60.0" in slide_60


def test_sheetonsheet_outer_loop_scan_angle_single_angle_stays_slide_in(tmp_path: Path) -> None:
    """A single angle keeps slide.in so the combined HPC path applies."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = 1.0
    config.general.scan_angle = [15.0]
    config.general.outer_loop = "scan_angle"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    lammps_dir = output_dir / "lammps"

    assert (lammps_dir / "slide.in").exists()
    assert not list(lammps_dir.glob("slide_a*.in"))


def test_sheetonsheet_outer_loop_scan_angle_rejects_scan_angle_force(tmp_path: Path) -> None:
    """The split can't honor a pressure-selective angle filter; reject it clearly.

    scan_angle_force restricts which pressure value(s) the scan_angle list
    applies to (sheetonsheet/slide.lmp's angle_loop gate, comparing against
    v_pressure at line ~202). Splitting into one script per angle applies
    each angle to every pressure value in that script, silently dropping the
    filter -- so reject the combination instead of changing what the scan
    means.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = [1.0, 2.0, 3.0]
    config.general.scan_angle = [0.0, 30.0, 60.0]
    config.general.scan_angle_force = 2.0
    config.general.outer_loop = "scan_angle"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    _seed_builder_state(builder, tmp_path)

    with pytest.raises(ValueError, match="scan_angle_force"):
        builder.write_inputs()


def test_hpc_job_name_goes_through_the_material_id_hook(tmp_path: Path) -> None:
    """The job name must read the hook, not `config.sheet.mat` a second time.

    `_material_id` exists so the hetero builders can key the name on the full
    `mat1+mat2` pair. A `_get_hpc_job_name` that re-reads `config.sheet.mat`
    happens to agree here, which is exactly why the divergence would go
    unnoticed until a subclass overrode one and not the other.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    assert builder._get_hpc_job_name() == f"sheet_{builder._material_id()}"

    class _Paired(SheetOnSheetSimulation):
        def _material_id(self) -> str:
            return "mat1+mat2"

    paired = _Paired(config, output_dir=str(tmp_path / "out"))
    assert paired._get_hpc_job_name() == "sheet_mat1+mat2"
