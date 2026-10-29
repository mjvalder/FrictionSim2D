"""Tests for SheetOnSheetSimulation builder behavior."""

import re
from pathlib import Path
from typing import Dict
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
# byte-identical to today. Updated for Task 14: the three fix ave/time lines
# now come from the observables registry (src/core/observables.py) via
# emit_output_fixes(obs_ctx) -- column order changed, and COM_ctop/COM_cbot/
# COM_bot were deleted as exact aliases of the layer-resolved COMs. Updated
# again for Task 17: three new computes (T_center, T_layer_1, T_Ltop) and
# four new window variables (w_lang_center, temp_center_3d, temp_L1_3d,
# temp_Ltop_3d) -- at n_layers=3, T_Ltop names layer_3, the real top layer.
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

# --- runaway-temperature guard ----------------------------------------------
variable        halt_T equal temp
fix             halt_hot all halt 1000 v_halt_T > 3000.0 error continue message yes

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

# --- observable registry channels (spec §3) ---------------------------------
compute T_center center temp
compute T_layer_1 layer_1 temp
compute T_Ltop layer_3 temp
compute COM_L1 layer_1 com
compute COM_L2 layer_2 com
compute COM_L3 layer_3 com
variable etot equal etotal
variable pe_now equal pe
variable ke_now equal ke
variable w_lang_center equal f_lang
variable temp_center_3d equal c_T_center
variable temp_L1_3d equal c_T_layer_1
variable temp_Ltop_3d equal c_T_Ltop
variable comx_L1 equal c_COM_L1[1]
variable comy_L1 equal c_COM_L1[2]
variable comz_L1 equal c_COM_L1[3]
variable comx_L2 equal c_COM_L2[1]
variable comy_L2 equal c_COM_L2[2]
variable comz_L2 equal c_COM_L2[3]
variable comx_L3 equal c_COM_L3[1]
variable comy_L3 equal c_COM_L3[2]
variable comz_L3 equal c_COM_L3[3]


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
if              "$(v_halt_T) > 3000.0" then &
                "print '# HALT stage=slide pressure=$(v_pressure) angle=$(v_a) speed=$(v_speed) T=$(v_halt_T) step=$(step)' append out/results/halted.txt screen yes" &
                "jump SELF slide_next"


# --- Constant-pressure compression hold (spec §6.1): the sheet's penetration
# analogue, closing the KEY_RESULTS §5 parity gap. d(P) is defined at the
# SHEARING interface (layer_2/layer_1); the
# other gaps are reported alongside. channel_role=diagnostic.
variable        d_shear equal v_comz_L2-v_comz_L1
variable        gap_L1_L2 equal v_comz_L2-v_comz_L1
variable        gap_L2_L3 equal v_comz_L3-v_comz_L2
fix             hold_ave all ave/time 1 1000 1000 v_d_shear v_gap_L1_L2 v_gap_L2_L3 v_fz file out/data/hold_p$(v_pressure)GPa.txt
run             20000
if              "$(v_halt_T) > 3000.0" then &
                "print '# HALT stage=slide pressure=$(v_pressure) angle=$(v_a) speed=$(v_speed) T=$(v_halt_T) step=$(step)' append out/results/halted.txt screen yes" &
                "jump SELF slide_next"

unfix           hold_ave


variable        speed_x equal cos(v_a*PI/180)
variable        speed_y equal sin(v_a*PI/180)

create_atoms    3 single $(xcm(layer_3,x)) $(xcm(layer_3,y)) $(xcm(layer_3,z)+10) units box
group           virtual type 3
velocity        virtual set 0.0 0.0 0.0

fix             spr layer_3 spring couple virtual 3.12075467163009 0.0 0.0 NULL 0
fix             move_virtual virtual move linear $(v_speed*v_speed_x/100) $(v_speed*v_speed_y/100) 0.0

# Three sampling tiers (spec §4). Column order is registry-driven (Task 14).
# NB v_xfrict's group/group compute at Nevery=1 is the documented spec-§4
# exemption: it is the label itself and its cost was already paid campaign-wide.
fix             fc_win all ave/time 1 1000 1000 v_etot v_pe_now v_ke_now v_w_lang_center v_temp_center_3d v_temp_L1_3d v_temp_Ltop_3d v_xfrict v_yfrict v_fz v_fx v_fy v_comx_top v_comy_top v_comz_top v_sx v_sy v_sz v_comx_L1 v_comy_L1 v_comz_L1 v_comx_L2 v_comy_L2 v_comz_L2 v_comx_L3 v_comy_L3 v_comz_L3 file out/results/friction_p${pressure}_a${a}_s${speed} title1 "# schema=2 lf_units=nN family=sheetonsheet"
fix             fc_raw all ave/time 100 1 100 v_xfrict v_yfrict v_fz v_comx_top v_comy_top file out/results/raw_p${pressure}_a${a}_s${speed} title1 "# schema=2 lf_units=nN family=sheetonsheet"
fix             fc_pa all ave/time 1000 1 1000 v_e_int_il file out/results/peratom_p${pressure}_a${a}_s${speed} title1 "# schema=2 lf_units=nN family=sheetonsheet"


run             500000
if              "$(v_halt_T) > 3000.0" then &
                "print '# HALT stage=slide pressure=$(v_pressure) angle=$(v_a) speed=$(v_speed) T=$(v_halt_T) step=$(step)' append out/results/halted.txt screen yes" &
                "jump SELF slide_next"


# Final-state retention (spec §5 F6)
write_data      out/data/final_p$(v_pressure)GPa_a$(v_a).data

# A halted point (common/halt.lmp) jumps here, past the rest of its slide.
label           slide_next"""


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


def test_sheetonsheet_build_produces_the_whole_run_tree(tmp_path: Path) -> None:
    """A REAL end-to-end build, asserting on the tree it produces.

    This replaces a version that monkeypatched `_create_directories`,
    `_init_provenance`, `_generate_potentials`, `write_inputs`,
    `_generate_hpc_scripts` and `components.build_sheet` and then asserted each
    had been called once -- the pattern `builders-construction.md` and
    `builders-afm.md` both flagged, and the source of all six of this file's
    entries in `tests/repo/test_mock_targets.py::KNOWN_VIOLATIONS`. Call counts
    cannot tell a working build from one that writes nothing.

    Nothing is mocked. The homogeneous sheet build needs no LAMMPS (the native
    CIF -> orthogonalize -> duplicate -> stack path) and completes in ~2.5 s on
    the real h-MoS2 fixture, so the artifacts themselves are the assertion.
    """
    settings = load_settings()
    settings.hpc.modules = ["lammps"]        # gates HPC generation, which build() runs
    config = SheetOnSheetSimulationConfig(**{
        "general": {"temp": 300.0, "scan_speed": 1, "pressure": 1.0},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/sw/MoS2_wen.sw", "pot_type": "sw",
               "x": 12.0, "y": 12.0, "layers": [4]},
    }, settings=settings)
    out = tmp_path / "out"

    builder = SheetOnSheetSimulation(config, output_dir=str(out))
    builder.build()

    # The stacked structure, named for the configured layer count -- the
    # artifact-side equivalent of the old `build_sheet_kwargs['n_layers_override']`
    # assertion, and it also proves the file was actually written.
    assert builder.structure_paths["sheet"] == out / "build" / "h-MoS2_4.lmp"
    assert (out / "build" / "h-MoS2_4.lmp").is_file()

    # Layers are stacked at integer multiples of the measured interlayer
    # spacing, bottom-first. lat_c is read off the built structure, so pin the
    # relationship rather than a hardcoded spacing.
    assert builder.lat_c is not None and builder.lat_c > 0.0
    assert sorted(builder.z_positions) == [f"layer_{i}" for i in range(1, 5)]
    for i in range(4):
        assert builder.z_positions[f"layer_{i + 1}"] == pytest.approx(i * builder.lat_c)

    # The potential spine reached disk, with one sw block per layer.
    settings_text = (out / "lammps" / "system.in.settings").read_text(encoding="utf-8")
    assert settings_text.count("# sheet Layer") == 4
    assert (out / "provenance" / "potentials" / "MoS2_wen.sw").is_file()
    assert (out / "provenance" / "cif" / "h-MoS2.cif").is_file()

    # The run script and the HPC job script, the two things the build exists to
    # produce. The old test asserted only that the methods writing them ran.
    slide = (out / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "4-layer model" in slide
    assert f"{out}/build/h-MoS2_4.lmp" in slide
    assert (out / "hpc" / "run.pbs").is_file()


def test_sheetonsheet_rejects_internal_lj_with_constraints(tmp_path: Path) -> None:
    """Internal-LJ potentials should be rejected for constrained modes.

    Un-mocked: `_create_directories` and `_init_provenance` were stubbed only to
    avoid file I/O ahead of the raise, and they write into `tmp_path` anyway.
    Letting them run keeps the two ratchet entries off this file and proves the
    refusal happens on the real path, not on a shortened one.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.sheet.pot_type = "reaxff"
    config.settings.simulation.constraint_mode = "atom_bonds"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    with pytest.raises(ValueError, match="does not support external LJ"):
        builder.build()

    # It refuses BEFORE building anything: no structure and no run script.
    assert not (tmp_path / "out" / "build").exists() or \
        not any((tmp_path / "out" / "build").iterdir())
    assert not (tmp_path / "out" / "lammps" / "slide.in").exists()


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


# --------------------------------------------------------------------------- #
# Phase 2c Task 11 -- builders-sheetonsheet.md findings 1-5.
#
# Everything below drives REAL material fixtures and asserts on the produced
# artifact (the written system.in.settings, the rendered slide.in, the
# converted data file), never on a mock call. None of it needs LAMMPS:
# _generate_potentials, _contact_cutoff and write_inputs are all pure
# bookkeeping plus a file write.
# --------------------------------------------------------------------------- #

MAT = "examples/materials"
POT = "examples/potentials"


def _real_material_config(
    *,
    constraint_mode: str = "none",
    pot_type: str = "sw",
    pot_path: str = f"{POT}/sw/MoS2_wen.sw",
    layers: int = 3,
    drive_method: str = "virtual_atom",
) -> SheetOnSheetSimulationConfig:
    """A sheet-on-sheet config on the REAL h-MoS2 CIF and a real potential file.

    `_make_sheetonsheet_config`'s dummy `.cif`/`.sw` are enough for the paths
    that only name files, but the potential spine and the UFF sigma lookup both
    parse them -- so the branches covered below need the real fixtures.
    """
    settings = load_settings()
    settings.simulation.constraint_mode = constraint_mode
    settings.simulation.drive_method = drive_method
    return SheetOnSheetSimulationConfig(**{
        "general": {"temp": 300.0, "scan_speed": 1},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": pot_path, "pot_type": pot_type,
               "x": 12.0, "y": 12.0, "layers": [layers]},
    }, settings=settings)


def _written_settings(config: SheetOnSheetSimulationConfig, tmp_path: Path) -> str:
    """Run the REAL _generate_potentials and return system.in.settings' text."""
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    (tmp_path / "out" / "lammps").mkdir(parents=True, exist_ok=True)
    builder._generate_potentials()
    return (tmp_path / "out" / "lammps" / "system.in.settings").read_text(encoding="utf-8")


def _interlayer_lj_epsilons(settings_text: str) -> Dict[str, float]:
    """`{'L1-L3': eps, ...}` for every real inter-LAYER lj/cut pair_coeff.

    The layer labels come from the comment the potential manager writes on each
    line (`# Mo(L1)-Mo(L3)`), so this reads the pairing the way a person would,
    not by re-deriving type ids the way the code under test does.
    """
    out: Dict[str, float] = {}
    for line in settings_text.splitlines():
        if not line.startswith("pair_coeff") or "lj/cut" not in line or "#" not in line:
            continue
        body, comment = line.split("#", 1)
        parts = body.split()
        if len(parts) < 5 or not parts[1].isdigit() or not parts[2].isdigit():
            continue          # `pair_coeff * N lj/cut ...` (the virtual atom)
        layers = re.findall(r"\(L(\d)\)", comment)
        if len(layers) == 2 and layers[0] != layers[1]:
            out[f"L{min(layers)}-L{max(layers)}"] = float(parts[4])
    return out


# --- finding 3: _generate_potentials' interlayer-LJ branches ---------------

def test_generate_potentials_none_gives_every_layer_pair_a_real_interlayer_lj(tmp_path):
    """constraint_mode='none' -> add_interlayer_interaction: REAL LJ everywhere.

    The discriminating pair is L1-L3 (non-adjacent). Adjacent pairs get a real
    epsilon under every mode, so asserting on those alone cannot tell the two
    branches apart.
    """
    text = _written_settings(_real_material_config(constraint_mode="none"), tmp_path)
    eps = _interlayer_lj_epsilons(text)

    assert set(eps) == {"L1-L2", "L1-L3", "L2-L3"}, eps
    assert all(value > 1e-6 for value in eps.values()), eps


@pytest.mark.parametrize("constraint_mode", ["atom_bonds", "com_spring"])
def test_generate_potentials_constrained_modes_ghost_out_non_adjacent_layers(
    constraint_mode, tmp_path,
):
    """atom_bonds/com_spring -> add_ghost_lj(max_real_distance=1).

    Adjacent layers keep their real LJ; NON-adjacent ones are zeroed to
    1e-100, because the bond/spring already couples the top pair and a second
    real interaction across the gap would double-count it.
    """
    text = _written_settings(
        _real_material_config(constraint_mode=constraint_mode), tmp_path)
    eps = _interlayer_lj_epsilons(text)

    assert eps["L1-L2"] > 1e-6, eps
    assert eps["L2-L3"] > 1e-6, eps
    assert eps["L1-L3"] == pytest.approx(1e-100), eps


def test_generate_potentials_internal_lj_potential_writes_no_interlayer_lj(tmp_path):
    """A potential with internal interlayer terms gets no explicit LJ at all.

    ReaxFF handles the interlayer interaction itself, so `_generate_potentials`
    takes neither the ghost nor the real branch. The one surviving lj/cut
    pair_coeff is the virtual drive atom's zero-LJ line, which is not an
    interlayer pair -- hence the comment-based filter above rather than a bare
    `'lj/cut' in text`.
    """
    config = _real_material_config(
        pot_type="reaxff", pot_path=f"{POT}/reaxff/ffield_Wang.reax.MoS")
    text = _written_settings(config, tmp_path)

    assert _interlayer_lj_epsilons(text) == {}
    assert "pair_style hybrid reaxff" in text


# --- finding 4: _register_drive_atom's virtual-atom body -------------------

def test_register_drive_atom_reserves_the_virtual_type_only_for_that_drive(tmp_path):
    """The base builder's positive case, which only its PES override tested.

    `test_pes_scan_builder.py::test_pes_sheet_no_virtual_drive_atom` pins the
    no-op override; nothing pinned that the base class DOES register the type
    for a virtual_atom drive, nor that it stays away for fix_move.
    """
    virtual = _written_settings(
        _real_material_config(drive_method="virtual_atom"), tmp_path / "v")
    fixed = _written_settings(
        _real_material_config(drive_method="fix_move"), tmp_path / "f")

    assert "Virtual atom" in virtual
    assert "lj/cut 1e-100 1e-100" in virtual
    assert "Virtual atom" not in fixed
    assert "1e-100" not in fixed


# --- finding 5: _contact_cutoff's success path -----------------------------

def test_contact_cutoff_derives_from_the_uff_sigma_of_a_real_cif():
    """The derivation, not the fallback -- only the fallback was covered.

    The expected value is re-derived here from the CIF and the UFF table
    directly, NOT read back from the builder: comparing the method's output
    against the attribute it computed would pass for any implementation.
    h-MoS2's largest same-sheet sigma is S-S, so the cutoff is K_CONTACT * that,
    well under the 0.95 * lj_cutoff cap (10.45 A) that would otherwise clamp it.
    """
    from itertools import product

    from src.builders.afm.base import K_CONTACT, _CONTACT_CUTOFF_FALLBACK
    from src.core.utils.structures import cifread
    from src.potentials.utils import lj_params

    config = _real_material_config()
    builder = SheetOnSheetSimulation(config, output_dir="unused")

    elements = cifread(config.sheet.cif_path)["elements"]
    sigma_max = max(lj_params(a, b)[1] for a, b in product(elements, elements))
    expected = K_CONTACT * sigma_max

    cutoff = builder._contact_cutoff()

    assert cutoff == pytest.approx(expected)
    assert cutoff != pytest.approx(_CONTACT_CUTOFF_FALLBACK)   # not the except path
    assert cutoff < 0.95 * config.settings.potential.lj_cutoff
    assert builder._contact_cutoff_source == f"uff_sigma_k{K_CONTACT}"


def test_contact_cutoff_is_capped_below_the_pair_cutoff():
    """LAMMPS errors if coord/atom's cutoff exceeds the pair cutoff, so the
    derived value is clamped. Drive the cap by shrinking lj_cutoff rather than
    by finding a material with a huge sigma."""
    config = _real_material_config()
    config.settings.potential.lj_cutoff = 4.0
    builder = SheetOnSheetSimulation(config, output_dir="unused")

    assert builder._contact_cutoff() == pytest.approx(0.95 * 4.0)


# --- findings 1 and 2: write_inputs' untested branches ---------------------

def _write_minimal_atomic_data(path: Path) -> Path:
    """A real (tiny) atom_style=atomic LAMMPS data file.

    `_seed_builder_state`'s `# dummy` placeholder is enough for the paths that
    only NAME the sheet file, but the molecular branch rewrites it in place, so
    that branch needs a file with a real Atoms section to rewrite.
    """
    path.write_text(
        "LAMMPS data\n\n"
        "2 atoms\n"
        "1 atom types\n\n"
        "0.0 10.0 xlo xhi\n"
        "0.0 10.0 ylo yhi\n"
        "0.0 10.0 zlo zhi\n\n"
        "Masses\n\n"
        "1 95.95\n\n"
        "Atoms # atomic\n\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    return path


def test_sheetonsheet_outer_loop_pressure_splits_slide_per_pressure(tmp_path):
    """outer_loop='pressure' emits one slide_p<p>gpa.in per pressure.

    Neither of the two SPLIT_OUTER_LOOPS values other than scan_angle was
    written by any test, while `SPLIT_OUTER_LOOPS` declares all three and
    `src.run.expand._run_scripts_per_build` predicts job counts from that same
    declaration -- see test_..._matches_the_predicted_job_count below.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = [1.0, 2.5]
    config.general.scan_speed = [2.0, 8.0]
    config.general.outer_loop = "pressure"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    lammps_dir = output_dir / "lammps"

    assert not (lammps_dir / "slide.in").exists()
    assert sorted(p.name for p in lammps_dir.glob("slide_*.in")) == [
        "slide_p1gpa.in", "slide_p2p5gpa.in",
    ]

    # Each script pins its OWN pressure as a scalar, while the inner speed axis
    # keeps the whole list (that is what makes it the inner loop).
    first = (lammps_dir / "slide_p1gpa.in").read_text(encoding="utf-8")
    assert "variable        pressure equal 1.0" in first
    assert "variable        speed index 2.0 8.0" in first
    second = (lammps_dir / "slide_p2p5gpa.in").read_text(encoding="utf-8")
    assert "variable        pressure equal 2.5" in second


def test_sheetonsheet_outer_loop_scan_speed_splits_slide_per_speed(tmp_path):
    """outer_loop='scan_speed' emits one slide_<v>ms.in per speed, with the
    pressure axis kept in-script."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = [1.0, 2.0]
    config.general.scan_speed = [2.0, 8.0]
    config.general.outer_loop = "scan_speed"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    lammps_dir = output_dir / "lammps"

    assert not (lammps_dir / "slide.in").exists()
    assert sorted(p.name for p in lammps_dir.glob("slide_*.in")) == [
        "slide_2ms.in", "slide_8ms.in",
    ]

    first = (lammps_dir / "slide_2ms.in").read_text(encoding="utf-8")
    assert "variable        speed equal 2.0" in first
    assert "variable        pressure index 1.0 2.0" in first


@pytest.mark.parametrize("outer_loop,values,field", [
    ("pressure", [1.0, 2.0, 3.0], "pressure"),
    ("scan_speed", [2.0, 8.0], "scan_speed"),
])
def test_sheetonsheet_outer_loop_split_matches_the_predicted_job_count(
    outer_loop, values, field, tmp_path,
):
    """What the builder writes must be what the job-count predictor promised.

    `write_inputs`' own comment says SPLIT_OUTER_LOOPS exists so that "the
    job-count prediction reads the same declaration, so what it predicts is
    what is written here". Nothing checked that for pressure or scan_speed, so
    the two could drift silently -- the GUI would promise N scripts and the
    build would write M.
    """
    from src.run.expand import _run_scripts_per_build

    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    setattr(config.general, field, values)
    config.general.outer_loop = outer_loop

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs()
    written = len(list((output_dir / "lammps").glob("slide*.in")))

    raw = {
        "general": {"temp": 300.0, "outer_loop": outer_loop, field: values},
        "2D": {"mat": "h-MoS2", "x": 20.0, "y": 20.0, "layers": [3]},
    }
    predicted, notes = _run_scripts_per_build(raw, "sheetonsheet")

    assert written == predicted == len(values), notes


def test_sheetonsheet_atom_bonds_renders_molecular_style_and_converts_the_sheet(tmp_path):
    """constraint_mode='atom_bonds' -> n_bond_types=1, atom_style molecular, and
    the sheet data file is rewritten in place by atomic2molecular.

    The conversion is the part that matters and the part nothing covered: it is
    data-file surgery on the file LAMMPS reads, and no sheet-family test ever
    invoked it. Asserted on the file's own Atoms header and column count, not
    on a call.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = 1.0
    config.settings.simulation.constraint_mode = "atom_bonds"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    sheet_path = _write_minimal_atomic_data(builder.structure_paths["sheet"])
    builder.write_inputs()

    slide = (output_dir / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "atom_style      molecular" in slide
    assert "constraint_mode: atom_bonds" in slide

    converted = sheet_path.read_text(encoding="utf-8")
    assert "Atoms # molecular" in converted
    assert "Atoms # atomic" not in converted
    # atomic `id type x y z` (5 cols) -> molecular `id mol type x y z` (6).
    assert converted.splitlines()[-1].split() == ["2", "0", "1", "1.0", "1.0", "1.0"]


def test_sheetonsheet_reaxff_renders_the_charge_atom_style(tmp_path):
    """A ReaxFF sheet renders atom_style charge and leaves the data file alone.

    Reachable only with constraint_mode='none' -- `build()` rejects ReaxFF with
    any other mode -- which is why `test_sheetonsheet_rejects_internal_lj_with_
    constraints` does not cover this branch despite naming ReaxFF.
    """
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.pressure = 1.0
    config.sheet.pot_type = "reaxff"
    config.settings.simulation.constraint_mode = "none"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    output_dir = _seed_builder_state(builder, tmp_path)
    sheet_path = _write_minimal_atomic_data(builder.structure_paths["sheet"])
    builder.write_inputs()

    slide = (output_dir / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "atom_style      charge" in slide
    assert "Atoms # atomic" in sheet_path.read_text(encoding="utf-8")   # untouched
