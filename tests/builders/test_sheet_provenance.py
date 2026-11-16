"""Every sheet-family run must write a provenance record, as AFM runs do.

Until `write_sheet_provenance` existed, **no sheet-on-sheet, hetero, flake or
PES-sheet run wrote a `provenance.json` at all** while every AFM run did:
`ProvenanceRecord.seed_sheet_center` ("sheet runs only") had no writer,
`SheetOnSheetSimulation._contact_cutoff` set a `_contact_cutoff_source` nothing
read, and `hpc enrich-provenance` reported `skipped` for every sheet run it
swept. Decision row B4-5.

The record must describe *this* run, not the family in general -- so the tests
below cross-check its fields against the script the same build emitted, rather
than against the builder that produced both.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.builders.sheetonsheet.base import (
    CUTOFF_NOT_APPLICABLE,
    SheetOnSheetSimulation,
    sheet_layer_type_map,
    write_sheet_provenance,
)
from src.builders.afm.base import _CONTACT_CUTOFF_FALLBACK
from src.data.provenance import ProvenanceRecord
from tests.builders.test_sheetonsheet_builder import (
    _make_sheetonsheet_config,
    _prepare_builder_for_slide_write,
    _real_material_config,
)

LMP = shutil.which("lmp_serial")


def _record(run_dir: Path) -> ProvenanceRecord:
    path = Path(run_dir) / "provenance.json"
    assert path.is_file(), sorted(p.name for p in Path(run_dir).iterdir())
    return ProvenanceRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _build_homogeneous(tmp_path: Path, layers=(3,), **general):
    """Render a homogeneous sheet build and return (run dir, slide text)."""
    config = _make_sheetonsheet_config(tmp_path, layers=list(layers))
    for key, value in general.items():
        setattr(config.general, key, value)
    builder = SheetOnSheetSimulation(config, str(tmp_path / "out"))
    slide = _prepare_builder_for_slide_write(builder, tmp_path)
    return builder.output_dir, slide.read_text(encoding="utf-8"), builder


# --------------------------------------------------------------------------
# The homogeneous slide builder
# --------------------------------------------------------------------------

def test_a_sheet_build_writes_a_valid_provenance_record(tmp_path):
    run_dir, _slide, _b = _build_homogeneous(tmp_path)
    record = _record(run_dir)
    assert record.schema_version
    assert record.run_role == "label"


def test_a_sheet_build_records_its_own_run_uuid(tmp_path):
    """Task 5b: every build is a new run, with a fresh uuid4."""
    from uuid import UUID
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    first = _record(_build_homogeneous(tmp_path / "a")[0]).run_uuid
    second = _record(_build_homogeneous(tmp_path / "b")[0]).run_uuid
    assert UUID(first).version == 4 and first != second


def test_the_recorded_seed_is_the_seed_baked_into_the_script(tmp_path):
    """The one field that proves the record belongs to THIS run.

    `_run_seed('sheet_center')` is drawn once per build and reaches both the
    script and the record; a record built from defaults, or from a second
    builder, would carry a different number.
    """
    run_dir, slide, _b = _build_homogeneous(tmp_path)
    record = _record(run_dir)

    assert record.seed_sheet_center is not None
    in_script = re.findall(r"langevin\s+[\d.]+\s+[\d.]+\s+\S+\s+(\d+)", slide)
    assert in_script, slide[:400]
    assert {int(s) for s in in_script} == {record.seed_sheet_center}


def test_the_record_is_written_before_the_outer_loop_split(tmp_path):
    """`write_inputs` has four exits; the record belongs to the build, not to
    whichever branch wrote the scripts."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.general.outer_loop = "pressure"
    config.general.pressure = [1.0, 2.0]
    builder = SheetOnSheetSimulation(config, str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)

    scripts = sorted((builder.output_dir / "lammps").glob("slide_p*.in"))
    assert len(scripts) == 2, [p.name for p in scripts]
    assert _record(builder.output_dir).seed_sheet_center is not None


def test_a_scalar_pressure_is_recorded_as_the_requested_pressure(tmp_path):
    """A sheet run is driven at constant pressure, not at a tip load, so the
    number belongs in pressure_requested and load_requested stays None."""
    run_dir, _slide, _b = _build_homogeneous(tmp_path, pressure=2.5)
    record = _record(run_dir)
    assert record.pressure_requested == pytest.approx(2.5)
    assert record.load_requested is None


def test_a_pressure_list_records_no_single_requested_pressure(tmp_path):
    """The list is an in-script loop: one run covers every value, so no single
    number is "the" requested pressure. Recording the first would be wrong."""
    run_dir, _slide, _b = _build_homogeneous(tmp_path, pressure=[1.0, 2.0, 3.0])
    assert _record(run_dir).pressure_requested is None


def test_the_layer_type_map_names_the_builders_own_layer_groups(tmp_path):
    run_dir, _slide, builder = _build_homogeneous(tmp_path)
    record = _record(run_dir)
    assert record.layer_type_map == {
        name: [int(t) for t in types.split()]
        for name, types in builder.groups.items()
        if name.startswith("layer_")
    }


def test_the_thermostat_scheme_names_the_group_the_template_thermostats(tmp_path):
    """`center` is layers 2..N-1 -- not AFM's tip/sub arrangement. The two
    families' records are read together, so the strings must not be confusable."""
    run_dir, _slide, _b = _build_homogeneous(tmp_path)
    assert _record(run_dir).thermostat_scheme == "langevin_center_nve"


def test_a_nose_hoover_run_records_nvt_not_langevin(tmp_path):
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.settings.thermostat.type = "nose-hoover"
    builder = SheetOnSheetSimulation(config, str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    assert _record(builder.output_dir).thermostat_scheme == "nose-hoover_center"


def test_the_recorded_cutoff_is_the_one_the_script_computes_with(tmp_path):
    """Opt `n_contact_il` in so the cutoff actually reaches a `coord/atom`
    line, then require the record to carry that same number.

    On the REAL h-MoS2 CIF, so `_contact_cutoff` takes its UFF-sigma route and
    lands well away from `_CONTACT_CUTOFF_FALLBACK`. With the dummy CIF the two
    coincide at 5.0 and a record that ignored the script entirely would still
    pass -- checked by mutation, which is how this fixture got chosen.
    """
    config = _real_material_config()
    config.settings.output.optional_channels = ["n_contact_il"]
    builder = SheetOnSheetSimulation(config, str(tmp_path / "out"))
    slide = _prepare_builder_for_slide_write(builder, tmp_path).read_text(
        encoding="utf-8")

    cutoffs = re.findall(r"coord/atom\s+cutoff\s+([\d.]+)", slide)
    assert cutoffs, [l for l in slide.splitlines() if "coord/atom" in l]
    record = _record(builder.output_dir)
    assert record.cutoff_source.startswith("uff_sigma"), record.cutoff_source
    assert record.contact_cutoff != _CONTACT_CUTOFF_FALLBACK, (
        "fixture must not coincide with the fallback, or the assertion below "
        "cannot tell a real derivation from a hardcoded constant")
    assert len(set(cutoffs)) == 1, cutoffs
    assert float(cutoffs[0]) == pytest.approx(record.contact_cutoff)


def test_the_cutoff_source_says_which_route_produced_it(tmp_path):
    """The dummy CIF cannot be read, so `_contact_cutoff` takes its documented
    fallback -- and the record must say so rather than presenting the fallback
    as a derived value."""
    run_dir, _slide, _b = _build_homogeneous(tmp_path)
    assert _record(run_dir).cutoff_source == "fallback_default"


def test_a_sheet_record_carries_no_tip_or_substrate_fields(tmp_path):
    """There is no tip, no separate velocity channel and no cached amorphous
    slab. Leaving AFM's fields at None is the honest record, not an omission."""
    record = _record(_build_homogeneous(tmp_path)[0])
    assert record.seed_thermal_tip is None
    assert record.seed_thermal_sub is None
    assert record.seed_velocity is None
    assert record.substrate_realization_id is None


# --------------------------------------------------------------------------
# materials (Task 6, spec §3 F3): every sheet, flake, tip and substrate
# --------------------------------------------------------------------------

def test_the_homogeneous_sheet_is_recorded_with_role_sheet(tmp_path):
    _run_dir, _slide, builder = _build_homogeneous(tmp_path)
    record = _record(builder.output_dir)
    roles = [m.role for m in record.materials]
    assert roles == ["sheet"]
    sheet = record.materials[0]
    assert sheet.name == builder.config.sheet.mat
    assert sheet.pot_type == builder.config.sheet.pot_type
    assert sheet.pot_file_sha256 is not None


def test_a_single_material_flake_still_gets_a_flake_role_material(tmp_path):
    """A `[Flake]` with no material of its own reuses `[2D]`'s potential
    (``FlakeSlideSimulation._filled_flake_config``); the record must not drop
    the flake from `materials` just because the RAW config carries no `mat` --
    the exact gap a review of this task found and this test pins.

    LAMMPS-free: constructing the builder and calling
    :func:`write_sheet_provenance` directly needs no geometry build, unlike
    :meth:`FlakeSlideSimulation.build`.
    """
    from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from tests.builders.test_flake_slide import _flake_on_sheet_cfg

    cfg = _flake_on_sheet_cfg(hetero_flake=False)
    assert cfg.flake.mat is None, "fixture must be the unfilled single-material flake"

    builder = FlakeSlideSimulation(cfg, str(tmp_path / "out"))
    write_sheet_provenance(builder, contact_cutoff=0.0,
                           cutoff_source="test", thermostat_scheme="test")

    record = _record(builder.output_dir)
    flake_entries = [m for m in record.materials if m.role == "flake"]
    assert len(flake_entries) == 1, record.materials
    flake_material = flake_entries[0]
    assert flake_material.name == cfg.sheet.mat
    assert flake_material.pot_type == cfg.sheet.pot_type
    assert flake_material.pot_file_sha256 is not None


def test_a_hetero_flake_is_recorded_with_its_own_material(tmp_path):
    """A [Flake-1]/[Flake-2] hetero flake already carries its own mat/pot, so
    `_filled_flake_config` is a no-op here -- the plain `config.flake` (still
    unfilled, a pure shape spec) must not be recorded a SECOND time on top of
    the two real `flake_sheets` entries."""
    from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from tests.builders.test_flake_slide import _flake_on_sheet_cfg

    cfg = _flake_on_sheet_cfg(hetero_flake=True)
    assert cfg.flake.mat is None, "the [Flake] shape spec itself names no material"

    builder = FlakeSlideSimulation(cfg, str(tmp_path / "out"))
    write_sheet_provenance(builder, contact_cutoff=0.0,
                           cutoff_source="test", thermostat_scheme="test")

    record = _record(builder.output_dir)
    flake_entries = [m for m in record.materials if m.role == "flake"]
    assert {m.name for m in flake_entries} == {fs.mat for fs in cfg.flake_sheets}
    assert len(flake_entries) == len(cfg.flake_sheets)


# --------------------------------------------------------------------------
# The static PES scan
# --------------------------------------------------------------------------

def _pes_builder(tmp_path: Path):
    from src.builders.sheetonsheet.pes_sheet import PESSheetSimulation
    from tests.builders.test_pes_scan_builder import _sheet_cfg

    builder = PESSheetSimulation(_sheet_cfg(tmp_path, layers=[2]),
                                 output_dir=str(tmp_path / "o"))
    builder.relative_run_dir = Path("pes_sheet/h-MoS2/30x_30y/K0")
    builder.pm = SimpleNamespace()
    builder.structure_paths = {"sheet": tmp_path / "h-MoS2_2.lmp"}
    builder.sheet_dims = {"xlo": 0.0, "xhi": 30.0, "ylo": 0.0,
                          "yhi": 30.0, "zhi": 20.0}
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.0, "ylo": 0.0, "yhi": 5.0}
    builder.lat_c = 6.5
    return builder


def test_a_pes_scan_writes_a_record_too(tmp_path):
    builder = _pes_builder(tmp_path)
    builder.write_inputs()
    assert _record(builder.output_dir).schema_version


def test_a_pes_scan_records_no_thermostat_and_no_meaningful_cutoff(tmp_path):
    """A static scan minimises at each grid point: no dynamics, no thermostat,
    and no coordination computed. `contact_cutoff` is a required float, so it
    gets 0.0 and a source saying the number means nothing -- better than a
    plausible value a reader would trust."""
    builder = _pes_builder(tmp_path)
    builder.write_inputs()
    record = _record(builder.output_dir)
    assert record.thermostat_scheme == "none_static_scan"
    assert record.contact_cutoff == 0.0
    assert record.cutoff_source == CUTOFF_NOT_APPLICABLE
    assert record.seed_sheet_center is None


def test_two_grid_variants_share_one_record(tmp_path):
    """slide.in and slide_refine.in are two resolutions of the same scan in one
    run directory, from one config."""
    builder = _pes_builder(tmp_path)
    builder.config.pes.grid_n_refine = 20
    builder.write_inputs()
    assert (builder.output_dir / "lammps" / "slide_refine.in").is_file()
    assert len(list(builder.output_dir.rglob("provenance.json"))) == 1


# --------------------------------------------------------------------------
# The layer-type-map helper's own contract
# --------------------------------------------------------------------------

def test_layer_groups_that_are_not_atom_types_map_to_nothing():
    """Hetero and flake partition by z-band region, so there are no per-layer
    type ids to record and {} is the honest answer -- the same thing hetero AFM
    writes, and why ProvenanceRecord allows the map to be empty."""
    assert sheet_layer_type_map(SimpleNamespace()) == {}
    assert sheet_layer_type_map(SimpleNamespace(groups={})) == {}
    assert sheet_layer_type_map(
        SimpleNamespace(groups={"center": "2", "all_types": "1 2 3"})) == {}


def test_only_layer_groups_are_read_not_every_group():
    assert sheet_layer_type_map(SimpleNamespace(groups={
        "layer_1": "1 4", "layer_2": "2 5",
        "center": "2 5", "all_types": "1 2 4 5",
    })) == {"layer_1": [1, 4], "layer_2": [2, 5]}


# --------------------------------------------------------------------------
# The two families that assemble through LAMMPS
# --------------------------------------------------------------------------

@pytest.mark.skipif(LMP is None,
                    reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_the_hetero_builder_writes_a_record(tmp_path):
    from tests.builders.test_hetero_slide import _hetero_2p2_config
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation

    builder = HeteroSheetOnSheetSimulation(_hetero_2p2_config(),
                                           str(tmp_path / "out"))
    builder.build()

    record = _record(builder.output_dir)
    assert record.seed_sheet_center is not None
    # z-band layers, not atom-type groups
    assert record.layer_type_map == {}
    assert record.cutoff_source == "fallback_default"


@pytest.mark.skipif(LMP is None,
                    reason="lmp_serial not on PATH (builder assembles via LAMMPS)")
def test_the_flake_builder_writes_a_record(tmp_path):
    from tests.builders.test_flake_slide import _flake_on_sheet_cfg
    from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    builder = FlakeSlideSimulation(_flake_on_sheet_cfg(hetero_flake=False),
                                   str(tmp_path / "out"))
    builder.build()

    record = _record(builder.output_dir)
    assert record.seed_sheet_center is not None
    assert record.thermostat_scheme == "langevin_center_nve"


# --------------------------------------------------------------------------
# The post-run sweep
# --------------------------------------------------------------------------

def test_a_sheet_run_is_enriched_after_the_run_instead_of_skipped(tmp_path):
    """`hpc enrich-provenance` used to report `skipped` for every sheet run it
    swept, for one reason: "no readable provenance.json". Now that the build
    writes one, the sweep folds the run's log into it like an AFM run's.

    A slide-phase log carries no RAMP_STATUS (that is the AFM settle loop), but
    every real LAMMPS log opens with a version banner, and recording which
    LAMMPS produced a result is the point of the field.
    """
    from src.builders.afm.base import enrich_provenance_post_run
    from src.cli import _enrich_one_run

    run_dir, _slide, _b = _build_homogeneous(tmp_path)
    assert _record(run_dir).lammps_version is None, "build time: not yet known"

    log = run_dir / "lammps" / "run_slide.log"
    log.write_text("LAMMPS (2 Aug 2023 - Update 3)\nStep Temp\n0 300\n",
                   encoding="utf-8")

    outcome, detail = _enrich_one_run(run_dir, None, enrich_provenance_post_run)
    assert outcome == "enriched", (outcome, detail)
    assert _record(run_dir).lammps_version == "2 Aug 2023 - Update 3"


def test_a_sheet_run_with_no_log_is_reported_not_crashed(tmp_path):
    """The sweep must never report a completed simulation as failed over a
    provenance nicety."""
    from src.builders.afm.base import enrich_provenance_post_run
    from src.cli import _enrich_one_run

    run_dir, _slide, _b = _build_homogeneous(tmp_path)
    outcome, detail = _enrich_one_run(run_dir, None, enrich_provenance_post_run)
    assert outcome == "skipped"
    assert "no run log" in detail


def test_sheet_provenance_records_drive_and_geometry(tmp_path):
    from src.data.models import NM_TO_EV_A2

    config_general = {"driving_spring": 50.0}
    run_dir, _slide, builder = _build_homogeneous(tmp_path, **config_general)
    record = _record(run_dir)
    assert record.drive_k == pytest.approx(50.0 / NM_TO_EV_A2)
    assert record.drive_rest_length == 0.0
    assert record.geometry is not None
    assert record.geometry.lat_c == builder.lat_c
    assert record.geometry.tip_r is None


def test_pes_sheet_records_no_drive(tmp_path):
    builder = _pes_builder(tmp_path)
    builder.write_inputs()
    record = _record(builder.output_dir)
    assert record.drive_k is None
    assert record.drive_rest_length is None
    assert record.geometry is None


# --------------------------------------------------------------------------
# The interlayer block (spec 2026-11-11 §7, Task 7)
# --------------------------------------------------------------------------

def _raw_provenance(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))


def test_provenance_interlayer_block(tmp_path, monkeypatch):
    """ilp: style, the bundled file by name, its sha256, comparison off."""
    from tests.builders.test_sheetonsheet_builder import _ilp_inputs
    out = _ilp_inputs(tmp_path, monkeypatch)
    block = _raw_provenance(out)["interlayer"]
    assert block["style"] == "ilp"
    assert block["file"] == "TMD.ILP"
    assert re.fullmatch(r"[0-9a-f]{64}", block["sha256"])
    assert block["comparison"] is False
    assert _record(out).interlayer == block       # round-trips through the model


def test_provenance_interlayer_block_records_comparison_mode(tmp_path, monkeypatch):
    from tests.builders.test_sheetonsheet_builder import _ilp_inputs
    out = _ilp_inputs(tmp_path, monkeypatch, interlayer_comparison=True)
    assert _raw_provenance(out)["interlayer"]["comparison"] is True


def test_lj_provenance_has_no_interlayer_key(tmp_path, monkeypatch):
    """lj without comparison: the record is byte-for-byte the pre-field one."""
    from tests.builders.test_sheetonsheet_builder import _build_example_sheet, _small
    out = _build_example_sheet(tmp_path, monkeypatch, **_small())
    assert "interlayer" not in _raw_provenance(out)
    assert _record(out).interlayer is None


def test_lj_comparison_mode_records_the_block(tmp_path, monkeypatch):
    """lj in comparison mode reports the reaction force, so the choice is recorded."""
    from tests.builders.test_sheetonsheet_builder import _build_example_sheet, _small
    out = _build_example_sheet(tmp_path, monkeypatch,
                               **_small(interlayer_comparison=True))
    block = _raw_provenance(out)["interlayer"]
    assert block["style"] == "lj" and block["comparison"] is True
    assert block["file"] is None and block["sha256"] is None


@pytest.mark.parametrize("override, source", [(None, "table"), (150.0, "override")])
def test_lj_g_provenance_g_source(tmp_path, monkeypatch, override, source):
    """lj_g: 'override' exactly when the sheet's interlayer_g was set."""
    from tests.builders.test_sheetonsheet_builder import _build_example_sheet, _small
    kwargs = _small(interlayer_style="lj_g")
    if override is not None:
        kwargs["sheet"]["interlayer_g"] = override
    out = _build_example_sheet(tmp_path, monkeypatch, **kwargs)
    g = _raw_provenance(out)["interlayer"]["g_meV"]
    assert g == {"sheet": {"value": override or 175.68, "source": source}}


def test_afm_provenance_interlayer_block(tmp_path, monkeypatch):
    """The AFM family records the choice too (ILP between the sheet layers)."""
    from tests.helpers.afm import build_example_afm
    out = build_example_afm(tmp_path, monkeypatch, "afm_config.ini",
                            general={"interlayer_style": "ilp"},
                            **{"2D": {"layers": [2], "lat_c": 6.15}})
    block = _raw_provenance(out / "L2")["interlayer"]
    assert block["style"] == "ilp" and block["file"] == "TMD.ILP"
    assert re.fullmatch(r"[0-9a-f]{64}", block["sha256"])
    assert block["comparison"] is False


def test_interlayer_provenance_block_rules():
    from src.builders.utils.interlayer import interlayer_provenance_block

    class _PM:
        def interlayer_provenance(self):
            return {"style": "ilp", "comparison": None}

    lj = SimpleNamespace(interlayer_style="lj", interlayer_comparison=False)
    ilp = SimpleNamespace(interlayer_style="ilp", interlayer_comparison=False)
    assert interlayer_provenance_block(_PM(), lj) is None
    assert interlayer_provenance_block(None, ilp) is None
    assert interlayer_provenance_block(_PM(), ilp) == {"style": "ilp", "comparison": False}
