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
)
from src.data.provenance import ProvenanceRecord
from tests.builders.test_sheetonsheet_builder import (
    _make_sheetonsheet_config,
    _prepare_builder_for_slide_write,
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
    line, then require the record to carry that same number. Passing the value
    in rather than recomputing it is what makes this hold."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.settings.output.optional_channels = ["n_contact_il"]
    builder = SheetOnSheetSimulation(config, str(tmp_path / "out"))
    slide = _prepare_builder_for_slide_write(builder, tmp_path).read_text(
        encoding="utf-8")

    cutoffs = re.findall(r"coord/atom\s+cutoff\s+([\d.]+)", slide)
    assert cutoffs, [l for l in slide.splitlines() if "coord/atom" in l]
    record = _record(builder.output_dir)
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
