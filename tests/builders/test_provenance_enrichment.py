"""Post-run provenance enrichment (spec §5 F5, review builders-afm.md AFM-3 / finding 5).

``AFMSimulation._write_provenance`` (build time) has always documented that
``RAMP_STATUS``/``UNLOADED_REF``/``lammps_version`` are "enriched post-run via
provenance.parse_log_extras", but nothing in ``frictionsim2d/`` ever called it -- a run
whose settle loop never converges (see ``afm/system_init.lmp``'s capped
``ramp_iter`` loop) produced no signal anywhere a user would see, only a
buried ``# RAMP_STATUS converged=0 ...`` log line nothing read back.
The module-level ``post_run.enrich_provenance_post_run`` is the reusable home for
that call: it reads the provenance.json a build already wrote, merges
``parse_log_extras(log_path)``'s fields into it, and rewrites it in place. It
takes no ``self`` because its real caller is a shell line in the generated HPC
job scripts, via ``FrictionSim2D hpc enrich-provenance`` -- which would
otherwise have to rebuild a whole ``AFMSimulationConfig`` to reach a method
that never read builder state.

These tests assert on the written provenance.json content directly (not a
mock) for both a converged and a non-converged log, per the review's
instruction.
"""
import json
import logging
from pathlib import Path

from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.builders.utils.post_run import enrich_provenance_post_run
from frictionsim2d.data.provenance import PROVENANCE_SCHEMA_VERSION
from tests.helpers.afm import _make_afm_config, _seed_builder_state


def _builder(tmp_path: Path) -> AFMSimulation:
    cfg = _make_afm_config(tmp_path, layers=[1])
    return AFMSimulation(cfg, str(tmp_path / "out"))


def _built_layer_dir(builder: AFMSimulation, tmp_path: Path) -> Path:
    """Seed builder state, write real inputs (incl. provenance.json), return L1."""
    layer_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs(1)
    return layer_dir


def _write_log(tmp_path: Path, name: str, text: str) -> Path:
    log_path = tmp_path / name
    log_path.write_text(text, encoding="utf-8")
    return log_path


def test_enrich_converged_true_updates_provenance(tmp_path):
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)

    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "LAMMPS (2 Aug 2023 - Update 3)\n"
        "# RAMP_STATUS converged=1 r=0.042 iterations=17\n"
        "# UNLOADED_REF layer=1 z=15.5\n",
    )

    result = enrich_provenance_post_run(layer_dir, log_path)
    assert result == layer_dir / "provenance.json"

    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["ramp_converged"] is True
    assert data["ramp_r"] == 0.042
    assert data["ramp_iterations"] == 17
    assert data["unloaded_reference_z"] == {"layer_1": 15.5}
    assert data["lammps_version"] == "2 Aug 2023 - Update 3"


def test_enrich_converged_false_updates_provenance_and_warns(tmp_path, caplog):
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)

    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=0 r=1.62 iterations=100\n",
    )

    with caplog.at_level(logging.WARNING):
        result = enrich_provenance_post_run(layer_dir, log_path)

    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["ramp_converged"] is False
    assert data["ramp_r"] == 1.62
    assert data["ramp_iterations"] == 100
    assert any("did not converge" in rec.message for rec in caplog.records)


def test_enrich_keeps_last_ramp_status_line(tmp_path):
    # multiple RAMP_STATUS lines (one per load) -- parse_log_extras keeps
    # the last, matching write order; the enrichment must carry that through.
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)

    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=0 r=0.9 iterations=100\n"
        "# RAMP_STATUS converged=1 r=0.01 iterations=8\n",
    )

    result = enrich_provenance_post_run(layer_dir, log_path)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["ramp_converged"] is True
    assert data["ramp_iterations"] == 8


def test_enrich_missing_log_returns_none_and_warns(tmp_path, caplog):
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    before = (layer_dir / "provenance.json").read_text(encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = enrich_provenance_post_run(
            layer_dir, tmp_path / "no_such_log.lammps")

    assert result is None
    assert (layer_dir / "provenance.json").read_text(encoding="utf-8") == before
    assert any("No log at" in rec.message for rec in caplog.records)


def test_enrich_missing_provenance_returns_none_and_warns(tmp_path, caplog):
    builder = _builder(tmp_path)
    empty_dir = tmp_path / "no_provenance_here"
    empty_dir.mkdir()
    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=1 r=0.01 iterations=3\n",
    )

    with caplog.at_level(logging.WARNING):
        result = enrich_provenance_post_run(empty_dir, log_path)

    assert result is None
    assert any("No provenance.json" in rec.message for rec in caplog.records)


def test_a_log_with_no_recognisable_lines_leaves_provenance_untouched(tmp_path):
    """The early-return branch: `parse_log_extras` found nothing at all, so the
    record is not rewritten.

    This is a defensive branch, not a shape production produces: every real
    LAMMPS run opens with a `LAMMPS (<version>)` banner, which alone is enough
    to return an update. A slide-phase log therefore *does* rewrite the record
    (with a version) -- what protects the system phase's settle result is the
    merge only touching keys the log supplied, not this branch. See the test
    below, and
    test_a_slide_phase_log_cannot_disturb_the_system_phase_s_settle_result
    in tests/cli/test_cli_hpc_group.py.
    """
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    before = (layer_dir / "provenance.json").read_text(encoding="utf-8")

    log_path = _write_log(tmp_path, "log_system.lammps", "nothing interesting here\n")

    result = enrich_provenance_post_run(layer_dir, log_path)
    assert result == layer_dir / "provenance.json"
    assert result.read_text(encoding="utf-8") == before


def test_enrich_truncated_provenance_returns_none_and_warns(tmp_path, caplog):
    """A killed job can leave a half-written provenance.json behind.

    That is precisely the failure this function exists to report on, so it
    must not be the one input that raises: pydantic's ``model_validate_json``
    throws ``ValidationError`` on a truncated body, and the docstring promises
    "logged, not raised".
    """
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    prov = layer_dir / "provenance.json"
    full = prov.read_text(encoding="utf-8")
    prov.write_text(full[:len(full) // 3], encoding="utf-8")  # truncated mid-object

    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=1 r=0.01 iterations=3\n",
    )

    with caplog.at_level(logging.WARNING):
        result = enrich_provenance_post_run(layer_dir, log_path)

    assert result is None
    assert any("Could not read provenance.json" in rec.message
               for rec in caplog.records)


def test_enrich_provenance_missing_required_fields_returns_none(tmp_path):
    """Well-formed JSON that is not a ProvenanceRecord is the same class of
    corruption (a partially flushed write can produce it) and must also be
    swallowed rather than raised."""
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    (layer_dir / "provenance.json").write_text("{}", encoding="utf-8")

    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=0 r=1.1 iterations=40\n",
    )

    assert enrich_provenance_post_run(layer_dir, log_path) is None


def test_a_realistic_slide_log_rewrites_only_the_version(tmp_path):
    """Companion to the branch test above, with a production-shaped log.

    A slide phase prints the version banner like every other run, so this does
    reach `write_provenance` -- and must still leave the system phase's fields
    exactly as they were.
    """
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)

    system_log = _write_log(
        tmp_path, "log_system.lammps",
        "LAMMPS (2 Aug 2023 - Update 3)\n"
        "# UNLOADED_REF layer=1 z=15.5\n"
        "# RAMP_STATUS converged=0 r=1.62 iterations=100\n")
    enrich_provenance_post_run(layer_dir, system_log)
    settled = json.loads(
        (layer_dir / "provenance.json").read_text(encoding="utf-8"))

    slide_log = _write_log(
        tmp_path, "log_slide.lammps",
        "LAMMPS (21 Nov 2023)\n"
        "Step Temp E_pair\n     0   300   -1.5\n"
        "Total wall time: 0:12:34\n")
    result = enrich_provenance_post_run(layer_dir, slide_log)

    assert result == layer_dir / "provenance.json"
    after = json.loads(result.read_text(encoding="utf-8"))
    for field in ("ramp_converged", "ramp_r", "ramp_iterations",
                  "unloaded_reference_z"):
        assert after[field] == settled[field], field
    # The one field a later log does take over, documented as accepted.
    assert after["lammps_version"] == "21 Nov 2023"


def test_enrich_records_ramp_and_load_per_load(tmp_path):
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    prov = tmp_path / "out" / "provenance"
    prov.mkdir(exist_ok=True)
    (prov / "config.json").write_text(
        json.dumps({"general": {"force": [10, 20]}}), encoding="utf-8")
    data = layer_dir / "data"
    data.mkdir(exist_ok=True)
    hdr = "# TimeStep v_hold_fz v_hold_tipz v_hold_shz\n"
    (data / "hold_10N.txt").write_text(hdr + "1000 1.5 5 6\n", encoding="utf-8")
    (data / "hold_20N.txt").write_text(hdr + "1000 2.5 5 6\n", encoding="utf-8")
    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=1 r=0.04 iterations=5 load=10\n"
        "# RAMP_STATUS converged=1 r=0.05 iterations=6 load=20\n")

    result = enrich_provenance_post_run(layer_dir, log_path)
    out = json.loads(result.read_text(encoding="utf-8"))
    assert set(out["ramp_by_load"]) == {"10", "20"}
    assert out["load_achieved_by_load"] == {"10": 1.5, "20": 2.5}
    assert out["load_achieved"] == 2.5


def test_load_achieved_without_configured_loads_is_the_largest_load(tmp_path):
    """No `general.force` to name the last load: `load_achieved` is the hold
    at the largest load, not the last file by name (hold_5N sorts after
    hold_10N)."""
    builder = _builder(tmp_path)
    layer_dir = _built_layer_dir(builder, tmp_path)
    prov = tmp_path / "out" / "provenance"
    prov.mkdir(exist_ok=True)
    (prov / "config.json").write_text(
        json.dumps({"general": {"force": None}}), encoding="utf-8")
    data = layer_dir / "data"
    data.mkdir(exist_ok=True)
    hdr = "# TimeStep v_hold_fz v_hold_tipz v_hold_shz\n"
    (data / "hold_10N.txt").write_text(hdr + "1000 9.5 5 6\n", encoding="utf-8")
    (data / "hold_5N.txt").write_text(hdr + "1000 4.5 5 6\n", encoding="utf-8")
    log_path = _write_log(
        tmp_path, "log_system.lammps",
        "# RAMP_STATUS converged=1 r=0.04 iterations=5 load=10\n")

    result = enrich_provenance_post_run(layer_dir, log_path)
    out = json.loads(result.read_text(encoding="utf-8"))
    assert out["load_achieved"] == 9.5


# --- run identity (Task 5b) -----------------------------------------------------

def test_an_afm_build_records_a_uuid4_run_uuid(tmp_path):
    from uuid import UUID
    builder = _builder(tmp_path)
    data = json.loads((_built_layer_dir(builder, tmp_path) / "provenance.json")
                      .read_text(encoding="utf-8"))
    assert UUID(data["run_uuid"]).version == 4
    assert "run_uuid_stamped_at" not in data


def test_a_rebuild_of_the_same_config_is_a_new_run(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    first = json.loads((_built_layer_dir(_builder(tmp_path / "a"), tmp_path / "a")
                        / "provenance.json").read_text(encoding="utf-8"))
    second = json.loads((_built_layer_dir(_builder(tmp_path / "b"), tmp_path / "b")
                         / "provenance.json").read_text(encoding="utf-8"))
    assert first["run_uuid"] != second["run_uuid"]


def test_enrichment_keeps_the_run_uuid(tmp_path):
    layer_dir = _built_layer_dir(_builder(tmp_path), tmp_path)
    prov = layer_dir / "provenance.json"
    before = json.loads(prov.read_text(encoding="utf-8"))["run_uuid"]
    log_path = _write_log(tmp_path, "log_system.lammps",
                          "LAMMPS (2 Aug 2023 - Update 3)\n"
                          "# RAMP_STATUS converged=1 r=0.042 iterations=17\n")

    enrich_provenance_post_run(layer_dir, log_path)

    assert json.loads(prov.read_text(encoding="utf-8"))["run_uuid"] == before


def test_enrichment_keeps_a_legacy_stamp(tmp_path):
    layer_dir = _built_layer_dir(_builder(tmp_path), tmp_path)
    prov = layer_dir / "provenance.json"
    data = json.loads(prov.read_text(encoding="utf-8"))
    data["run_uuid_stamped_at"] = "2026-11-16"
    prov.write_text(json.dumps(data), encoding="utf-8")
    stamped = json.loads(prov.read_text(encoding="utf-8"))
    log_path = _write_log(tmp_path, "log.lammps", "LAMMPS (2 Aug 2023)\n")

    enrich_provenance_post_run(layer_dir, log_path)

    after = json.loads(prov.read_text(encoding="utf-8"))
    assert after["run_uuid"] == stamped["run_uuid"]
    assert after["run_uuid_stamped_at"] == stamped["run_uuid_stamped_at"]
