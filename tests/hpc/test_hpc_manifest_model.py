"""JobManifest is the read/write side of a file format: round-tripping matters.

Covers hpc.md (c) finding 1 (save/load round-trip), plus the counts/query surface
the CLI dispatch path (`src/run/hpc_dispatch.py`) relies on. The six `JobManifest`
status methods marked dead in hpc.md (b) H-1..H-6 (get_jobs_by_status,
get_jobs_by_material, update_job_status, mark_all_submitted,
mark_completed_from_results, get_summary), `JobEntry.update_status`, and the
AiiDA-uuid fields were deleted outright (spec Sec 4.3, Task 26) -- they had no
non-test caller, so the status they tracked was already fiction. `test_status_half_is_gone`
pins their absence.
"""
import json
from pathlib import Path

import pytest

from src.hpc.manifest import JobEntry, JobManifest, JobStatus


def _entry(job_id="j1", material="h-MoS2", status=JobStatus.PREPARED,
           script="slide.in", simulation_path="afm/mat0/L1") -> JobEntry:
    return JobEntry(job_id=job_id, simulation_path=simulation_path, material=material,
                     status=status.value, lammps_script=script)


def test_entry_round_trips_through_to_dict_and_from_dict():
    original = _entry()
    restored = JobEntry.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_counts_track_every_status_that_has_a_counter():
    manifest = JobManifest()
    manifest.add_job(_entry("a", status=JobStatus.PREPARED))
    manifest.add_job(_entry("b", status=JobStatus.SUBMITTED))
    manifest.add_job(_entry("c", status=JobStatus.RUNNING))
    manifest.add_job(_entry("d", status=JobStatus.COMPLETED))
    manifest.add_job(_entry("e", status=JobStatus.FAILED))
    manifest.add_job(_entry("f", status=JobStatus.IMPORTED))

    assert manifest.n_jobs == 6
    assert manifest.n_prepared == 1
    assert manifest.n_submitted == 2  # SUBMITTED and RUNNING both count here
    assert manifest.n_completed == 1
    assert manifest.n_failed == 1
    assert manifest.n_imported == 1


def test_get_job_returns_none_for_an_unknown_id():
    assert JobManifest().get_job("nope") is None


def test_system_and_slide_jobs_are_split_by_script_name():
    manifest = JobManifest()
    manifest.add_job(_entry("sys", script="system.in"))
    manifest.add_job(_entry("sld", script="slide.in"))

    assert [j.job_id for j in manifest.get_system_jobs()] == ["sys"]
    assert [j.job_id for j in manifest.get_slide_jobs()] == ["sld"]
    assert manifest.has_system_jobs() is True


def test_a_slide_only_manifest_reports_no_system_jobs():
    manifest = JobManifest()
    manifest.add_job(_entry("sld", script="slide.in"))
    assert manifest.has_system_jobs() is False


def test_save_writes_json_that_reloads_to_the_same_jobs(tmp_path):
    manifest = JobManifest()
    manifest.add_job(_entry("a"))
    path = tmp_path / "manifest.json"
    manifest.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["jobs"][0]["job_id"] == "a"


def test_save_then_load_round_trips_the_full_manifest(tmp_path):
    manifest = JobManifest(name="my-manifest", source_directory="sim_root", scheduler="slurm")
    manifest.add_job(JobEntry(
        job_id="a", simulation_path="afm/mat0/L1", lammps_script="slide.in",
        material="h-MoS2", layers=2, force=1.5, angle=30.0, speed=5.0,
        status=JobStatus.COMPLETED.value, hpc_job_id="123",
        submitted_at="2026-01-01T00:00:00", completed_at="2026-01-02T00:00:00",
        error_message=None,
    ))
    path = tmp_path / "manifest.json"
    manifest.save(path)

    reloaded = JobManifest.load(path)
    assert reloaded.name == manifest.name
    assert reloaded.source_directory == manifest.source_directory
    assert reloaded.scheduler == manifest.scheduler
    assert [j.to_dict() for j in reloaded.jobs] == [j.to_dict() for j in manifest.jobs]


def test_status_half_is_gone():
    from src.hpc.manifest import JobEntry as _JobEntry, JobManifest as _JobManifest

    for name in ('get_jobs_by_status', 'get_jobs_by_material', 'update_job_status',
                 'mark_all_submitted', 'mark_completed_from_results', 'get_summary'):
        assert not hasattr(_JobManifest, name), name
    assert not hasattr(_JobEntry, 'update_status')
    assert 'simulation_node_uuid' not in _JobEntry.__dataclass_fields__
    assert 'results_node_uuid' not in _JobEntry.__dataclass_fields__
