"""JobManifest is the read/write side of a file format: round-tripping matters.

Covers hpc.md (c) findings 1 (save/load round-trip) and 2 (JobEntry.update_status
timestamp/kwargs branching), plus the counts/query surface the CLI dispatch path
(`src/run/hpc_dispatch.py`) relies on. Deliberately does not exercise the six
`JobManifest` methods marked dead in hpc.md (b) H-1..H-6 (get_jobs_by_status,
get_jobs_by_material, update_job_status, mark_all_submitted,
mark_completed_from_results, get_summary) -- those are kept as an unfinished
work-in-progress layer (see docs/superpowers/specs/TO-DO/future-works.md), not
tested here.
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
        simulation_node_uuid="uuid-1", results_node_uuid="uuid-2",
        error_message=None,
    ))
    path = tmp_path / "manifest.json"
    manifest.save(path)

    reloaded = JobManifest.load(path)
    assert reloaded.name == manifest.name
    assert reloaded.source_directory == manifest.source_directory
    assert reloaded.scheduler == manifest.scheduler
    assert [j.to_dict() for j in reloaded.jobs] == [j.to_dict() for j in manifest.jobs]


def test_update_status_records_the_new_status_on_the_entry():
    entry = _entry(status=JobStatus.PREPARED)
    entry.update_status(JobStatus.SUBMITTED)
    assert entry.status == JobStatus.SUBMITTED.value


def test_update_status_sets_submitted_at_timestamp():
    entry = _entry(status=JobStatus.PREPARED)
    assert entry.submitted_at is None
    entry.update_status(JobStatus.SUBMITTED)
    assert entry.submitted_at is not None


def test_update_status_sets_completed_at_timestamp_for_terminal_states():
    entry = _entry(status=JobStatus.SUBMITTED)
    assert entry.completed_at is None
    entry.update_status(JobStatus.COMPLETED)
    assert entry.completed_at is not None


def test_update_status_applies_extra_kwargs():
    entry = _entry(status=JobStatus.SUBMITTED)
    entry.update_status(JobStatus.FAILED, error_message="boom")
    assert entry.status == JobStatus.FAILED.value
    assert entry.error_message == "boom"
