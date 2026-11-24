"""JobManifest is the read/write side of a file format: round-tripping matters.

Covers hpc.md (c) finding 1 (save/load round-trip), plus the counts/query surface
the CLI dispatch path (`frictionsim2d/run/hpc_dispatch.py`) relies on. The six `JobManifest`
status methods marked dead in hpc.md (b) H-1..H-6 (get_jobs_by_status,
get_jobs_by_material, update_job_status, mark_all_submitted,
mark_completed_from_results, get_summary), `JobEntry.update_status`, and the
AiiDA-uuid fields were deleted outright (spec Sec 4.3, Task 26) -- they had no
non-test caller, so the status they tracked was already fiction. Task 26 only
deleted the two per-job fields (`JobEntry.simulation_node_uuid`/
`results_node_uuid`); the two manifest-level fields
(`JobManifest.config_node_uuid`/`provenance_node_uuid`) were missed and are
deleted in the Task 34 review fix round. `test_status_half_is_gone` pins the
absence of all four.
"""
import json
from pathlib import Path

import pytest

from frictionsim2d.hpc.manifest import JobEntry, JobManifest


def _entry(job_id="j1", material="h-MoS2",
           script="slide.in", simulation_path="afm/mat0/L1") -> JobEntry:
    return JobEntry(job_id=job_id, simulation_path=simulation_path, material=material,
                     lammps_script=script)


def test_entry_round_trips_through_to_dict_and_from_dict():
    original = _entry()
    restored = JobEntry.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_n_jobs_counts_entries():
    manifest = JobManifest()
    manifest.add_job(_entry("a"))
    manifest.add_job(_entry("b"))
    assert manifest.n_jobs == 2
    assert repr(manifest) == "<JobManifest: friction2d_manifest (2 jobs)>"


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
    ))
    path = tmp_path / "manifest.json"
    manifest.save(path)

    reloaded = JobManifest.load(path)
    assert reloaded.name == manifest.name
    assert reloaded.source_directory == manifest.source_directory
    assert reloaded.scheduler == manifest.scheduler
    assert [j.to_dict() for j in reloaded.jobs] == [j.to_dict() for j in manifest.jobs]


def test_load_tolerates_legacy_uuid_fields_and_round_trips_without_them(tmp_path):
    """A manifest.json written before the status-half deletion (Task 26) has
    simulation_node_uuid/results_node_uuid on every job entry, and one written
    before the review fix round has config_node_uuid/provenance_node_uuid at
    the manifest level; this one also carries a bogus removed manifest-level
    key, to exercise the same tolerance at both levels and both eras.
    """
    legacy = {
        "name": "legacy-manifest",
        "created_at": "2026-01-01T00:00:00",
        "last_updated": "2026-01-01T00:00:00",
        "config_file": None,
        "source_directory": "sim_root",
        "config_node_uuid": "config-uuid-1",
        "provenance_node_uuid": "provenance-uuid-1",
        "scheduler": "pbs",
        "package_directory": None,
        "some_removed_manifest_key": "unused",
        "jobs": [{
            "job_id": "a",
            "simulation_path": "afm/mat0/L1",
            "lammps_script": "slide.in",
            "material": "h-MoS2",
            "layers": 1,
            "force": 0.0,
            "angle": 0.0,
            "speed": 2.0,
            "status": "completed",
            "hpc_job_id": None,
            "created_at": "2026-01-01T00:00:00",
            "submitted_at": None,
            "completed_at": "2026-01-02T00:00:00",
            "simulation_node_uuid": "uuid-1",
            "results_node_uuid": "uuid-2",
            "error_message": None,
        }],
    }
    path = tmp_path / "legacy_manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    manifest = JobManifest.load(path)

    assert manifest.name == "legacy-manifest"
    assert manifest.n_jobs == 1
    assert manifest.jobs[0].job_id == "a"
    assert not hasattr(manifest.jobs[0], "simulation_node_uuid")
    assert not hasattr(manifest.jobs[0], "results_node_uuid")
    assert not hasattr(manifest, "some_removed_manifest_key")
    assert not hasattr(manifest, "config_node_uuid")
    assert not hasattr(manifest, "provenance_node_uuid")

    resaved_path = tmp_path / "resaved_manifest.json"
    manifest.save(resaved_path)
    resaved_data = json.loads(resaved_path.read_text(encoding="utf-8"))
    assert "simulation_node_uuid" not in resaved_data["jobs"][0]
    assert "results_node_uuid" not in resaved_data["jobs"][0]
    assert "some_removed_manifest_key" not in resaved_data
    assert "config_node_uuid" not in resaved_data
    assert "provenance_node_uuid" not in resaved_data


def test_status_half_is_gone():
    from frictionsim2d.hpc.manifest import JobEntry as _JobEntry, JobManifest as _JobManifest

    for name in ('get_jobs_by_status', 'get_jobs_by_material', 'update_job_status',
                 'mark_all_submitted', 'mark_completed_from_results', 'get_summary'):
        assert not hasattr(_JobManifest, name), name
    assert not hasattr(_JobEntry, 'update_status')
    for name in ('get_job', 'n_prepared', 'n_submitted', 'n_completed', 'n_failed',
                 'n_imported'):
        assert not hasattr(_JobManifest, name), name
    for name in ('status', 'hpc_job_id', 'submitted_at', 'completed_at', 'error_message'):
        assert name not in _JobEntry.__dataclass_fields__, name
    assert 'simulation_node_uuid' not in _JobEntry.__dataclass_fields__
    assert 'results_node_uuid' not in _JobEntry.__dataclass_fields__
    assert 'config_node_uuid' not in _JobManifest.__dataclass_fields__
    assert 'provenance_node_uuid' not in _JobManifest.__dataclass_fields__
