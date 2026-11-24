"""JobManifest in-memory surface: counts, system/slide split, no tracker or file format."""
from frictionsim2d.hpc.manifest import JobEntry, JobManifest


def _entry(job_id="j1", material="h-MoS2",
           script="slide.in", simulation_path="afm/mat0/L1") -> JobEntry:
    return JobEntry(job_id=job_id, simulation_path=simulation_path, material=material,
                     lammps_script=script)


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
    for name in ('save', 'load'):
        assert not hasattr(_JobManifest, name), name
    for name in ('to_dict', 'from_dict'):
        assert not hasattr(_JobEntry, name), name


def test_slide_script_names_carry_speed_force_and_angle(tmp_path):
    lammps = tmp_path / "afm" / "graphene" / "L1" / "lammps"
    lammps.mkdir(parents=True)
    for name in ("slide_5p5ms.in", "slide_f2p5N.in", "slide_am30.in", "slide_a15p5.in", "slide.in"):
        (lammps / name).write_text("# slide\n", encoding="utf-8")

    jobs = {j.lammps_script: j for j in JobManifest.from_simulation_directory(tmp_path).jobs}

    assert (jobs["slide_5p5ms.in"].speed, jobs["slide_5p5ms.in"].force) == (5.5, 0.0)
    assert (jobs["slide_f2p5N.in"].force, jobs["slide_f2p5N.in"].speed) == (2.5, 2.0)
    assert jobs["slide_am30.in"].angle == -30.0
    assert jobs["slide_a15p5.in"].angle == 15.5
    assert (jobs["slide.in"].speed, jobs["slide.in"].force, jobs["slide.in"].angle) == (2.0, 0.0, 0.0)
    assert jobs["slide_f2p5N.in"].job_id == "afm_graphene_L1_slide_f2p5N"
