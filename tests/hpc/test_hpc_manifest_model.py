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


def test_an_entry_defaults_to_a_cpu_job() -> None:
    entry = JobEntry(job_id="job_0", simulation_path="afm/graphene/L2")
    assert entry.needs_gpu is False
    assert entry.framework is None
