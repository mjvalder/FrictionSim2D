from pathlib import Path

from src.hpc.manifest import JobManifest


def test_from_simulation_directory_extracts_material_layers_and_slide_speed(tmp_path: Path) -> None:
    lammps_dir = tmp_path / "afm" / "graphene" / "L2" / "lammps"
    lammps_dir.mkdir(parents=True)
    (lammps_dir / "slide_20ms.in").write_text("# slide\n", encoding="utf-8")

    manifest = JobManifest.from_simulation_directory(tmp_path)

    assert manifest.n_jobs == 1
    job = manifest.jobs[0]
    assert job.material == "graphene"
    assert job.layers == 2
    assert job.lammps_script == "slide_20ms.in"
    assert job.speed == 20.0
