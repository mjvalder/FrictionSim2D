from pathlib import Path

from src.hpc.scripts import HPCConfig, create_hpc_package


def test_create_hpc_package_writes_hpc_directory(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim_root"
    lammps_dir = sim_root / "afm" / "mat" / "L1" / "lammps"
    lammps_dir.mkdir(parents=True)
    (lammps_dir / "slide.in").write_text("# lammps\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    config = HPCConfig()
    config.hpc_settings.modules = ["lammps"]
    package_dir = create_hpc_package(sim_root, out_dir, scheduler="pbs", config=config)

    assert (package_dir / "hpc" / "run.pbs").exists()
    assert (package_dir / "hpc" / "submit_all.txt").exists()
    assert not (package_dir / "scripts").exists()

    readme = (package_dir / "README.md").read_text(encoding="utf-8")
    assert "cd hpc && bash submit_all.txt" in readme
