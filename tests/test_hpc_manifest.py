from pathlib import Path

import pytest

from src.core.config import GeneralConfig, parse_config
from src.core.config import write_config
from src.run.expand import count_expansion
from src.run.simulate import run_simulations
from src.hpc.manifest import JobManifest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


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


def test_from_simulation_directory_extracts_scan_angle_scripts(tmp_path: Path) -> None:
    """outer_loop='scan_angle' emits slide_a<token>.in per angle (Tasks 2-3);
    each one must become its own job entry with the angle recovered from the
    format_numeric_token encoding: plain int, 'm' negative prefix, 'p' decimal
    point. Mirrors the existing slide_f<load>N.in (force) coverage above."""
    lammps_dir = tmp_path / "afm" / "graphene" / "L2" / "lammps"
    lammps_dir.mkdir(parents=True)
    (lammps_dir / "slide_a0.in").write_text("# slide\n", encoding="utf-8")
    (lammps_dir / "slide_a30.in").write_text("# slide\n", encoding="utf-8")
    (lammps_dir / "slide_am15.in").write_text("# slide\n", encoding="utf-8")
    (lammps_dir / "slide_a2p5.in").write_text("# slide\n", encoding="utf-8")

    manifest = JobManifest.from_simulation_directory(tmp_path)

    assert manifest.n_jobs == 4
    by_script = {job.lammps_script: job for job in manifest.jobs}
    assert set(by_script) == {"slide_a0.in", "slide_a30.in", "slide_am15.in", "slide_a2p5.in"}
    assert by_script["slide_a0.in"].angle == 0.0
    assert by_script["slide_a30.in"].angle == 30.0
    assert by_script["slide_am15.in"].angle == -15.0
    assert by_script["slide_a2p5.in"].angle == 2.5
    # Angle scripts must not be misparsed as speed/force scripts (defaults
    # would silently mask the split).
    for job in manifest.jobs:
        assert job.speed == 2.0
        assert job.force == 0.0
        assert job.material == "graphene"
        assert job.layers == 2


def test_scan_angle_manifest_count_matches_predicted_job_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job-count prediction (_run_scripts_per_build, pinned separately in
    test_expansion_count.py) and the HPC manifest built from the real output
    tree are two independently-computed numbers. If the builder's per-angle
    split ever reaches the manifest differently than predicted, campaign-size
    reporting would silently lie -- this builds a real tree end-to-end and
    checks both agree."""
    monkeypatch.chdir(REPO_ROOT)
    raw = parse_config(EXAMPLES / 'afm_flake_config.ini')
    raw['general']['scan_angle'] = [0, 30, 60]
    raw['general']['outer_loop'] = 'scan_angle'
    validated_general = GeneralConfig(**raw['general'])
    raw['general'] = validated_general.model_dump()

    predicted = count_expansion(raw, 'afm')
    assert predicted.jobs == 3

    config_path = write_config(raw, tmp_path / 'scan_angle.ini')
    created, simulation_root, _, _ = run_simulations(
        config_file=str(config_path), model='afm', settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )
    assert len(created) == predicted.configs

    manifest = JobManifest.from_simulation_directory(simulation_root)
    slide_jobs = manifest.get_slide_jobs()
    assert len(slide_jobs) == predicted.jobs
    assert sorted(job.angle for job in slide_jobs) == [0.0, 30.0, 60.0]
