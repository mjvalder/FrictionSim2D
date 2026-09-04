"""`hpc generate`: artifact tests over a real simulation tree.

`hpc generate` writes files rather than calling one obvious expensive
boundary, so these tests assert on the files it produces and on
`result.output` -- never on a mock (tests/README.md).
"""
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli
from src.core.config import GlobalSettings
from src.run.hpc_dispatch import collect_hpc_simulation_paths


@pytest.fixture
def runner():
    return CliRunner()


def _sim_tree(tmp_path: Path, script: str = "slide.in", n_sims: int = 1) -> Path:
    """A simulation tree shaped like a real one: <root>/afm/<mat>/L1/lammps/<script>."""
    root = tmp_path / "sim_root"
    for i in range(n_sims):
        lammps = root / "afm" / f"mat{i}" / "L1" / "lammps"
        lammps.mkdir(parents=True)
        (lammps / script).write_text("# lammps\n", encoding="utf-8")
    return root


def _settings_file_with_modules(tmp_path: Path) -> Path:
    """`hpc generate` calls HPCScriptGenerator._validate_config, which raises
    unless settings.hpc.modules is set -- the default GlobalSettings() leaves
    it None."""
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    return path


def test_scheduler_flag_picks_the_pbs_script(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--scheduler", "pbs"])

    assert result.exit_code == 0, result.output
    assert (sim_root / "hpc" / "run.pbs").exists()
    assert not (sim_root / "hpc" / "run.sh").exists()


def test_scheduler_flag_picks_the_slurm_script(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--scheduler", "slurm"])

    assert result.exit_code == 0, result.output
    assert (sim_root / "hpc" / "run.sh").exists()
    assert not (sim_root / "hpc" / "run.pbs").exists()


def test_a_tree_with_no_lammps_directory_exits_non_zero(runner, tmp_path):
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    (empty_root / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "generate", str(empty_root)])

    assert result.exit_code != 0
    assert "No simulation directories found" in result.output
    # hpc/ is created up front (before discovery), so it exists-and-is-empty
    # today; the right fix is moving that mkdir after the refusal checks, at
    # which point hpc/ would never be created here at all. Tolerate both so
    # this test doesn't block that fix.
    hpc_dir = empty_root / "hpc"
    assert not hpc_dir.exists() or list(hpc_dir.iterdir()) == []


def test_output_dir_option_is_honoured(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)
    out_dir = tmp_path / "somewhere_else"

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--output-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "run.pbs").exists()
    assert not (sim_root / "hpc").exists()


def test_discovery_agrees_with_collect_hpc_simulation_paths_for_suffixed_scripts(
        runner, tmp_path):
    """C2-1 equivalence check.

    ``hpc_generate`` used to re-implement simulation discovery inline with a
    private ``script_exists`` that special-cased only ``slide.in`` (matching
    ``slide_*.in``), never ``system.in`` -- so a directory containing only a
    suffixed system script (e.g. ``system_30nN.in``, the naming a
    per-condition system-phase sweep would use, mirroring slide's existing
    per-load suffixes) was invisible to the CLI even though
    ``run.hpc_dispatch.collect_hpc_simulation_paths`` (used by every other
    caller: ``run --hpc-scripts``, ``aiida rebuild --hpc-scripts``) already
    finds it via ``*system*.in``. This must find both directories, matching
    ``collect_hpc_simulation_paths`` exactly -- it fails against the
    pre-refactor CLI, which only ever found the slide one.
    """
    sim_root = tmp_path / "sim_root"

    system_dir = sim_root / "afm" / "mat_system" / "L1" / "lammps"
    system_dir.mkdir(parents=True)
    (system_dir / "system_30nN.in").write_text("# lammps\n", encoding="utf-8")

    slide_dir = sim_root / "afm" / "mat_slide" / "L1" / "lammps"
    slide_dir.mkdir(parents=True)
    (slide_dir / "slide_p1gpa.in").write_text("# lammps\n", encoding="utf-8")

    settings_file = _settings_file_with_modules(tmp_path)

    expected = collect_hpc_simulation_paths(sim_root)
    assert len(expected) == 2, expected  # sanity: both dirs are real candidates

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file)])

    assert result.exit_code == 0, result.output
    assert f"Found {len(expected)} simulations" in result.output


def test_missing_modules_is_blocked_before_anything_is_written(runner, tmp_path):
    """cli.md finding 2 (fix, user-approved).

    On the common AFM path (1:1 system.in/slide.in per simulation directory)
    ``dispatch_hpc_generation`` calls ``generate_combined_scripts``, which
    never reaches ``HPCScriptGenerator._validate_config`` -- so with no HPC
    modules configured, ``hpc generate`` used to exit 0 and write a
    ``run.pbs`` containing ``module purge`` and no ``module load`` line: a
    job script that only fails once submitted to the cluster.
    ``hpc_generation_blocker`` (the same precondition
    ``generate_hpc_scripts_for_root`` already enforces for the GUI/
    ``run --hpc-scripts`` path) must gate this before any script or manifest
    file is written.
    """
    sim_root = tmp_path / "sim_root"
    lammps = sim_root / "afm" / "mat0" / "L1" / "lammps"
    lammps.mkdir(parents=True)
    (lammps / "system.in").write_text("# lammps\n", encoding="utf-8")
    (lammps / "slide.in").write_text("# lammps\n", encoding="utf-8")

    # No --settings-file: default GlobalSettings() leaves hpc.modules unset.
    result = runner.invoke(cli, ["hpc", "generate", str(sim_root)])

    assert result.exit_code != 0
    assert "'Modules'" in result.output
    assert not (sim_root / "hpc" / "run.pbs").exists()
