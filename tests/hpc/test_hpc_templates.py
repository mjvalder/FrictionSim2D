"""Every hpc/*.j2 must be rendered by some test: they are reached only by
f-string name construction (`scripts.py` builds `f"{scheduler}_array.j2"` /
`f"{scheduler}_combined.j2"`), so a filename grep across src/ finds no static
reference and a typo in one scheduler/shape combination would ship undetected.

Renders through HPCScriptGenerator directly (not the now-deleted
create_hpc_package, which only ever took the plain array path -- see hpc.md
H-7), covering both schedulers and both script shapes so all four templates
are exercised. `generate_two_phase_scripts` reuses the *_array.j2 templates
already covered here, so it needs no separate rendering for template
liveness (its own dependency-submission logic is out of scope per the task's
Phase 2c deferral).
"""
from pathlib import Path

from src.hpc.manifest import JobEntry, JobManifest
from src.hpc.scripts import HPCConfig, HPCScriptGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "src" / "templates" / "hpc"


def _config() -> HPCConfig:
    config = HPCConfig()
    config.hpc_settings.modules = ["lammps"]
    return config


def _combined_manifest() -> JobManifest:
    manifest = JobManifest()
    manifest.add_job(JobEntry(job_id="sys", simulation_path="afm/mat0/L1",
                               lammps_script="system.in", material="mat0"))
    manifest.add_job(JobEntry(job_id="sld", simulation_path="afm/mat0/L1",
                               lammps_script="slide.in", material="mat0"))
    return manifest


def test_every_hpc_template_is_rendered_by_the_script_generator(tmp_path):
    texts = {}
    for scheduler in ("pbs", "slurm"):
        script_ext = ".pbs" if scheduler == "pbs" else ".sh"
        generator = HPCScriptGenerator(_config())

        array_dir = tmp_path / scheduler / "array"
        generator.generate_scripts(["mat0/L1"], array_dir, scheduler=scheduler)
        texts[(scheduler, "array")] = (array_dir / f"run{script_ext}").read_text(encoding="utf-8")

        combined_dir = tmp_path / scheduler / "combined"
        generator.generate_combined_scripts(_combined_manifest(), combined_dir, scheduler=scheduler)
        texts[(scheduler, "combined")] = (combined_dir / f"run{script_ext}").read_text(encoding="utf-8")

    on_disk = {p.name for p in TEMPLATE_DIR.glob("*.j2")}
    assert on_disk, f"no templates found under {TEMPLATE_DIR}"

    for template in on_disk:
        scheduler, _, shape = template.replace(".j2", "").partition("_")
        assert (scheduler, shape) in texts, (
            f"{template}: no test renders scheduler={scheduler!r} shape={shape!r}")
        text = texts[(scheduler, shape)]
        if shape == "combined":
            # Combined scripts hardcode the two-phase filenames rather than
            # reading them from a manifest file.
            assert "system.in" in text and "slide.in" in text, template
        else:
            # Array scripts resolve their per-task path from a manifest file.
            assert "sed -n" in text, template
