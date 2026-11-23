from pathlib import Path
from types import SimpleNamespace

from frictionsim2d.builders.simulation_base import SimulationBase


class _ConcreteSimulation(SimulationBase):
    def build(self):
        return None

    def write_inputs(self):
        return None


def _make_simulation(tmp_path: Path) -> _ConcreteSimulation:
    cfg = SimpleNamespace(settings=SimpleNamespace(hpc=SimpleNamespace()))
    return _ConcreteSimulation(cfg, output_dir=tmp_path)


def test_add_component_files_to_provenance_ignores_missing_resolved_paths(tmp_path, monkeypatch):
    """Resolution failures should not break provenance collection flow."""
    sim = _make_simulation(tmp_path)

    monkeypatch.setattr(
        "frictionsim2d.builders.simulation_base.get_material_path",
        lambda _mat: (_ for _ in ()).throw(FileNotFoundError("missing material")),
    )
    monkeypatch.setattr(
        "frictionsim2d.builders.simulation_base.get_potential_path",
        lambda _pot: (_ for _ in ()).throw(FileNotFoundError("missing potential")),
    )

    component_cfg = SimpleNamespace(mat="MoS2", cif_path=None, pot_path="missing.sw")

    sim._add_component_files_to_provenance("sheet", component_cfg)


def test_add_component_files_to_provenance_keeps_existing_explicit_paths(tmp_path, monkeypatch):
    """Existing explicit cif_path should still be forwarded to provenance."""
    sim = _make_simulation(tmp_path)

    cif_path = tmp_path / "mat.cif"
    cif_path.write_text("data", encoding="utf-8")

    calls = []

    def _record(path, category, component=None):
        calls.append((Path(path), category, component))
        return Path(path)

    monkeypatch.setattr(sim, "add_to_provenance", _record)
    monkeypatch.setattr("frictionsim2d.builders.simulation_base.get_potential_path", lambda _pot: None)

    component_cfg = SimpleNamespace(mat="MoS2", cif_path=str(cif_path), pot_path=None)

    sim._add_component_files_to_provenance("sheet", component_cfg)

    assert calls == [(cif_path, "cif", "sheet")]


def _sim(tmp_path, **cfg):
    cfg.setdefault("settings", SimpleNamespace(hpc=SimpleNamespace()))
    return _ConcreteSimulation(SimpleNamespace(**cfg), output_dir=tmp_path)


def test_add_to_provenance_merges_components_for_one_file(tmp_path):
    import json
    sim = _sim(tmp_path)
    src = tmp_path / "Si.sw"
    src.write_text("potential", encoding="utf-8")
    sim.add_to_provenance(src, "potential", component="tip")
    sim.add_to_provenance(src, "potential", component="sub")
    files = json.loads((tmp_path / "provenance" / "manifest.json").read_text())["files"]
    assert len(files) == 1
    assert files[0]["components"] == ["tip", "sub"]


def test_add_to_provenance_dedupes_same_checksum_other_path(tmp_path):
    import json
    sim = _sim(tmp_path)
    a, b = tmp_path / "a.sw", tmp_path / "sub" / "b.sw"
    b.parent.mkdir()
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    sim.add_to_provenance(a, "potential", component="tip")
    sim.add_to_provenance(b, "potential", component="sub")
    files = json.loads((tmp_path / "provenance" / "manifest.json").read_text())["files"]
    assert len(files) == 1
    assert files[0]["components"] == ["tip", "sub"]


def test_git_identity_falls_back_when_git_is_absent(monkeypatch):
    def _no_git(*_a, **_k):
        raise FileNotFoundError("git")
    import subprocess
    monkeypatch.setattr(subprocess, "run", _no_git)
    assert SimulationBase._git_identity() == ("unknown", False)


def test_simulation_root_of_a_standalone_builder(tmp_path):
    run_dir = Path("afm") / "MoS2" / "K300"
    sim = _ConcreteSimulation(SimpleNamespace(), output_dir=tmp_path / run_dir)
    sim.relative_run_dir = run_dir
    assert sim.simulation_root == tmp_path.resolve()


def test_write_file_checks_in_decks_against_the_z_box(tmp_path):
    import pytest
    from frictionsim2d.builders.utils.z_vacuum import ZBox
    sim = _sim(tmp_path)
    sim.z_box = ZBox(zmin=0.0, zmax=10.0, cutoff=10.0, skin=2.0, margin=1.0)
    bad = "region box block 0 1 0 1 0 5\n"
    with pytest.raises(ValueError):
        sim.write_file("lammps/bad.in", bad)
    assert sim.write_file("lammps/bad.txt", bad).exists()


def test_hpc_job_name_is_prefix_and_material_id():
    class _Named(_ConcreteSimulation):
        HPC_JOB_PREFIX = "demo"

        def _material_id(self):
            return "A+B"

    assert _Named(SimpleNamespace(), output_dir=".")._get_hpc_job_name() == "demo_A+B"


def test_family_job_name_prefixes():
    from frictionsim2d.builders.afm.base import AFMSimulation
    from frictionsim2d.builders.afm.freestanding import FreestandingPESTipSimulation
    from frictionsim2d.builders.afm.hetero import HeteroIndentSimulation, HeteroAFMSimulation
    from frictionsim2d.builders.afm.indent import IndentSimulation
    from frictionsim2d.builders.afm.pes_tip import HeteroPESTipSimulation, PESTipSimulation
    from frictionsim2d.builders.sheetonsheet.base import SheetOnSheetSimulation
    from frictionsim2d.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
    from frictionsim2d.builders.sheetonsheet.pes_sheet import (
        HeteroPESSheetSimulation, PESSheetSimulation)
    expected = {
        AFMSimulation: "afm", HeteroAFMSimulation: "afm", IndentSimulation: "indent",
        HeteroIndentSimulation: "indent", PESTipSimulation: "pes_tip",
        HeteroPESTipSimulation: "pes_tip", FreestandingPESTipSimulation: "pes_tip_free",
        SheetOnSheetSimulation: "sheet", HeteroSheetOnSheetSimulation: "friction",
        PESSheetSimulation: "pes_sheet", HeteroPESSheetSimulation: "pes_sheet",
    }
    for cls, prefix in expected.items():
        assert cls.HPC_JOB_PREFIX == prefix, cls.__name__
