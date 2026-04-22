from pathlib import Path
from types import SimpleNamespace

from src.core.simulation_base import SimulationBase


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
        "src.core.simulation_base.get_material_path",
        lambda _mat: (_ for _ in ()).throw(FileNotFoundError("missing material")),
    )
    monkeypatch.setattr(
        "src.core.simulation_base.get_potential_path",
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
    monkeypatch.setattr("src.core.simulation_base.get_potential_path", lambda _pot: None)

    component_cfg = SimpleNamespace(mat="MoS2", cif_path=str(cif_path), pot_path=None)

    sim._add_component_files_to_provenance("sheet", component_cfg)

    assert calls == [(cif_path, "cif", "sheet")]
