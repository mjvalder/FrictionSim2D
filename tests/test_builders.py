from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.builders.afm import AFMSimulation
from src.builders import components
from src.core.config import AFMSimulationConfig, load_settings


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mock_atomsk() -> MagicMock:
    return MagicMock(name="atomsk")


@pytest.fixture
def afm_config(temp_dir: Path) -> AFMSimulationConfig:
    pot_path = temp_dir / "dummy.sw"
    cif_path = temp_dir / "dummy.cif"
    pot_path.write_text("# dummy potential\n", encoding="utf-8")
    cif_path.write_text("# dummy cif\n", encoding="utf-8")

    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {
            'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot_path),
            'cif_path': str(cif_path), 'r': 10.0, 'dspring': 0.0, 's': 1.0,
        },
        'sub': {
            'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot_path),
            'cif_path': str(cif_path), 'thickness': 10.0,
        },
        '2D': {
            'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot_path),
            'cif_path': str(cif_path), 'x': 50.0, 'y': 50.0, 'layers': [1],
        },
        'lj_override': {},
        'settings': load_settings().model_dump(),
    }
    return AFMSimulationConfig(**data)


@patch("src.builders.components.run_lammps_commands")
def test_component_build_tip(mock_lmp, afm_config, mock_atomsk, temp_dir):
    """Tip builder returns expected path/radius and invokes LAMMPS commands."""
    settings = afm_config.settings

    def _mock_create_base_slab(*_args, **kwargs):
        kwargs['output_path'].write_text("# mock lammps data\n", encoding="utf-8")

    with patch("src.builders.components.get_material_path", return_value=Path("mock.cif")), \
         patch("src.builders.components._create_base_slab", side_effect=_mock_create_base_slab), \
         patch("src.builders.components.get_model_dimensions", return_value={'xhi': 20.0, 'xlo': 0.0, 'yhi': 20.0, 'ylo': 0.0}):
        path, radius = components.build_tip(afm_config.tip, mock_atomsk, temp_dir, settings)

    assert radius == 10.0
    assert path == temp_dir / "tip.lmp"
    mock_lmp.assert_called()


def test_afm_builder_structure(afm_config, mock_atomsk, temp_dir):
    """AFM builder orchestrates component generation for one-layer config."""
    with patch("src.builders.components.build_tip", return_value=(Path("tip.lmp"), 10.0)), \
         patch("src.builders.components.build_substrate", return_value=Path("sub.lmp")), \
         patch("src.builders.components.build_sheet", return_value=(Path("sheet.lmp"), {'xlo': 0, 'xhi': 10, 'ylo': 0, 'yhi': 10, 'zlo': 0, 'zhi': 10}, 3.0)), \
            patch("src.builders.components.apply_langevin_regions"), \
         patch("src.builders.afm.AFMSimulation._init_provenance"), \
         patch("src.builders.afm.AFMSimulation._generate_potentials", return_value=MagicMock()), \
         patch("src.builders.afm.AFMSimulation._calculate_z_positions"), \
         patch("src.builders.afm.AFMSimulation.write_inputs"), \
         patch("src.builders.afm.AFMSimulation._generate_hpc_scripts"):

        builder = AFMSimulation(afm_config, output_dir=str(temp_dir))
        builder.atomsk = mock_atomsk
        builder.build()

        assert 1 in builder.output_dir_layer
        assert builder.output_dir_layer[1].exists()
        assert (builder.output_dir_layer[1] / "lammps").exists()