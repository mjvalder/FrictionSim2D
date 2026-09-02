"""Per-material contact cutoff from tip-sheet UFF sigma (spec §5 F3, Task 6)."""
from pathlib import Path

import pytest

from src.builders.afm import AFMSimulation, K_CONTACT
from src.potentials.utils import lj_params
from tests.helpers.afm import _make_afm_config

MATERIALS = Path(__file__).parents[2] / 'examples' / 'materials'


def max_sigma_si_pair(sheet_elements):
    return max(lj_params('Si', el)[1] for el in sheet_elements)


def _builder(tmp_path, mat: str) -> AFMSimulation:
    cfg = _make_afm_config(tmp_path, layers=[2])
    cfg.sheet.mat = mat
    cfg.sheet.cif_path = str(MATERIALS / f'{mat}.cif')
    return AFMSimulation(cfg, str(tmp_path / 'out'))


def test_cutoff_scales_with_sigma(tmp_path):
    c_s = _builder(tmp_path, 'h-MoS2')._contact_cutoff()    # S anion
    c_te = _builder(tmp_path, 'h-MoTe2')._contact_cutoff()  # bigger anion
    assert c_te > c_s
    assert c_s == pytest.approx(K_CONTACT * max_sigma_si_pair(['Mo', 'S']),
                                rel=1e-9)


def test_cutoff_below_pair_cutoff(tmp_path):
    # LAMMPS errors if coord/atom's cutoff exceeds the pair cutoff
    b = _builder(tmp_path, 'h-MoTe2')
    assert b._contact_cutoff() < b.config.settings.potential.lj_cutoff


def test_cutoff_source_recorded(tmp_path):
    b = _builder(tmp_path, 'h-MoS2')
    b._contact_cutoff()
    assert b._contact_cutoff_source == f'uff_sigma_k{K_CONTACT}'


def test_dummy_cif_falls_back_loudly(tmp_path, caplog):
    cfg = _make_afm_config(tmp_path, layers=[2])   # dummy cif
    b = AFMSimulation(cfg, str(tmp_path / 'out'))
    cut = b._contact_cutoff()
    assert b._contact_cutoff_source == 'fallback_default'
    assert cut == pytest.approx(5.0)
