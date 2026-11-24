"""Pin the renders of the two templates no deck-snapshot case reaches.

``hetero/pes_scan.lmp`` and ``common/make_amorphous.lmp`` render empty fields
silently when a key is renamed; these goldens catch that and any template edit.
Refresh a golden by running this file with ``FS2D_WRITE_GOLDEN=1``.
"""
import os
from pathlib import Path

import pytest

from frictionsim2d.interfaces.jinja import render_template

_GOLDEN = Path(__file__).resolve().parents[1] / 'fixtures' / 'golden' / 'templates'

_HETERO_PES = {
    'atom_style': 'atomic', 'neighbor_list': 2.0, 'neigh_modify_command': 'neigh_modify delay 0',
    'xlo': 0.0, 'xhi': 10.0, 'ylo': 0.0, 'yhi': 12.0, 'box_zlo': -5.0, 'box_zhi': 20.0,
    'data_file': 'hetero.data', 'potential_file': 'lammps/system.in.settings',
    'layers': [{'idx': 1, 'zlo': -1.0, 'zhi': 1.0}, {'idx': 2, 'zlo': 2.0, 'zhi': 5.0}],
    'needs_molecule_ids': False, 'pot_type': 'reaxff', 'z_relax': True,
    'min_style': 'cg', 'minimization_command': 'minimize 1e-8 1e-10 1000 10000',
    'timestep': 0.001, 'thermo': 100, 'grid_n': 4, 'period_x': 3.2, 'period_y': 5.5,
    'n_cells': 6, 'interlayer_energy_substyle': 'ilp/tmd', 'results_csv': 'results/pes.csv',
    'data_dir': 'results',
}
_AMORPHOUS = {
    'neighbor_list': 2.0, 'neigh_modify_command': 'neigh_modify delay 0', 'base_block': 'base.data',
    'pot_commands': ['pair_style sw', 'pair_coeff * * x.sw Si'], 'min_style': 'cg',
    'minimization_command': 'minimize 1e-8 1e-10 1000 10000', 'timestep': 0.001,
    'melt_temp': 3000, 'melt_seed': 42, 'melt_steps': 1000, 'quench_temp': 300,
    'quench_nsteps': 5000, 'equilibrate_steps': 2000, 'amor_file': 'amor.data',
}


def _variants():
    yield 'hetero_pes_scan_relax', 'hetero/pes_scan.lmp', _HETERO_PES
    yield 'hetero_pes_scan_rigid', 'hetero/pes_scan.lmp', {
        **_HETERO_PES, 'z_relax': False, 'pot_type': 'sw', 'interlayer_energy_substyle': None,
        'mlip_sheet': True, 'mlip_deck_directives': ['pair_style mlip']}
    yield 'make_amorphous', 'common/make_amorphous.lmp', _AMORPHOUS


@pytest.mark.parametrize('name,template,ctx', list(_variants()), ids=lambda v: v if isinstance(v, str) else '')
def test_render_matches_golden(name, template, ctx):
    text = render_template(template, ctx)
    path = _GOLDEN / f'{name}.lmp'
    if os.environ.get('FS2D_WRITE_GOLDEN'):
        path.write_text(text, encoding='utf-8')
    assert text == path.read_text(encoding='utf-8')
