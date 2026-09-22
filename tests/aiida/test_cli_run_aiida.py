"""`run --aiida` against a real store (spec §4.2, R8).

Lives here (not `tests/cli/`) so `aiida_profile_clean` -- registered only by
`tests/aiida/conftest.py` -- is in scope. `run_simulations` is still patched
(R29: no CLI test runs real generation), but `ingest_root` is the real
implementation: this is the one test proving an all-empty, freshly-generated
tree (no results written yet -- exactly what `run --aiida` hands ingest right
after generation) registers a set node and a run node and still exits 0.
"""
from pathlib import Path

import pytest

pytest.importorskip('aiida')

from click.testing import CliRunner  # noqa: E402

from src.cli import cli  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

pytestmark = pytest.mark.aiida

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def test_run_aiida_registers_an_all_empty_fresh_tree(aiida_profile_clean, tmp_path,
                                                      monkeypatch):
    def _fake_run_simulations(config_file, model, **kwargs):
        root = Path(kwargs['output_root']) / 'simulation_test'
        run_dir = make_run(root, REL, model)
        # `run --aiida` ingests immediately after generation, before any
        # LAMMPS job has produced results -- strip what the synthetic helper
        # wrote so the tree is genuinely all-empty.
        for path in (run_dir / 'results').glob('*'):
            path.unlink()
        return ([run_dir], root, [config_file], None)

    monkeypatch.setattr('src.cli.run_simulations', _fake_run_simulations)

    result = CliRunner().invoke(cli, [
        'run', 'afm', 'examples/afm_config.ini',
        '-o', str(tmp_path), '--aiida', '--label', 'empty-fresh'])

    assert result.exit_code == 0, result.output
    assert 'runs created:        1' in result.output
    assert 'empty runs:          1' in result.output

    from aiida.orm import Group
    group = Group.collection.get(label='friction2d/empty-fresh')
    assert len(group.nodes) == 2  # the set node + the one run node
