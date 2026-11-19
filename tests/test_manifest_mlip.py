"""MLIP flavour inference in :class:`~src.hpc.manifest.JobManifest`.

`hpc generate` builds its manifest by *discovering* a built simulation tree
(`JobManifest.from_simulation_directory`), not by reading configs. So the only
place the GPU/framework facts survive into the manifest is the run-directory
flavour token that ``src.core.run.mlip_flavor_token`` stamps onto the leaf
segment (spec §1.3, e.g. ``K300_deepmd-lj-base``).

Without this inference the ``(needs_gpu, framework)`` array splitting added in
Task 16 can never fire in a real campaign: every job keeps the
``needs_gpu=False, framework=None`` defaults and MLIP work lands in the CPU
array with the wrong binary and no GPU request.
"""

from pathlib import Path

import pytest

from src.core.mlip import get_mlip
from src.core.run import mlip_flavor_token
from src.hpc.manifest import JobManifest, parse_mlip_flavor


def _make_sim(root: Path, rel: str, scripts=('system.in', 'slide.in')) -> None:
    """Create a discoverable simulation directory (a `lammps/` dir + scripts)."""
    lammps_dir = root / rel / 'lammps'
    lammps_dir.mkdir(parents=True, exist_ok=True)
    for name in scripts:
        (lammps_dir / name).write_text('# stub\n', encoding='utf-8')


class TestParseMLIPFlavor:
    """The token parser is the inverse of ``run.mlip_flavor_token``."""

    @pytest.mark.parametrize('pot_type,dispersion', [
        ('deepmd', 'none'),
        ('deepmd', 'd3'),
        ('mace', 'none'),
        ('mace', 'd3'),
    ])
    def test_round_trips_the_emitted_token(self, pot_type, dispersion):
        """Whatever run.py stamps, the manifest must be able to read back."""
        token = mlip_flavor_token({
            'pot_type': pot_type,
            'dispersion': dispersion,
        })
        assert token, 'precondition: an MLIP config must produce a token'
        parsed = parse_mlip_flavor(f"sheetonsheet/h-MoS2/50x_50y/K300{token}")
        assert parsed is not None
        # Coverage is constant now (full-cell only, v4 spec §3/§5).
        assert parsed == (pot_type, 'native',
                          'base' if dispersion == 'none' else 'd3')

    def test_still_decodes_a_legacy_lj_tree(self):
        """Campaigns built before v4 carry ``-lj-`` leaves; keep reading them.

        The token is the ONLY surviving record of which framework a discovered
        job needs, so an old tree that stopped decoding would silently land its
        MLIP jobs in the CPU array with the wrong binary.
        """
        assert parse_mlip_flavor('sheetonsheet/h-MoS2/50x_50y/K300_deepmd-lj-base') \
            == ('deepmd', 'lj', 'base')

    def test_finds_the_token_in_a_non_leaf_segment(self):
        """AFM nests an L<N> level BELOW the flavoured run segment."""
        path = 'afm/h-MoS2/50x_50y/sub_aSi_tip_Si_r25/K300_deepmd-native-base/L3'
        assert parse_mlip_flavor(path) == ('deepmd', 'native', 'base')

    @pytest.mark.parametrize('path', [
        'sheetonsheet/h-MoS2/50x_50y/K300',
        'afm/h-MoS2/50x_50y/sub_aSi_tip_Si_r25/K300',
        'sheetonsheet/h-MoS2/50x_50y/K0',
        '',
    ])
    def test_classical_paths_have_no_flavor(self, path):
        assert parse_mlip_flavor(path) is None

    def test_does_not_match_an_unknown_framework(self):
        """A material or directory that merely looks token-shaped is not a hit."""
        assert parse_mlip_flavor('sheetonsheet/foo/50x_50y/K300_tersoff-lj-base') is None


class TestManifestInfersGPUFields:
    """Discovery must populate the fields the array splitter groups on."""

    def test_classical_tree_is_unchanged(self, tmp_path):
        _make_sim(tmp_path, 'sheetonsheet/h-MoS2/50x_50y/K300')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        for job in manifest.jobs:
            assert job.needs_gpu is False
            assert job.framework is None
            assert job.launcher_flags == ''

    @pytest.mark.parametrize('framework,leaf', [
        ('deepmd', 'K300_deepmd-lj-base'),
        ('mace', 'K300_mace-native-d3'),
    ])
    def test_mlip_tree_sets_gpu_framework_and_flags(self, tmp_path, framework, leaf):
        _make_sim(tmp_path, f'sheetonsheet/h-MoS2/50x_50y/{leaf}')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        for job in manifest.jobs:
            assert job.needs_gpu is True
            assert job.framework == framework
            assert job.launcher_flags == get_mlip(framework).launcher.lmp_flags

    def test_mixed_campaign_splits_into_three_groups(self, tmp_path):
        """A classical + deepmd + mace campaign must yield three distinct keys."""
        _make_sim(tmp_path, 'sheetonsheet/h-MoS2/50x_50y/K300')
        _make_sim(tmp_path, 'sheetonsheet/h-MoS2/50x_50y/K300_deepmd-lj-base')
        _make_sim(tmp_path, 'sheetonsheet/h-MoS2/50x_50y/K300_mace-native-d3')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        keys = {(j.needs_gpu, j.framework) for j in manifest.jobs}
        assert keys == {(False, None), (True, 'deepmd'), (True, 'mace')}

    def test_afm_layer_subdirs_inherit_the_parent_flavor(self, tmp_path):
        base = 'afm/h-MoS2/50x_50y/sub_aSi_tip_Si_r25/K300_deepmd-native-base'
        _make_sim(tmp_path, f'{base}/L3')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        assert all(j.framework == 'deepmd' and j.needs_gpu for j in manifest.jobs)

    def test_round_trips_through_json(self, tmp_path):
        """The new fields must survive save/load (old manifests still load)."""
        _make_sim(tmp_path, 'sheetonsheet/h-MoS2/50x_50y/K300_deepmd-lj-base')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        out = tmp_path / 'manifest.json'
        manifest.save(out)
        reloaded = JobManifest.load(out)
        assert [(j.needs_gpu, j.framework, j.launcher_flags) for j in reloaded.jobs] == \
               [(j.needs_gpu, j.framework, j.launcher_flags) for j in manifest.jobs]
