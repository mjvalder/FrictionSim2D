"""MLIP flavour inference in :class:`~frictionsim2d.hpc.manifest.JobManifest`.

`hpc generate` builds its manifest by *discovering* a built simulation tree
(`JobManifest.from_simulation_directory`), not by reading configs. So the only
place the GPU/framework facts survive into the manifest is the run path: the
``pot_<slug>_<stem>`` segment ``frictionsim2d.run.naming.output_dir_for`` emits for an
MLIP sheet (``pot_mace_<stem>`` / ``pot_deepmd-d3_<stem>``, merge decision 1).

Without this inference the ``(needs_gpu, framework)`` job grouping can never
fire in a real campaign: every job keeps the ``needs_gpu=False,
framework=None`` defaults and MLIP work lands in the CPU array with the wrong
binary and no GPU request.
"""

from pathlib import Path

import pytest

from frictionsim2d.hpc.manifest import JobManifest, parse_mlip_flavor
from frictionsim2d.run.naming import output_dir_for

SHEET = 'sheetonsheet/h-MoS2/50x_50y/K300'
AFM = 'afm/h-MoS2/50x_50y/sub_amorph-Si_tip_Si_r25/K300'


def _make_sim(root: Path, rel: str, scripts=('system.in', 'slide.in')) -> None:
    """Create a discoverable simulation directory (a `lammps/` dir + scripts)."""
    lammps_dir = root / rel / 'lammps'
    lammps_dir.mkdir(parents=True, exist_ok=True)
    for name in scripts:
        (lammps_dir / name).write_text('# stub\n', encoding='utf-8')


class TestParseMLIPFlavor:
    """The parser is the inverse of what ``output_dir_for`` stamps."""

    @pytest.mark.parametrize('pot_type,model,dispersion', [
        ('deepmd', 'dpa3.pth', 'none'),
        ('deepmd', 'dpa3.pb', 'd3'),
        ('mace', 'mace-mp.json', 'none'),
        ('mace', 'mace-mp.json', 'd3'),
    ])
    def test_round_trips_the_emitted_path(self, pot_type, model, dispersion):
        """Whatever naming emits, the manifest must be able to read back."""
        path = output_dir_for({
            'general': {'temp': 300},
            '2D': {'mat': 'h-MoS2', 'x': 50, 'y': 50, 'lat_c': 6.15,
                   'pot_type': pot_type, 'pot_path': f'models/{model}',
                   'dispersion': dispersion},
        }, 'sheetonsheet')
        assert parse_mlip_flavor(str(path)) == (pot_type, dispersion)

    def test_finds_the_token_in_a_non_leaf_segment(self):
        """AFM nests an L<N> level BELOW the pot_ segment."""
        assert parse_mlip_flavor(f'{AFM}/pot_deepmd_dpa3/L3') == ('deepmd', 'none')

    def test_finds_the_token_behind_a_simulation_root_prefix(self):
        assert parse_mlip_flavor(f'/scratch/campaign/{SHEET}/pot_mace-d3_m') \
            == ('mace', 'd3')

    @pytest.mark.parametrize('path', [
        SHEET,
        AFM,
        f'{SHEET}/pot_sw_MoS',
        'sheetonsheet/h-MoS2/50x_50y/K0',
        '',
    ])
    def test_classical_paths_have_no_flavor(self, path):
        assert parse_mlip_flavor(path) is None

    def test_does_not_match_an_unknown_framework(self):
        """A segment that merely looks token-shaped is not a hit."""
        assert parse_mlip_flavor(f'{SHEET}/pot_tersoff-d3_x') is None


class TestManifestInfersGPUFields:
    """Discovery must populate the fields the job grouping splits on."""

    def test_classical_tree_is_unchanged(self, tmp_path):
        _make_sim(tmp_path, SHEET)
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        for job in manifest.jobs:
            assert job.needs_gpu is False
            assert job.framework is None

    @pytest.mark.parametrize('framework,segment', [
        ('deepmd', 'pot_deepmd_dpa3'),
        ('mace', 'pot_mace-d3_mace-mp'),
    ])
    def test_mlip_tree_sets_gpu_and_framework(self, tmp_path, framework, segment):
        _make_sim(tmp_path, f'{SHEET}/{segment}')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        for job in manifest.jobs:
            assert job.needs_gpu is True
            assert job.framework == framework

    def test_mixed_campaign_splits_into_three_groups(self, tmp_path):
        """A classical + deepmd + mace campaign must yield three distinct keys."""
        _make_sim(tmp_path, SHEET)
        _make_sim(tmp_path, f'{SHEET}/pot_deepmd_dpa3')
        _make_sim(tmp_path, f'{SHEET}/pot_mace-d3_mace-mp')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        keys = {(j.needs_gpu, j.framework) for j in manifest.jobs}
        assert keys == {(False, None), (True, 'deepmd'), (True, 'mace')}

    def test_afm_layer_subdirs_inherit_the_parent_flavor(self, tmp_path):
        _make_sim(tmp_path, f'{AFM}/pot_deepmd_dpa3/L3')
        manifest = JobManifest.from_simulation_directory(tmp_path)
        assert manifest.jobs
        assert all(j.framework == 'deepmd' and j.needs_gpu for j in manifest.jobs)
