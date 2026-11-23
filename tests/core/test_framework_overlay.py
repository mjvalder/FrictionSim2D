"""HPC framework overlay settings and their merge semantics.

Named HPC profiles were dropped in the merge (clusters are switched with
``--settings-file``); what remains is a per-framework overlay on
``HPCSettings`` (``hpc.frameworks``), applied per job group by
``frictionsim2d.core.config.apply_framework_overlay``. The contract pinned here:

* an overlay is **additive** for ``modules``/``env_lines`` (a framework still
  needs the cluster's own modules) and **replacing** for ``binary`` (->
  ``lmp_binary``) and ``conda_env``;
* a classical group (``framework=None``) or a framework with no overlay entry
  gets an unchanged copy -- this is what keeps CPU-only scripts byte-identical;
* the base settings object is never mutated (every job group merges from the
  same base).
"""

from frictionsim2d.core.config import (
    FrameworkOverlay,
    GPUSettings,
    HPCSettings,
    apply_framework_overlay,
)


def _base() -> HPCSettings:
    """Base settings with a GPU block and two framework overlays."""
    return HPCSettings(
        queue='gpu72',
        num_cpus=8,
        memory_gb=64,
        walltime_hours=20,
        modules=['tools/prod', 'lammps'],
        mpi_command='mpirun',
        gpu=GPUSettings(ngpus=1, gpu_type='L40S'),
        env_lines=['export BASE=1'],
        frameworks={
            'deepmd': FrameworkOverlay(
                binary='$HOME/deepmd-kit/bin/lmp',
                modules=['cuda/12.4'],
                env_lines=['export DP_INTRA_OP_PARALLELISM_THREADS=1'],
            ),
            'mace': FrameworkOverlay(conda_env='/opt/envs/mace'),
        },
    )


def test_framework_overlay_appends_modules_and_env_lines_and_sets_binary():
    merged = apply_framework_overlay(_base(), 'deepmd')

    assert merged.lmp_binary == '$HOME/deepmd-kit/bin/lmp'
    assert merged.modules == ['tools/prod', 'lammps', 'cuda/12.4']
    assert merged.env_lines == ['export BASE=1',
                                'export DP_INTRA_OP_PARALLELISM_THREADS=1']
    # Fields the overlay cannot touch keep the base values.
    assert merged.queue == 'gpu72'
    assert merged.num_cpus == 8
    assert merged.gpu == GPUSettings(ngpus=1, gpu_type='L40S')


def test_framework_overlay_conda_env_replaces_base_conda_env():
    base = _base()
    base.conda_env = '/opt/envs/cpu'
    merged = apply_framework_overlay(base, 'mace')

    assert merged.conda_env == '/opt/envs/mace'
    # An overlay without binary/modules leaves them alone.
    assert merged.lmp_binary is None
    assert merged.modules == ['tools/prod', 'lammps']


def test_framework_overlay_does_not_mutate_the_base_settings():
    base = _base()
    apply_framework_overlay(base, 'deepmd')

    assert base.modules == ['tools/prod', 'lammps']
    assert base.env_lines == ['export BASE=1']
    assert base.lmp_binary is None


def test_framework_overlay_returns_a_copy_not_the_base_object():
    base = _base()
    merged = apply_framework_overlay(base, 'deepmd')
    merged.modules.append('extra')

    assert 'extra' not in base.modules


def test_classical_group_gets_an_unchanged_copy():
    base = _base()

    assert apply_framework_overlay(base, None) == base
    assert apply_framework_overlay(base, '') == base


def test_framework_without_an_overlay_entry_is_a_no_op_not_an_error():
    """A classical pot_type (or an MLIP with no overlay) is normal."""
    base = _base()
    merged = apply_framework_overlay(base, 'sw')

    assert merged == base
    assert merged.lmp_binary is None


def test_hpc_settings_defaults_are_backwards_compatible():
    """New fields default to 'nothing configured' so untouched settings.yaml
    files render exactly today's scripts."""
    hpc = HPCSettings()

    assert hpc.env_lines == []
    assert hpc.gpu is None
    assert hpc.lmp_binary is None
    assert hpc.frameworks == {}
    assert not hasattr(hpc, 'profiles')
    assert not hasattr(hpc, 'active_profile')


def test_frameworks_parse_from_plain_dicts():
    """settings.yaml delivers nested plain dicts, not model instances."""
    hpc = HPCSettings(
        gpu={'ngpus': 2, 'gpu_type': 'L40S'},
        frameworks={'mace': {'binary': '/opt/lmp-mace'}},
    )

    assert isinstance(hpc.gpu, GPUSettings)
    assert hpc.gpu.ngpus == 2
    assert isinstance(hpc.frameworks['mace'], FrameworkOverlay)
    assert hpc.frameworks['mace'].binary == '/opt/lmp-mace'
    assert hpc.frameworks['mace'].modules == []
    assert hpc.frameworks['mace'].env_lines == []


def test_gpu_settings_gpu_type_is_optional():
    assert GPUSettings().ngpus == 1
    assert GPUSettings().gpu_type is None
