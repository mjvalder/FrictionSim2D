"""HPC profile settings models and profile merge semantics (plan Task 15).

Lives in its own module rather than ``tests/test_config.py`` (where the plan
placed it) purely to keep the file uncontended while other config work is in
flight; the subject under test is still ``src.core.config``.

The contract these tests pin down:

* a profile is a **partial** override — it changes only the fields it actually
  sets, and every unset field keeps the base value;
* asking for a profile that does not exist is a loud ``ValueError``, not a
  silent fall-through to the base settings (a typo'd ``--profile`` must not
  quietly submit a CPU job to the wrong queue);
* no profile means *today's* settings object, unchanged — this is what makes
  the byte-identical script regression in ``test_hpc_scripts.py`` meaningful.
"""

import pytest

from src.core.config import (
    FrameworkOverlay,
    GPUSettings,
    HPCProfile,
    HPCSettings,
    resolve_profile,
)


def _base() -> HPCSettings:
    """Base settings with every field a profile might override set explicitly."""
    return HPCSettings(
        queue='v1_small24',
        num_cpus=32,
        memory_gb=62,
        walltime_hours=20,
        modules=['tools/prod', 'lammps'],
        mpi_command='mpirun',
        profiles={
            'cx3': HPCProfile(
                queue='gpu72',
                gpu=GPUSettings(ngpus=1, gpu_type='L40S'),
                num_cpus=8,
                memory_gb=64,
                frameworks={
                    'deepmd': FrameworkOverlay(
                        binary='$HOME/deepmd-kit/bin/lmp',
                        modules=['cuda/12.4'],
                        env_lines=['export DP_INTRA_OP_PARALLELISM_THREADS=1'],
                    ),
                },
            ),
            'hx1': HPCProfile(gpu=GPUSettings(ngpus=1, gpu_type='A100')),
        },
    )


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------

def test_profile_overrides_only_the_fields_it_sets():
    base = _base()
    merged = resolve_profile(base, 'cx3')

    # Overridden by the profile.
    assert merged.queue == 'gpu72'
    assert merged.num_cpus == 8
    assert merged.memory_gb == 64
    assert merged.gpu is not None
    assert merged.gpu.ngpus == 1
    assert merged.gpu.gpu_type == 'L40S'

    # NOT set by the profile -> base values survive untouched.
    assert merged.walltime_hours == 20
    assert merged.modules == ['tools/prod', 'lammps']
    assert merged.mpi_command == 'mpirun'
    assert merged.scheduler_type == base.scheduler_type
    assert merged.max_array_size == base.max_array_size


def test_resolve_profile_does_not_mutate_the_base_settings():
    base = _base()
    resolve_profile(base, 'cx3')

    assert base.queue == 'v1_small24'
    assert base.num_cpus == 32
    assert base.gpu is None


def test_partial_profile_leaves_everything_else_alone():
    """``hx1`` sets only ``gpu`` — the queue must stay the base queue."""
    merged = resolve_profile(_base(), 'hx1')

    assert merged.gpu is not None and merged.gpu.gpu_type == 'A100'
    assert merged.queue == 'v1_small24'
    assert merged.num_cpus == 32
    assert merged.memory_gb == 62


# ---------------------------------------------------------------------------
# Framework overlay merges last
# ---------------------------------------------------------------------------

def test_framework_overlay_merges_last():
    merged = resolve_profile(_base(), 'cx3', framework='deepmd')

    assert merged.lmp_binary == '$HOME/deepmd-kit/bin/lmp'
    # Framework modules/env_lines are additive on top of the profile's.
    assert merged.modules == ['tools/prod', 'lammps', 'cuda/12.4']
    assert merged.env_lines == ['export DP_INTRA_OP_PARALLELISM_THREADS=1']
    # Profile-level fields still applied.
    assert merged.queue == 'gpu72'


def test_unknown_framework_is_a_no_op_not_an_error():
    """A classical (non-MLIP) group has no overlay; that is normal."""
    merged = resolve_profile(_base(), 'cx3', framework='sw')

    assert merged.lmp_binary is None
    assert merged.modules == ['tools/prod', 'lammps']


# ---------------------------------------------------------------------------
# Error and identity paths
# ---------------------------------------------------------------------------

def test_unknown_profile_name_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_profile(_base(), 'cx4')

    message = str(excinfo.value)
    assert 'cx4' in message
    assert 'cx3' in message  # the error names the valid choices


def test_no_profile_returns_base_settings_unchanged():
    base = _base()

    assert resolve_profile(base, None) == base
    assert resolve_profile(base, '') == base


def test_hpc_settings_defaults_are_backwards_compatible():
    """New fields must default to 'nothing configured' so untouched
    settings.yaml files render exactly today's scripts."""
    hpc = HPCSettings()

    assert hpc.profiles == {}
    assert hpc.active_profile is None
    assert hpc.env_lines == []
    assert hpc.gpu is None
    assert hpc.lmp_binary is None


def test_profiles_parse_from_plain_dicts():
    """settings.yaml delivers nested plain dicts, not model instances."""
    hpc = HPCSettings(
        active_profile='cx3',
        profiles={
            'cx3': {
                'queue': 'gpu72',
                'gpu': {'ngpus': 2, 'gpu_type': 'L40S'},
                'frameworks': {'mace': {'binary': '/opt/lmp-mace'}},
            }
        },
    )

    profile = hpc.profiles['cx3']
    assert isinstance(profile, HPCProfile)
    assert isinstance(profile.gpu, GPUSettings)
    assert profile.gpu.ngpus == 2
    assert isinstance(profile.frameworks['mace'], FrameworkOverlay)
    assert profile.frameworks['mace'].binary == '/opt/lmp-mace'
    assert profile.frameworks['mace'].modules == []


def test_gpu_settings_gpu_type_is_optional():
    assert GPUSettings().ngpus == 1
    assert GPUSettings().gpu_type is None
