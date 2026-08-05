"""'How many runs will this generate' is three different numbers.

configs (separate output directories), builds (L{n} trees), and jobs (run
scripts / HPC array entries) have all been mistaken for each other. The counts
are predictions made before anything is generated, so the important tests here
are the ones that generate a real tree and compare.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import GeneralConfig, parse_config
from src.core.config_io import write_config
from src.core.run import (
    count_expansion,
    count_generated_jobs,
    loopability,
    run_simulations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'

#: (config, model, expected configs, builds, jobs) -- measured from real trees.
CASES = [
    ('indent_config.ini', 'indent', 1, 2, 10),
    ('afm_flake_config.ini', 'afm', 1, 1, 1),
    ('pes_tip_config.ini', 'pes_tip', 1, 1, 1),
    ('pes_sheet_config.ini', 'pes_sheet', 1, 1, 2),
]


@pytest.mark.parametrize('name,model,configs,builds,jobs', CASES,
                         ids=[c[0] for c in CASES])
def test_predicted_counts(name, model, configs, builds, jobs, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    summary = count_expansion(parse_config(EXAMPLES / name), model)
    assert (summary.configs, summary.builds, summary.jobs) == (configs, builds, jobs), \
        summary.render()


@pytest.mark.parametrize('name,model,configs,builds,jobs', CASES,
                         ids=[c[0] for c in CASES])
def test_prediction_matches_a_real_generated_tree(name, model, configs, builds, jobs,
                                                  tmp_path, monkeypatch):
    """The whole point: the number shown before generating must be the number
    generated. If a builder changes how it splits the run phase, this fails."""
    monkeypatch.chdir(REPO_ROOT)
    config_path = EXAMPLES / name

    predicted = count_expansion(parse_config(config_path), model)
    created, simulation_root, _, _ = run_simulations(
        config_file=str(config_path),
        model=model,
        settings_file=None,
        output_root=tmp_path,
        generate_hpc=False,
    )

    assert len(created) == predicted.configs
    actual_builds = len([p for p in simulation_root.rglob('lammps') if p.is_dir()])
    assert actual_builds == predicted.builds
    assert count_generated_jobs(simulation_root) == predicted.jobs


class TestLoopability:
    """A list is only a sweep when it changes the output directory name."""

    @pytest.mark.parametrize('section,key,expected', [
        ('2D', 'materials_list', 'sweep'),
        ('general', 'temp', 'sweep'),
        ('2D', 'layers', 'layer'),
        ('general', 'force', 'in_file'),
        ('general', 'scan_angle', 'in_file'),
        ('general', 'scan_angle_force', 'in_file'),
        ('general', 'pressure', 'in_file'),
        ('general', 'scan_speed', 'in_file'),
        ('2D', 'x', 'scalar'),
        ('tip', 'r', 'scalar'),
        ('sub', 'thickness', 'scalar'),
    ])
    def test_classification(self, section, key, expected):
        assert loopability(section, key) == expected

    def test_unknown_keys_default_to_scalar(self):
        assert loopability('general', 'not_a_real_key') == 'scalar'


class TestTemperatureSweep:
    """temp is the one [general] key that is genuinely sweep-safe: it appears in
    the directory name as K{temp}."""

    def test_temp_list_multiplies_configs(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'indent_config.ini')
        raw['general']['temp'] = [300, 500]
        summary = count_expansion(raw, 'indent')
        assert summary.configs == 2
        assert summary.builds == 4
        assert summary.jobs == 20

    def test_temp_sweep_produces_distinct_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'indent_config.ini')
        raw['general']['temp'] = [300, 500]
        # run_simulations takes a path, not a mapping.
        config_path = write_config(raw, tmp_path / 'temp_sweep.ini')

        created, _, _, _ = run_simulations(
            config_file=str(config_path), model='indent', settings_file=None,
            output_root=tmp_path / 'out', generate_hpc=False,
        )
        assert len(created) == 2
        # Distinct K{temp} directories: the sweep must not overwrite itself.
        assert len({p.name for p in created}) == 2


class TestInFileLoopsDoNotMultiplyConfigs:

    def test_force_list_does_not_add_configs(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'afm_flake_config.ini')
        raw['general']['force'] = [5, 10, 20, 30]
        assert count_expansion(raw, 'afm').configs == 1

    def test_outer_loop_force_multiplies_jobs_only(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'afm_flake_config.ini')
        raw['general']['force'] = [5, 10, 20, 30]
        raw['general']['outer_loop'] = 'force'
        summary = count_expansion(raw, 'afm')
        assert (summary.configs, summary.builds, summary.jobs) == (1, 1, 4)

    def test_outer_loop_scan_angle_multiplies_jobs_only(self, monkeypatch):
        """Pins two things at once, deliberately routed through the validated
        model rather than a raw dict:

        1. outer_loop='scan_angle' must pass GeneralConfig validation -- the
           GeneralConfig(**raw['general']) call below raises ValidationError
           if the Literal in config.py is ever reverted to its pre-Task-1
           three values, so this test actually depends on the change made in
           this task (a raw-dict-only version does not: count_expansion ->
           expand_config_sweeps -> _run_scripts_per_build never construct a
           GeneralConfig, so a raw dict sails through regardless of what the
           Literal allows).
        2. Once validated, _run_scripts_per_build's generic
           general.get(outer_loop) handling (no per-value branching) counts
           one job per angle -- the same generic mechanism already pinned for
           outer_loop='force' above, now exercised for 'scan_angle'. This is
           the number Task 5's GUI expansion count depends on.

        No builder splits the slide phase by angle yet (that lands in Tasks
        2-3); this test only pins the *counting* layer, which already handles
        it because it has no per-value branching to update.
        """
        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'afm_flake_config.ini')
        raw['general']['scan_angle'] = [0, 45, 90]
        raw['general']['outer_loop'] = 'scan_angle'

        validated_general = GeneralConfig(**raw['general'])
        raw['general'] = validated_general.model_dump()

        summary = count_expansion(raw, 'afm')
        assert (summary.configs, summary.builds, summary.jobs) == (1, 1, 3)
