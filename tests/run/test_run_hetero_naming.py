"""Tests for primary_sheet/material_token and pair-aware output-dir naming.

These back Task 1 of the gui-p2d heterostructure backend: output directories
must distinguish two hetero runs that differ only in their second material,
and the naming/counting code that currently hard-keys '2D' must fall back to
the first numbered '2D-N' section for numbered-only configs (legal because
the pydantic fold mirrors 2D-1 onto '2D', but the naming code runs on the raw
dict, before any model is built).
"""

from pathlib import Path

import pytest

from src.run.naming import material_token, output_dir_for, primary_sheet


def test_primary_sheet_falls_back_to_numbered_sections():
    assert primary_sheet({'2D': {'mat': 'a'}})['mat'] == 'a'
    assert primary_sheet({'2D-2': {'mat': 'c'}, '2D-1': {'mat': 'b'}})['mat'] == 'b'
    assert primary_sheet({'2D-10': {'mat': 'x'}, '2D-2': {'mat': 'y'}})['mat'] == 'y'


def test_primary_sheet_raises_key_error_when_absent():
    with pytest.raises(KeyError):
        primary_sheet({'general': {'temp': 300}})


def test_material_token_joins_pairs_in_order():
    assert material_token({'2D': {'mat': 'h-MoS2'}}) == 'h-MoS2'
    assert material_token({'2D-1': {'mat': 'h-MoS2'}, '2D-2': {'mat': 'h-WS2'}}) == 'h-MoS2+h-WS2'


def test_output_dir_for_single_material_is_byte_identical_to_today():
    """Pins the exact directory name for a plain [2D] config -- the entire
    installed base -- against a real, previously-observed run_simulations
    output (see test_output_dir_for_matches_run_layout in test_run.py)."""
    run_dict = {
        '2D': {'mat': 'h_MoS2', 'x': 100, 'y': 100},
        'general': {'temp': 300},
        'tip': {'mat': 'Si', 'amorph': 'a', 'r': 25},
        'sub': {'mat': 'Si', 'amorph': 'a'},
    }

    path = output_dir_for(run_dict, 'afm')

    assert str(path) == 'afm/h_MoS2/100x_100y/sub_aSi_tip_aSi_r25/K300'


def test_output_dir_for_hetero_distinguishes_second_material():
    """Two hetero runs differing only in material 2 must not collide."""
    base = {
        '2D-1': {'mat': 'h-MoS2', 'x': 100, 'y': 100},
        'general': {'temp': 300},
    }
    run_a = {**base, '2D-2': {'mat': 'h-WS2', 'x': 100, 'y': 100}}
    run_b = {**base, '2D-2': {'mat': 'h-WSe2', 'x': 100, 'y': 100}}

    path_a = output_dir_for(run_a, 'sheetonsheet')
    path_b = output_dir_for(run_b, 'sheetonsheet')

    assert path_a != path_b
    assert str(path_a) == 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300'
    assert str(path_b) == 'sheetonsheet/h-MoS2+h-WSe2/100x_100y/K300'


def test_output_dir_for_numbered_only_uses_first_sheet_for_size():
    """x/y for the size token come from the first numbered sheet when there
    is no plain '2D' section."""
    run_dict = {
        '2D-1': {'mat': 'h-MoS2', 'x': 50, 'y': 60},
        '2D-2': {'mat': 'h-WS2', 'x': 999, 'y': 999},
        'general': {'temp': 300},
    }

    path = output_dir_for(run_dict, 'sheetonsheet')

    assert str(path) == 'sheetonsheet/h-MoS2+h-WS2/50x_60y/K300'


class TestShippedPairSweepExample:
    """`examples/hetero_pair_sweep.ini` is documented as runnable.

    A shipped example that does not run is worse than none, and the ways it
    rots are silent: a renamed CIF, a pair that stops matching, a placeholder
    that no longer substitutes. These check everything a build would consume
    without paying for the build.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]

    @pytest.fixture(name='expanded')
    def _expanded(self):
        from src.run.expand import expand_config_sweeps
        from src.core.config import parse_config

        root = self.REPO_ROOT
        return expand_config_sweeps(
            parse_config(root / 'examples' / 'hetero_pair_sweep.ini'), root)

    def test_it_expands_to_the_documented_runs(self, expanded):
        assert [str(output_dir_for(c, 'sheetonsheet')) for c in expanded] == [
            'sheetonsheet/h-MoS2+h-WS2/12x_12y/K300/pot_sw_h-MoS2',
            'sheetonsheet/h-MoSe2+h-WSe2/12x_12y/K300/pot_sw_h-MoSe2',
            'sheetonsheet/h-MoTe2+h-WTe2/12x_12y/K300/pot_sw_h-MoTe2',
        ]

    def test_each_member_lands_in_its_own_numbered_section(self, expanded):
        assert [(c['2D-1']['mat'], c['2D-2']['mat']) for c in expanded] == [
            ('h-MoS2', 'h-WS2'),
            ('h-MoSe2', 'h-WSe2'),
            ('h-MoTe2', 'h-WTe2'),
        ]

    def test_every_referenced_structure_and_potential_exists(self, expanded):
        for config in expanded:
            for section in ('2D-1', '2D-2'):
                for key in ('cif_path', 'pot_path'):
                    path = self.REPO_ROOT / config[section][key]
                    assert path.is_file(), f"{section}.{key} -> {path}"

    def test_no_placeholder_survives_expansion(self, expanded):
        for config in expanded:
            for section in ('2D-1', '2D-2'):
                for key, value in config[section].items():
                    assert '{mat' not in str(value), f"{section}.{key} = {value}"

    def test_every_pair_is_commensurate(self, expanded, tmp_path, monkeypatch):
        """The failure that resolving files does NOT catch.

        The first draft of this example paired h-MoS2 with h-WSe2, whose ~4%
        mismatch exceeds the default 2% strain_tol: every file resolved, and
        two of the three runs still failed to build. Only running the matcher
        catches that, so it is worth the few seconds.
        """
        from src.builders.construction.components import build_monolayer
        from src.builders.construction.hetero_stack import build_matched_supercells
        from src.core.config import SheetConfig

        monkeypatch.chdir(self.REPO_ROOT)
        for config in expanded:
            monos = [
                Path(build_monolayer(SheetConfig(**config[section]))[0])
                for section in ('2D-1', '2D-2')
            ]
            pair = f"{config['2D-1']['mat']}+{config['2D-2']['mat']}"
            build_matched_supercells(monos[0], monos[1], tmp_path / pair)
