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

from src.core.run import material_token, output_dir_for, primary_sheet


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
