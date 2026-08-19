"""QC classifier tests (spec §7, Task 10)."""
from pathlib import Path

import pandas as pd
import pytest

from src.postprocessing.qc import QC_VERSION, classify

FIX = Path(__file__).parent / 'fixtures/qc'

CASES = [('converged.csv', 'converged'), ('seized.csv', 'seized'),
         ('detached.csv', 'detached'), ('punched.csv', 'punched_through'),
         ('drifted.csv', 'drifted')]

PROV = {'load_requested': 30.0, 'ramp_converged': True}


@pytest.mark.parametrize('fname,expected', CASES)
def test_fixture_classification(fname, expected):
    df = pd.read_csv(FIX / fname)
    assert classify(df, PROV).status == expected


def test_idempotent():
    df = pd.read_csv(FIX / 'converged.csv')
    assert classify(df, PROV) == classify(df, PROV)


def test_version_recorded():
    df = pd.read_csv(FIX / 'converged.csv')
    assert classify(df, PROV).version == QC_VERSION


def test_com_fallback_flagged_in_reasons():
    # without tip_min_z the punch-through check runs on the (weaker) tip COM
    df = pd.read_csv(FIX / 'converged.csv')
    res = classify(df, PROV)
    assert any('tip COM' in r for r in res.reasons)


def test_unconverged_ramp_is_diagnostic_not_status():
    df = pd.read_csv(FIX / 'converged.csv')
    res = classify(df, {'load_requested': 30.0, 'ramp_converged': False})
    assert res.status == 'converged'
    assert any('ramp unconverged' in r for r in res.reasons)
