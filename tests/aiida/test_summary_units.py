"""Every number a tracked summary holds carries a unit (``SUMMARY_UNITS``).

No aiida import: this file runs in the fast suite. The families come from
``core.observables.FAMILIES``, the one family vocabulary.
"""
import dataclasses
import numbers

import numpy as np
import pandas as pd
import pytest

from src.aiida.block_schema import (OBSERVABLE_UNIT_ALIASES, SUMMARY_UNITS,
                                    normalise_unit, summary_units)
from src.core.observables import FAMILIES, channel_by_role
from src.data.models import RunSummary, summarise_run
from src.postprocessing.post_channels import AFM_LIKE_FAMILIES

_ALLOWED = {'1', 'nN', 'angstrom', 'eV', 'eV/angstrom^2', 'nN/angstrom',
            'nN*angstrom', 'count', 'samples'}


def _numeric(value) -> bool:
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def test_every_family_has_a_table():
    assert set(SUMMARY_UNITS) == set(FAMILIES)


@pytest.mark.parametrize('family', FAMILIES)
def test_every_numeric_runsummary_field_has_a_unit(family):
    hints = {f.name: f.type for f in dataclasses.fields(RunSummary)}
    numeric = [name for name, hint in hints.items()
               if any(t in str(hint) for t in ('float', 'int'))]
    assert 'mean_cof' in numeric and 'ntimesteps' in numeric
    missing = [name for name in numeric if name not in SUMMARY_UNITS[family]]
    assert missing == []


@pytest.mark.parametrize('family', FAMILIES)
def test_units_are_from_the_vocabulary(family):
    assert set(SUMMARY_UNITS[family].values()) <= _ALLOWED


def test_cof_is_dimensionless_and_forces_are_nn():
    for family in FAMILIES:
        assert SUMMARY_UNITS[family]['mean_cof'] == '1'
        for key in ('mean_nf', 'std_nf', 'mean_lf', 'mean_lfx', 'std_lfy'):
            assert SUMMARY_UNITS[family][key] == 'nN', (family, key)


@pytest.mark.parametrize('family', ['afm', 'indent', 'sheetonsheet', 'hetero', 'flake'])
def test_force_units_agree_with_the_observable_registry(family):
    """The summary's force unit is the unit of the channel it averages: the
    registry records nN for AFM (afm/slide.lmp:115-117) and for the sheet
    family (sheetonsheet/slide.lmp:136-141,172-173 scale by ev_a_to_nn)."""
    for role, key in (('normal_load', 'mean_nf'), ('lateral_force_x', 'mean_lfx'),
                      ('lateral_force_y', 'mean_lfy')):
        assert channel_by_role(family, role).units == SUMMARY_UNITS[family][key]


def _frames(family):
    """Frames that make ``summarise_run`` emit every key it can for ``family``."""
    n = 60
    if family in ('pes_tip', 'pes_sheet'):
        rows = [(i, j, -1.0 + 0.1 * (i + j) + 0.05 * np.sin(i * j))
                for i in range(4) for j in range(4)]
        df = pd.DataFrame(rows, columns=['x', 'y', 'energy_eV'])
        if family == 'pes_tip':
            df['fx'], df['fy'], df['fz'] = 0.1, 0.2, -3.0
        return {'window': df}, {}
    window = pd.DataFrame({'time': np.arange(n) * 1000})
    roles = ('normal_load', 'lateral_force_x', 'lateral_force_y',
             'driven_com_x', 'driven_com_y', 'driven_com_z')
    rng = np.random.default_rng(0)
    for role in roles:
        window[channel_by_role(family, role).name] = 5.0 + rng.normal(size=n)
    frames = {'window': window}
    lfx = np.tile([0.0, 1.0, 2.0, 3.0, 0.0], 20)
    frames['raw'] = pd.DataFrame({'time': np.arange(lfx.size) * 100, 'raw_lfx': lfx})
    if family == 'indent':
        z = np.linspace(0.0, 10.0, 30)
        frames['retract'] = pd.DataFrame({'z_tip': z, 'fz_tip': 2.0 - z * 0.6 + 0.04 * z * z})
    post = ({'gap_min': 2.5, 'deflect_max': 1.0} if family in AFM_LIKE_FAMILIES
            else {'gap_min_il': 3.1, 'corrugation_max': 0.4})
    return frames, post


@pytest.mark.parametrize('family', FAMILIES)
def test_every_key_summarise_run_emits_has_a_unit(family):
    """Built from a real ``summarise_run``: a new extras key with no unit
    fails here, not silently in the store."""
    frames, post = _frames(family)
    content = summarise_run(frames, family, post=post).as_dict()
    numeric = {k for k, v in content.items() if _numeric(v)}
    assert len(numeric) >= 9   # not vacuous: the family's own keys are there
    units = summary_units(family, content)
    assert numeric <= set(units)


def test_a_summary_field_without_a_unit_is_refused():
    content = summarise_run(*_frames('afm')[:1], 'afm').as_dict()
    content['new_scalar'] = 1.5
    with pytest.raises(ValueError, match='new_scalar'):
        summary_units('afm', content)


def test_none_and_non_numeric_fields_still_need_no_unit():
    content = {'mean_cof': None, 'time_series_hash': 'ab', 'is_complete': True,
               'pes_eta': None}
    assert summary_units('pes_sheet', content) == {
        'mean_cof': '1', 'pes_eta': '1'}


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match='family'):
        summary_units('nope', {'mean_cof': 0.1})


# --- the two spellings, mapped explicitly (fix rounds 1-2, ruling P2-R14) -------

def _registry_units():
    from src.core.observables import REGISTRY
    return {(fam, ob.name): ob.units for ob in REGISTRY.values()
            for fam in ob.sim_types if ob.units}


def _shared():
    """``(family, summary key, summary unit, registry unit)`` for every
    summary key that IS an observable (the post channels) or a statistic of
    one (mean/std of nf, lfx, lfy, lf)."""
    registry = _registry_units()
    out = []
    for family, table in SUMMARY_UNITS.items():
        for key, unit in table.items():
            name = key.split('_', 1)[1] if key.startswith(('mean_', 'std_')) else key
            if (family, name) in registry and name != 'cof':
                out.append((family, key, unit, registry[(family, name)]))
    return out


def test_shared_quantities_agree_after_normalisation():
    shared = _shared()
    assert len(shared) >= 20
    lengths = [s for s in shared if s[2] == 'angstrom']
    assert lengths and {s[3] for s in lengths} == {'A'}        # the alias is exercised
    for family, key, unit, registry_unit in shared:
        assert normalise_unit(registry_unit) == unit, (family, key)


def test_an_unaliased_mismatch_fails(monkeypatch):
    """Only the alias table bridges the spellings: 'Angstrom' is not 'angstrom'."""
    assert normalise_unit('Angstrom') != 'angstrom'
    monkeypatch.delitem(OBSERVABLE_UNIT_ALIASES, 'A')
    assert any(normalise_unit(r) != u for _f, _k, u, r in _shared())


def test_every_registry_unit_normalises_into_the_schema_vocabulary():
    """A registry unit is either spelled as block_schema spells it or is in
    the alias table; block_schema itself never uses the registry spelling."""
    from src.aiida.block_schema import UNITS
    schema = set(UNITS.values()) | {u for t in SUMMARY_UNITS.values() for u in t.values()}
    assert not schema & set(OBSERVABLE_UNIT_ALIASES)
    assert set(OBSERVABLE_UNIT_ALIASES.values()) <= schema | {'eV/angstrom'}
    known = schema | {'eV/angstrom', 'step'}
    assert {normalise_unit(u) for u in _registry_units().values()} <= known
