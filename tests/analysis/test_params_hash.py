"""The parameters hash of every analysis is part of a stored result's identity:
refactoring a ``Params`` class must not move it."""
import pytest

from frictionsim2d.analysis import store
from frictionsim2d.analysis.base import ANALYSES, get_analysis

_DEFAULTS = {
    'cof_ranking': 'a37eaa101d79481e38fcf7a79c071520741bd9497b94e6038858ce14fdb8b193',
    'contact': 'be62fc1996b5464b50260c1097b25ea873239d3733dc931fee2e28eb406c81c8',
    'integrity_traces': '7b001b13ca31747e6f5c8736602b462362ebe36d55d99f09fc261f503a725306',
    'period_summary': '91c41f7e93cc8118bdc47f0fc846f2f7712546a935e67b217252b22a0c3fb0cb',
    'pes_landscape': '029341b3efe3f6f71139ee267514a8b3a7dc5fe979c00d76740f238d7d0ec4cc',
    'run_status': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    'sheet_geometry': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    'sliding_physics': '8b229b0eb50a34e884f5ed8d5a6d7c28a06c32cc4e6d09075482e409f8c5d488',
    'stick_slip': 'bd265e9ae6a0d1005f391ab110e83e4eb21bc1b7ac1ae3ed85a01db2fbce261b',
}
_GIVEN = {
    'cof_ranking': '6dcf27dd8df7dc3f38c37ef96e3eef6720f4aa149f03ce0098ae6403dc1beabe',
    'contact': 'd2d7ced486e87f6352b7951651bed90499a1351eca6b07d80f3b5ec21c2707d7',
    'stick_slip': '89b128b40d1c7b75bce0e4ac356149f7397580f4ca00504f3890d3e0528bb99f',
    'sliding_physics': '927b7207311926023656714d98cbc97bdd77f86ddb36cdfdefd6288c2c49d3c2',
}


def _hash(name, **given):
    get_analysis(name)
    return store.params_hash(store._params_dict(ANALYSES[name].Params(**given)))  # pylint: disable=protected-access


def test_every_analysis_is_pinned():
    get_analysis('contact')
    assert sorted(ANALYSES) == sorted(_DEFAULTS)


@pytest.mark.parametrize('name', sorted(_DEFAULTS))
def test_default_params_hash_is_stable(name):
    assert _hash(name) == _DEFAULTS[name]


@pytest.mark.parametrize('name', sorted(_GIVEN))
def test_given_params_hash_is_stable(name):
    assert _hash(name, loads='5,10,20', angle=15, lattice='{"h_MoS2": 3.16}') == _GIVEN[name]
