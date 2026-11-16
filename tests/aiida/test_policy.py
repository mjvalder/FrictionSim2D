"""The version policy: which verdict and which summary "latest" means
(critique I4). Plain dicts only -- no AiiDA needed."""
import pytest

from src.aiida.policy import Policy, select


def _verdict(uuid, thresholds, code='0.2.0', ctime='2026-01-01', **extra):
    return {'uuid': uuid, 'thresholds_version': thresholds, 'code_version': code,
            'ctime': ctime, 'status': 'complete', **extra}


def test_newest_is_by_semver_not_by_ctime():
    older_ctime_newer_version = _verdict('b', '1.10.0', ctime='2025-01-01')
    newer_ctime_older_version = _verdict('a', '1.2.0', ctime='2026-01-01')
    got = select([newer_ctime_older_version, older_ctime_newer_version], 'verdict',
                 Policy())
    assert got is older_ctime_newer_version


def test_pinned_verdict_version_is_an_exact_match():
    one = _verdict('a', '1.2.0')
    ten = _verdict('b', '1.10.0')
    assert select([one, ten], 'verdict', Policy(verdict='1.2.0')) is one
    assert select([one, ten], 'verdict', Policy(verdict='9.9')) is None


def test_code_version_breaks_a_thresholds_tie_then_uuid():
    old = _verdict('z', '1.0.0', code='0.2.0')
    new = _verdict('a', '1.0.0', code='0.10.0')
    assert select([old, new], 'verdict', Policy()) is new
    left, right = _verdict('a', '1.0.0'), _verdict('b', '1.0.0')
    assert select([right, left], 'verdict', Policy()) is right
    assert select([left, right], 'verdict', Policy()) is right


@pytest.mark.parametrize('order', [0, 1, 2])
def test_ruled_beats_withdrawn_beats_checker(order):
    checker = _verdict('c', '1.0.0', kind='verdict')
    withdrawn = _verdict('a', '1.0.0', kind='withdrawn_verdict', withdrawn_ruling='x')
    ruled = _verdict('b', '1.0.0', kind='ruled_verdict', ruled_by_user=True)
    pool = [checker, withdrawn, ruled]
    pool = pool[order:] + pool[:order]
    assert select(pool, 'verdict', Policy()) is ruled
    assert select([checker, withdrawn], 'verdict', Policy()) is withdrawn


def test_kind_is_inferred_from_content_without_a_kind_key():
    checker = _verdict('z', '1.0.0')
    ruled = _verdict('a', '1.0.0', ruled_by_user=True, original_status='quarantined')
    withdrawn = _verdict('y', '1.0.0', withdrawn_ruling='r')
    assert select([checker, ruled], 'verdict', Policy()) is ruled
    assert select([checker, withdrawn], 'verdict', Policy()) is withdrawn


def test_a_newer_generation_outranks_an_older_ruling():
    ruled_old = _verdict('a', '1.0.0', kind='ruled_verdict')
    checker_new = _verdict('b', '1.1.0', kind='verdict')
    assert select([ruled_old, checker_new], 'verdict', Policy()) is checker_new


def test_summary_selection_is_by_code_version():
    old = {'uuid': 'b', 'code_version': '0.2.0', 'ctime': '2027-01-01'}
    new = {'uuid': 'a', 'code_version': '0.10.0', 'ctime': '2026-01-01'}
    assert select([old, new], 'summary', Policy()) is new
    assert select([old, new], 'summary', Policy(summary='0.2.0')) is old
    assert select([old, new], 'summary', Policy(summary='0.3.0')) is None


def test_empty_candidates_select_nothing():
    assert select([], 'verdict', Policy()) is None
    assert select([], 'summary', Policy()) is None


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match='kind'):
        select([], 'nondim', Policy())


def test_policy_round_trips():
    policy = Policy(verdict='1.2.0')
    assert policy.as_dict() == {'verdict': '1.2.0', 'summary': 'newest'}
    assert Policy.from_dict(policy.as_dict()) == policy
    assert Policy().as_dict() == {'verdict': 'newest', 'summary': 'newest'}
    assert Policy.from_dict({}) == Policy()


def test_policy_is_frozen():
    with pytest.raises(AttributeError):
        Policy().verdict = '1.0.0'  # type: ignore[misc]


def test_summaries_are_filtered_by_options_before_versions():
    old_02 = {'uuid': 'a', 'code_version': '0.2.0', 'options_hash': 'h02'}
    new_03 = {'uuid': 'b', 'code_version': '0.9.0', 'options_hash': 'h03'}
    assert select([old_02, new_03], 'summary', Policy(), options_hash='h02') is old_02
    assert select([old_02, new_03], 'summary', Policy(), options_hash='h03') is new_03
    assert select([old_02, new_03], 'summary', Policy(), options_hash='nope') is None
    assert select([old_02, new_03], 'summary', Policy(summary='0.2.0'),
                  options_hash='h03') is None
    assert select([old_02, new_03], 'summary', Policy()) is new_03


def test_options_hash_is_for_summaries_only():
    with pytest.raises(ValueError, match='summaries'):
        select([], 'verdict', Policy(), options_hash='h')
