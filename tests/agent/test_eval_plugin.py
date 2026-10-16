"""The eval tier's results plugin: aggregation, the results file and the table.

The end-to-end test runs two eval cases in a subprocess against a scripted
FunctionModel (``AGENT_EVAL_PROVIDER=function``): no network, no real provider.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests.helpers.agent_eval import (FUNCTION_PROVIDER, EvalEnv, build_session, eval_env,
                                      format_table, percentile, results_path, summarise,
                                      warm_up)
from tests.helpers.agent import make_deps

REPO = Path(__file__).resolve().parents[2]
FUNCTION_REPLY_SNIPPET = 'which one you mean'


def _rec(case, outcome, latency=1.0, requests=2, tokens=100, repeat=0, group='g',
         cold=False, input_tokens=None, output_tokens=0):
    return {'case': case, 'group': group, 'repeat': repeat, 'outcome': outcome,
            'latency_s': latency, 'cold': cold, 'requests': requests, 'tokens': tokens,
            'input_tokens': tokens if input_tokens is None else input_tokens,
            'output_tokens': output_tokens, 'message': '', 'detail': ''}


def test_percentile_is_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile([], 50) is None


def test_summary_counts_pass_rate_per_case_and_overall_without_skips():
    records = [_rec('a', 'passed', latency=1.0), _rec('a', 'failed', latency=3.0),
               _rec('b', 'passed', latency=2.0, requests=4, tokens=300),
               _rec('b', 'skipped', latency=9.0, requests=1, tokens=1)]
    summary = summarise(records)
    assert summary['cases']['a'] == {'runs': 2, 'passed': 1, 'failed': 1, 'skipped': 0,
                                     'pass_rate': 0.5}
    assert summary['cases']['b']['pass_rate'] == 1.0 and summary['cases']['b']['skipped'] == 1
    assert summary['overall']['pass_rate'] == pytest.approx(2 / 3)
    assert summary['latency_p50_s'] == 2.0 and summary['latency_p95_s'] == 3.0
    assert summary['mean_requests'] == pytest.approx(8 / 3)
    assert summary['mean_tokens'] == pytest.approx(500 / 3)


def test_all_skipped_case_has_no_pass_rate():
    summary = summarise([_rec('a', 'skipped')])
    assert summary['cases']['a']['pass_rate'] is None
    assert summary['overall']['pass_rate'] is None and summary['latency_p50_s'] is None


def test_results_path_is_safe_for_model_names(tmp_path):
    path = results_path(tmp_path, 'ollama', 'qwen3:4b', datetime(2026, 10, 16, 12, 0, 0))
    assert path == tmp_path / 'ollama-qwen3_4b-20261016T120000.json'


def test_table_lists_every_case_and_the_totals():
    summary = summarise([_rec('alpha', 'passed'), _rec('beta', 'failed')])
    table = format_table(summary, 'gemini', 'm1')
    for text in ('alpha', 'beta', 'overall', '100%', '0%', '50%', 'p50', 'p95',
                 'gemini', 'm1'):
        assert text in table


def test_env_defaults(monkeypatch):
    for name in ('AGENT_EVAL_PROVIDER', 'AGENT_EVAL_MODEL', 'AGENT_EVAL_REPEATS',
                 'AGENT_EVAL_PAUSE', 'AGENT_EVAL_RESULTS_DIR'):
        monkeypatch.delenv(name, raising=False)
    env = eval_env()
    assert (env.provider, env.model, env.repeats, env.pause) == ('gemini', None, 3, 0.0)
    assert env.results_dir == REPO / 'tests' / 'agent' / 'evals' / '.results'


def test_function_provider_run_writes_results_and_prints_table(tmp_path):
    pytest.importorskip('pydantic_ai')
    env = {**os.environ, 'AGENT_EVAL_PROVIDER': 'function', 'AGENT_EVAL_REPEATS': '2',
           'AGENT_EVAL_RESULTS_DIR': str(tmp_path)}
    env.pop('AGENT_EVAL_MODEL', None)
    # The scripted model only ever asks back: the must-ask case passes, the
    # setup case fails -- both outcomes must reach the file and the table.
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/agent/evals', '-m', 'agent_eval',
         '-k', 'unknown_material or afm_single', '-p', 'no:cacheprovider', '-q'],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300, check=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out  # afm_single fails against an ask-back model
    files = list(tmp_path.glob('function-function-*.json'))
    assert len(files) == 1, out
    data = json.loads(files[0].read_text())
    assert (data['provider'], data['model']) == ('function', 'function')
    cases = data['summary']['cases']
    assert cases['unknown_material']['pass_rate'] == 1.0
    assert cases['afm_single']['pass_rate'] == 0.0
    assert cases['unknown_material']['runs'] == 2
    assert data['summary']['overall']['pass_rate'] == 0.5
    assert data['summary']['mean_requests'] == 1.0
    assert data['summary']['mean_tokens'] > 0
    assert data['summary']['mean_input_tokens'] > 0
    assert data['summary']['mean_output_tokens'] > 0
    records = data['records']
    assert len(records) == 4
    assert [r['cold'] for r in records] == [True, False, False, False]
    assert {r['group'] for r in records} == {'setup', 'must_ask'}
    failed = [r for r in records if r['outcome'] == 'failed']
    assert failed and all(FUNCTION_REPLY_SNIPPET in r['detail'] for r in failed)
    assert all(r['detail'] == '' for r in records if r['outcome'] == 'passed')
    assert 'agent eval: function / function' in out and 'unknown_material' in out


def test_warm_latency_leaves_out_the_cold_turn_and_tokens_split():
    records = [_rec('a', 'passed', latency=60.0, cold=True, input_tokens=900, output_tokens=10),
               _rec('a', 'passed', latency=2.0, input_tokens=500, output_tokens=30),
               _rec('b', 'failed', latency=4.0, input_tokens=700, output_tokens=50)]
    summary = summarise(records)
    assert summary['latency_p50_warm_s'] == 2.0 and summary['latency_p95_warm_s'] == 4.0
    assert summary['latency_p95_s'] == 60.0
    assert summary['mean_input_tokens'] == 700 and summary['mean_output_tokens'] == 30


def test_group_pass_rates_and_skip_warning_in_the_table():
    records = [_rec('a', 'passed', group='setup'), _rec('b', 'failed', group='setup'),
               _rec('c', 'passed', group='analysis'), _rec('d', 'skipped', group='analysis')]
    summary = summarise(records)
    assert summary['groups']['setup']['pass_rate'] == 0.5
    assert summary['groups']['analysis']['pass_rate'] == 1.0
    table = format_table(summary, 'gemini', 'm1')
    assert 'setup' in table and 'analysis' in table
    assert 'pass rate not comparable: 1 skipped' in table
    assert 'warm' in table and 'input' in table and 'output' in table
    assert 'not comparable' not in format_table(summarise(records[:3]), 'gemini', 'm1')


def test_warm_up_runs_one_turn_and_leaves_nothing_behind(tmp_path, mock_settings):
    pytest.importorskip('pydantic_ai')
    env = EvalEnv(FUNCTION_PROVIDER, None, 1, 0.0, tmp_path)
    session = build_session(env)
    deps = make_deps(tmp_path, mock_settings)
    assert warm_up(session, deps) is True
    assert session.history == [] and deps.pending == []
    assert warm_up(session, deps) is False  # once per process and provider
