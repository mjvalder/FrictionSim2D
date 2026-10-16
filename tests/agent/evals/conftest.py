"""Results plugin for the eval tier.

Each eval test leaves an ``('agent_eval', {...})`` user property (case,
repeat, latency, requests, tokens). At session end the outcomes are written
to ``<AGENT_EVAL_RESULTS_DIR or tests/agent/evals/.results>/<provider>-<model>-
<timestamp>.json`` and a summary table is printed. Without eval runs (the
default: the tier is deselected) it does nothing.
"""
from datetime import datetime

from tests.helpers.agent_eval import (eval_env, format_table, model_name, results_path,
                                      summarise, write_results)

_RECORDS = []
_STATE = {}


def pytest_runtest_logreport(report):
    """Keep the outcome of every eval run (call phase, or a skip in setup)."""
    if report.when != 'call' and not (report.when == 'setup' and report.skipped):
        return
    props = dict(report.user_properties).get('agent_eval')
    if props is None:
        return
    message = ''
    if report.skipped and isinstance(report.longrepr, tuple):
        message = str(report.longrepr[2])
    _RECORDS.append({**props, 'outcome': report.outcome, 'message': message})


def pytest_sessionfinish(session):  # pylint: disable=unused-argument
    """Write the results file."""
    if not _RECORDS:
        return
    env = eval_env()
    model = model_name(env)
    summary = summarise(_RECORDS)
    path = results_path(env.results_dir, env.provider, model, datetime.now())
    write_results(path, env.provider, model, _RECORDS, summary)
    _STATE.update(summary=summary, provider=env.provider, model=model, path=path)


def pytest_terminal_summary(terminalreporter):
    """Print the summary table."""
    if not _STATE:
        return
    terminalreporter.section('agent eval')
    terminalreporter.write_line(format_table(_STATE['summary'], _STATE['provider'],
                                             _STATE['model']))
    terminalreporter.write_line(f"results: {_STATE['path']}")
