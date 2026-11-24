"""Subprocess entry point for "Run locally" on a quench root.

Runs every member serially with ``lmp`` (hours each), then imports the blocks
into the library -- the same two steps as ``run quench --local``. It lives in a
subprocess so Cancel can stop it: the controller sends SIGTERM, the handler
below kills the ``lmp`` child, and nothing is imported.

Invoked as ``python -m frictionsim2d.gui.quench_worker <request.json>``; the
request names the quench root and the result file.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

from .worker_common import emit as _emit, run_worker


class _ProgressHandler(logging.Handler):
    """Forwards the library's per-member log lines as progress records."""

    def emit(self, record: logging.LogRecord) -> None:
        _emit('progress', message=record.getMessage())


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run the members of ``request['root']`` and import what finished."""
    # pylint: disable-next=import-outside-toplevel
    from ..builders.construction import amorphous

    root = Path(request['root'])
    logger = logging.getLogger(amorphous.__name__)
    logger.setLevel(logging.INFO)
    handler = _ProgressHandler()
    logger.addHandler(handler)
    try:
        done, failed = amorphous.run_quench_root_local(root)
        _emit('progress', message='importing the finished blocks')
        report = amorphous.import_quench_root(root, source='local')
    finally:
        logger.removeHandler(handler)

    def names(items) -> list:
        return [str(Path(item[0] if isinstance(item, tuple) else item).name)
                for item in items]

    problems = [f"{Path(member).parent.name}/{Path(member).name}: {error}"
                for member, error in failed]
    problems += [f"{Path(member).name}: refused: {why}" for member, why in report.refused]
    problems += [f"{Path(member).name}: incomplete: {why}" for member, why in report.incomplete]
    return {
        'ok': not problems,
        'ran': len(done),
        'imported': names(report.imported),
        'unchanged': names(report.unchanged),
        'problems': problems,
        'warnings': list(report.warnings),
    }


def _on_terminate(_signum, _frame) -> None:
    # pylint: disable-next=import-outside-toplevel
    from ..builders.construction.amorphous import stop_active_lmp
    stop_active_lmp()
    sys.exit(1)


def _problems_from_error(result: Dict[str, Any]) -> None:
    result['problems'] = [result['error']]


def main(argv=None) -> int:
    """`python -m frictionsim2d.gui.quench_worker <request.json>`."""
    signal.signal(signal.SIGTERM, _on_terminate)
    return run_worker(run, argv, on_error=_problems_from_error)


if __name__ == '__main__':
    sys.exit(main())
