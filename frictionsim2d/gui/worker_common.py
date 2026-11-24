"""What every ``python -m frictionsim2d.gui.<x>_worker <request.json>`` shares.

The GUI runs builds, plots, store queries and quench runs out of process: a
LAMMPS abort or a hung psycopg2 then kills the worker, not the window, and
Cancel is a kill. Progress goes to stdout as one JSON record per line; the
result is written to ``request['result_file']``.

Qt-free and import-light: workers start in a bare interpreter.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def emit(kind: str, **payload: Any) -> None:
    """Write one progress record to stdout."""
    print(json.dumps({'kind': kind, **payload}), flush=True)


def run_worker(run: Callable[[Dict[str, Any]], Dict[str, Any]], argv=None, *,
               on_error: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
    """Read the request, call ``run``, write its result, emit ``done``.

    An exception from ``run`` becomes ``{'ok': False, 'error', 'traceback'}``,
    which ``on_error`` may extend. Returns the process exit code.
    """
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        emit('error', message='no request file given')
        return 2

    request = json.loads(Path(argv[0]).read_text(encoding='utf-8'))
    try:
        result = run(request)
    except Exception as exc:  # pylint: disable=broad-except
        result = {
            'ok': False,
            'error': f"{type(exc).__name__}: {exc}",
            'traceback': traceback.format_exc(),
        }
        if on_error is not None:
            on_error(result)

    Path(request['result_file']).write_text(
        json.dumps(result, indent=2, default=str), encoding='utf-8')
    emit('done', ok=result.get('ok', False))
    return 0 if result.get('ok') else 1
