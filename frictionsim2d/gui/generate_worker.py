"""Subprocess entry point for generating a real simulation package.

Separate from the preview worker because the two differ in what they are
allowed to do: this one runs the full sweep, may emit HPC scripts, and may
generate a missing amorphous structure.

Invoked as ``python -m frictionsim2d.gui.generate_worker <request.json>``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


def _emit(kind: str, **payload: Any) -> None:
    print(json.dumps({'kind': kind, **payload}), flush=True)


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    from ..run.artifacts import generate_package
    from ..run.hpc_dispatch import count_generated_jobs, hpc_generation_blocker

    _emit('progress', message=f"generating {request['model']}")

    wants_hpc = bool(request.get('generate_hpc'))

    result = generate_package(
        request['config_file'], request['model'],
        output_root=Path(request['output_root']),
        settings_file=request.get('settings_file'),
        generate_hpc=wants_hpc,
        simulation_root_name=request.get('root_name'),
    )

    payload: Dict[str, Any] = {
        'ok': bool(result.created),
        'created': [str(p) for p in result.created],
        'requested': result.requested,
        'simulation_root': str(result.simulation_root) if result.simulation_root else None,
        'failures': [f.render() for f in result.failures],
    }

    if result.simulation_root and result.created:
        payload['jobs'] = count_generated_jobs(result.simulation_root)

    if wants_hpc:
        # Report what actually happened. HPC generation skips silently when
        # settings.hpc.modules is unset, which otherwise looks like the tick
        # box doing nothing.
        payload['hpc_requested'] = True
        payload['hpc_blocked'] = hpc_generation_blocker(result.settings) \
            if result.settings else None
        hpc_dir = (result.simulation_root / 'hpc') if result.simulation_root else None
        payload['hpc_scripts'] = (
            len([p for p in hpc_dir.iterdir() if p.is_file()])
            if hpc_dir and hpc_dir.is_dir() else 0
        )

    if not result.created:
        payload['error'] = (result.failures[0].render() if result.failures
                            else 'nothing was generated')

    return payload


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        _emit('error', message='no request file given')
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

    Path(request['result_file']).write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    _emit('done', ok=result.get('ok', False))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
