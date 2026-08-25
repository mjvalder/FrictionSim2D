"""Subprocess entry point for a preview build.

Builds run out-of-process for two reasons that a thread cannot address:
component construction calls LAMMPS in-process with output suppressed, so there
is no interruption point to cancel at and no progress to report; and a
C-level LAMMPS error aborts the whole process, which in a thread would take the
application down with it.

Invoked as ``python -m src.gui.preview_worker <request.json>``. Progress lines
go to stdout as JSON; the result is written to the requested output file.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


def _emit(kind: str, **payload: Any) -> None:
    """Write one progress record to stdout."""
    print(json.dumps({'kind': kind, **payload}), flush=True)


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single-member package for preview."""
    from ..run.artifacts import discover_preview_target, generate_package

    config_file = request['config_file']
    model = request['model']
    output_root = Path(request['output_root'])
    layer = request.get('layer')

    _emit('progress', message=f"building {model}")

    result = generate_package(
        config_file, model,
        output_root=output_root,
        settings_file=request.get('settings_file'),
        generate_hpc=False,
        simulation_root_name='preview',
        # A preview must never start a melt-quench: it takes minutes and writes
        # into the packaged materials directory.
        allow_generate_amorphous=False,
    )

    if not result.created:
        return {
            'ok': False,
            'error': (result.failures[0].render() if result.failures
                      else 'nothing was generated'),
        }

    target = discover_preview_target(result.simulation_root, result.created[0], layer)
    if target is None:
        return {'ok': False, 'error': 'no assembly script was produced'}

    _emit('progress', message='parsing generated system')

    return {
        'ok': True,
        'simulation_root': str(target.simulation_root),
        'run_dir': str(target.run_dir),
        'init_script': str(target.init_script),
        'settings_file': str(target.settings_file) if target.settings_file else None,
        'layer': target.layer,
        'warnings': [f.render() for f in result.failures],
    }


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        _emit('error', message='no request file given')
        return 2

    request_path = Path(argv[0])
    request = json.loads(request_path.read_text(encoding='utf-8'))

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
