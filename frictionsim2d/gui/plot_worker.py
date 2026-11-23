"""Subprocess entry point for rendering plots.

Out of process for the same reason previews are (see
:mod:`frictionsim2d.gui.preview_worker`): matplotlib installs its own backend and its
own font cache, and the GUI already runs ovito's Qt backend. Keeping the two
in one process is how you get a second QApplication and a hang with nothing
to click.

Invoked as ``python -m frictionsim2d.gui.plot_worker <request.json>``. Progress lines
go to stdout as JSON; the result is written to the requested output file.

No Qt import anywhere in this module: it must run in a bare interpreter.
"""

from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _emit(kind: str, **payload: Any) -> None:
    """Write one progress record to stdout."""
    print(json.dumps({'kind': kind, **payload}), flush=True)


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Merge `overlay` into `base`; a list REPLACES, never extends."""
    for key, value in overlay.items():
        if (key in base and isinstance(base[key], dict)
                and isinstance(value, dict)):
            _merge(base[key], value)
        else:
            base[key] = value


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Render the requested plots and describe what was written.

    Raises on a document that does not validate or a dataset directory with
    no export -- those are failures of the whole batch, and `main` turns them
    into an `ok: false` result. A single plot that raises, or renders nothing,
    is a warning instead: the other nine figures are still worth having.
    """
    from ..postprocessing.plot_config import load_plot_document  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.plot_data import Plotter  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.plot_style import DEFAULT_SETTINGS  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.presets import load_preset  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    document = load_plot_document(Path(request['document_file']))

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if request.get('preset'):
        _merge(settings, load_preset(request['preset']))
    if request.get('settings'):
        _merge(settings, request['settings'])

    output_dir = Path(request['output_dir'])
    plotter = Plotter(
        document.data_dirs, document.labels, str(output_dir), settings,
        dataset_display_labels=document.dataset_display_labels,
        series_color_map=document.series_color_map)

    indices = request.get('plot_indices')
    wanted = (list(range(len(document.plots))) if indices is None
              else [int(i) for i in indices])

    figures: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for position, index in enumerate(wanted, start=1):
        spec = document.plots[index]
        _emit('progress',
              message=f'plot {position}/{len(wanted)}: {spec.plot_type}')
        before = len(plotter.written)
        try:
            plotter.generate_plot(spec)
        except Exception as exc:  # pylint: disable=broad-except
            # One bad plot must not lose the other nine: the GUI shows the
            # warning beside the figures that did render.
            warnings.append(f'plot {position}: {type(exc).__name__}: {exc}')
            continue
        written = [str(p) for p in plotter.written[before:]]
        if written:
            figures.append({'index': index, 'plot_type': spec.plot_type,
                            'files': written})
        else:
            warnings.append(f'plot {position}: produced no file')

    return {'ok': True, 'figures': figures, 'warnings': warnings}


def main(argv=None) -> int:
    """`python -m frictionsim2d.gui.plot_worker <request.json>`."""
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
