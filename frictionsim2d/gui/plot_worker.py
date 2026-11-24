"""Subprocess entry point for rendering plots (see :mod:`.worker_common`).

Matplotlib must not share a process with ovito's Qt backend. No Qt import here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

from .worker_common import emit as _emit, run_worker


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Render the requested plots and describe what was written.

    Raises on a document that does not validate or a dataset directory with
    no export -- those are failures of the whole batch, and `main` turns them
    into an `ok: false` result. A single plot that raises, or renders nothing,
    is a warning instead: the other nine figures are still worth having.
    """
    from ..postprocessing.plot_config import load_plot_document  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.plot_data import Plotter  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.plot_style import DEFAULT_SETTINGS, merge_settings  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from ..postprocessing.presets import load_preset  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    document = load_plot_document(Path(request['document_file']))

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if request.get('preset'):
        merge_settings(settings, load_preset(request['preset']))
    if request.get('settings'):
        merge_settings(settings, request['settings'])

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
    return run_worker(run, argv)


if __name__ == '__main__':
    sys.exit(main())
