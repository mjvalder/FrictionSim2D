"""Analysis tools: find result sets, summarise one, describe the plot types.

Read-only. Two kinds of result set exist in a workspace:

* a **dataset** -- a folder with `outputs/output_full_*.json` (the rule the
  Plots tab uses, `src.gui.datasets.is_dataset`), which is what plots read;
* a **simulation root** -- a folder whose `<model>/` subtrees hold runs
  (`src.run.tree.discover_runs`), built but maybe not post-processed yet.

A folder can be both: `postprocess read` writes `outputs/` into the root.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...gui.datasets import DATASETS_DIRNAME, dataset_files, is_dataset
from ...postprocessing import plot_config
from ...postprocessing.run_status import STATUSES, expected_chains, read_run_config
from ...run.naming import MODELS
from ...run.tree import discover_runs
from ..changes import ToolInputError
from ..sandbox import display_path, resolve_readable
from . import compact

#: Largest `output_full_*.json` `summarise_results` will parse.
MAX_JSON_MB = 200
#: Largest file `find_result_sets` parses just to count its runs.
COUNT_JSON_MB = 20
MAX_FIND_DEPTH = 4
MAX_FIND_HITS = 200
#: At most this many scalar `post` entries shown per row.
MAX_POST = 3


# ---- finding -----------------------------------------------------------------

def _children(folder: Path) -> List[Path]:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_dir() and not p.name.startswith('.'))
    except OSError:
        return []


def _megabytes(path: Path) -> float:
    try:
        return path.stat().st_size / 1e6
    except OSError:
        return 0.0


def _load(path: Path, max_mb: float) -> Tuple[Optional[dict], str]:
    """The parsed document, or None and why not."""
    size = _megabytes(path)
    if size > max_mb:
        return None, f'{size:.0f} MB, too large (limit {max_mb:g} MB)'
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, 'unreadable'
    return (doc, '') if isinstance(doc, dict) else (None, 'not a results document')


def _size_of(path: Path) -> str:
    return path.stem[len('output_full_'):]


def _dataset_line(deps, folder: Path) -> str:
    files = dataset_files(folder)
    sizes = ', '.join(_size_of(p) for p in files)
    counts = []
    for path in files:
        doc, _ = _load(path, COUNT_JSON_MB)
        if doc is None:
            counts = None
            break
        counts.append(len(doc.get('runs') or []))
    tail = (f'{sum(counts)} runs' if counts is not None
            else f'{sum(_megabytes(p) for p in files):.0f} MB')
    return f'{display_path(folder, deps.roots)} — dataset (sizes: {sizes}; {tail})'


def _root_line(deps, folder: Path) -> Optional[str]:
    runs = discover_runs(folder)
    if not runs:
        return None
    models = Counter(run.tokens.model for run in runs)
    shown = ', '.join(f'{model}: {models[model]} runs' for model in MODELS if model in models)
    return f'{display_path(folder, deps.roots)} — simulation root ({shown})'


def find_result_sets(deps, max_hits: int = MAX_FIND_HITS) -> str:
    """Datasets and simulation roots in the workspace (4 levels deep at most)."""
    root = Path(deps.roots.workspace)
    queue = deque([(root, 0)])
    exported = root / '.preview' / DATASETS_DIRNAME
    if exported.is_dir():
        queue.append((exported, 2))
    lines: List[str] = []
    capped = False
    while queue:
        folder, depth = queue.popleft()
        found = []
        if is_dataset(folder):
            found.append(_dataset_line(deps, folder))
        has_runs = False
        if any((folder / model).is_dir() for model in MODELS):
            line = _root_line(deps, folder)
            if line is not None:
                found.append(line)
                has_runs = True
        for line in found:
            if len(lines) >= max_hits:
                capped = True
                break
            lines.append(line)
        if capped:
            break
        if depth >= MAX_FIND_DEPTH:
            continue
        for child in _children(folder):
            if child.name == 'outputs' or (has_runs and child.name in MODELS):
                continue
            queue.append((child, depth + 1))
    if not lines:
        return ('no datasets or simulation roots in the workspace (searched '
                f'{MAX_FIND_DEPTH} levels deep)')
    if capped:
        lines.append(f'… stopped after {max_hits} result sets')
    return compact('\n'.join(lines))


# ---- summarising -------------------------------------------------------------

def _norm_material(name: Any) -> str:
    return str(name or '').lower().replace('-', '_')


def _norm_size(token: Any) -> str:
    return str(token or '').lower().replace('_', '')


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


def _post(post: Any) -> str:
    if not isinstance(post, dict):
        return ''
    shown = [f'{key}={_num(value)}' for key, value in post.items()
             if isinstance(value, (int, float, str, bool)) and value is not None]
    return ' '.join(shown[:MAX_POST])


def _matches(run: dict, material: Optional[str], size: Optional[str]) -> bool:
    if material:
        names = {_norm_material(run.get('material'))}
        names.update(_norm_material(m) for m in run.get('materials') or [])
        if _norm_material(material) not in names:
            return False
    return not size or _norm_size(run.get('size')) == _norm_size(size)


def _row(run: dict) -> str:
    parts = [str(run.get('material')), str(run.get('size')),
             f"T{_num(run.get('temperature'))}"]
    if run.get('layers') is not None:
        parts.append(f"L{run['layers']}")
    kind = run.get('load_kind')
    if kind and kind != 'none':
        parts.append(f"{kind} {_num(run.get('load'))}")
    parts += [f"angle {_num(run.get('angle'))}", f"v {_num(run.get('speed'))}"]
    post = _post(run.get('post'))
    if post:
        parts.append(post)
    parts.append(str(run.get('status')))
    return ' '.join(parts)


def _loads(meta: dict) -> str:
    out = []
    for kind, by_load in (meta.get('loads_by_kind') or {}).items():
        if by_load:
            loads = ', '.join(_num(float(load)) for load in by_load)
            out.append(f'{kind} {loads}')
    return '; '.join(out)


def _file_header(path: Path, doc: dict) -> List[str]:
    meta = doc.get('metadata') or {}
    counts = meta.get('status_counts') or {}
    status = ', '.join(f'{s} {counts[s]}' for s in STATUSES if counts.get(s))
    lines = [f"{path.name}: {len(doc.get('runs') or [])} runs — {status or 'no status'}"]
    lines.append(f"materials: {', '.join(meta.get('materials') or [])}; "
                 f"T: {', '.join(_num(t) for t in meta.get('temperatures') or [])}; "
                 f"loads: {_loads(meta) or '-'}")
    return lines


def _summarise_dataset(deps, folder: Path, files: List[Path], filters: Dict[str, Any],
                       max_rows: int, max_json_mb: float) -> str:
    lines = [f'{display_path(folder, deps.roots)} — dataset, {len(files)} file(s)']
    rows: List[Tuple[int, str]] = []
    for path in files:
        if filters['size'] and _norm_size(_size_of(path)) != _norm_size(filters['size']):
            continue
        doc, why = _load(path, max_json_mb)
        if doc is None:
            lines.append(f'{path.name}: {why}')
            continue
        lines += _file_header(path, doc)
        for run in doc.get('runs') or []:
            if isinstance(run, dict) and _matches(run, filters['material'], filters['size']):
                status = run.get('status')
                rank = STATUSES.index(status) if status in STATUSES else len(STATUSES)
                rows.append((rank, _row(run)))
    return '\n'.join(lines + _table(rows, filters, max_rows))


def _table(rows: List[Tuple[int, str]], filters: Dict[str, Any], max_rows: int) -> List[str]:
    if not rows:
        wanted = ', '.join(f'{k}={v}' for k, v in filters.items() if v)
        return [f'no runs match {wanted}'] if wanted else ['no runs']
    rows.sort(key=lambda r: r[0])  # stable: complete first, file order kept
    max_rows = max(1, int(max_rows))
    out = [row for _, row in rows[:max_rows]]
    if len(rows) > max_rows:
        out.append(f'… {len(rows) - max_rows} more rows')
    return out


def _summarise_root(deps, folder: Path, runs, filters: Dict[str, Any],
                    max_rows: int) -> str:
    lines = [f'{display_path(folder, deps.roots)} — simulation root, {len(runs)} runs '
             '(not post-processed: run `FrictionSim2D postprocess read` on it for a dataset)']
    rows: List[Tuple[int, str]] = []
    for run in runs:
        tokens = run.tokens
        size = f'{tokens.size_x:g}x{tokens.size_y:g}y'
        fake = {'material': tokens.materials[0] if tokens.materials else None,
                'materials': tokens.materials, 'size': size}
        if not _matches(fake, filters['material'], filters['size']):
            continue
        parts = [tokens.model if tokens.family == tokens.model
                 else f'{tokens.model}/{tokens.family}',
                 '+'.join(tokens.materials), size, f'T{tokens.temperature:g}']
        if tokens.layers is not None:
            parts.append(f'L{tokens.layers}')
        if tokens.pot_token:
            parts.append(f'pot {tokens.pot_token}')
        chains = expected_chains(read_run_config(run), tokens.family)
        expected = ('grid unknown' if chains is None
                    else f'{sum(len(c) for c in chains)} points expected')
        try:
            n_files = sum(1 for p in run.results_dir.iterdir() if p.is_file())
        except OSError:
            n_files = 0
        rows.append((0, f"{' '.join(parts)}: {expected}, {n_files} results files"))
    return '\n'.join(lines + _table(rows, filters, max_rows))


def summarise_results(deps, path: str, material: Optional[str] = None,
                      size: Optional[str] = None, max_rows: int = 30,
                      max_json_mb: float = MAX_JSON_MB) -> str:
    """Status counts, materials, loads and a run table for a dataset or simulation root."""
    target = resolve_readable(path, deps.roots)
    filters = {'material': material, 'size': size}
    if target.is_file() and target.match('output_full_*.json'):
        return compact(_summarise_dataset(deps, target.parent.parent, [target], filters,
                                          max_rows, max_json_mb))
    if is_dataset(target):
        return compact(_summarise_dataset(deps, target, dataset_files(target), filters,
                                          max_rows, max_json_mb))
    runs = discover_runs(target) if target.is_dir() else []
    if runs:
        return compact(_summarise_root(deps, target, runs, filters, max_rows))
    raise ToolInputError(f'{display_path(target, deps.roots)}: neither a dataset nor a '
                         'simulation root; use find_result_sets to list them')


# ---- plots -------------------------------------------------------------------

def _plot_models() -> Dict[str, type]:
    """plot_type -> its model, read off the PlotBase subclasses."""
    models = {}
    for model in plot_config.PlotBase.__subclasses__():
        field = model.model_fields.get('plot_type')
        if field is not None:
            models[field.default] = model
    return models


def plot_types(deps) -> str:  # pylint: disable=unused-argument
    """Every plot type with a one-line description."""
    models = _plot_models()
    lines = []
    for plot_type in plot_config.PLOT_TYPES:
        doc = (models[plot_type].__doc__ or '').strip().splitlines() if plot_type in models \
            else []
        lines.append(f"{plot_type}: {doc[0] if doc else ''}".rstrip())
    return compact('\n'.join(lines))


_ROW = re.compile(r'^\| `([^`]+)` \| `(.*)` \| (.*) \|$')


def _strip_optional(text: str) -> str:
    """``Optional[X]`` -> ``X`` everywhere, brackets balanced."""
    while 'Optional[' in text:
        start = text.index('Optional[')
        depth, end = 0, start + len('Optional')
        for end in range(start + len('Optional'), len(text)):
            depth += {'[': 1, ']': -1}.get(text[end], 0)
            if depth == 0:
                break
        text = text[:start] + text[start + len('Optional['):end] + text[end + 1:]
    return text


def _short_type(annotation: str) -> str:
    text = annotation.replace('src.postprocessing.plot_config.', '')
    text = re.sub(r"<class '(\w+)'>", r'\1', text)
    return _strip_optional(text).replace('List[', 'list[').replace('Dict[', 'dict[')


def _wrapped(label: str, items: List[str], width: int = 110) -> List[str]:
    lines, line = [], label
    for item in items:
        if len(line) + len(item) + 2 > width:
            lines.append(line.rstrip(','))
            line = ' '
        line += f' {item},'
    lines.append(line.rstrip(','))
    return lines


def plot_schema(deps, plot_type: str) -> str:  # pylint: disable=unused-argument
    """The fields one plot type accepts, from the plot schema; its own fields first."""
    if plot_type not in plot_config.PLOT_TYPES:
        raise ToolInputError(f'unknown plot type {plot_type!r}; one of: '
                             + ', '.join(plot_config.PLOT_TYPES))
    section = plot_config.schema_markdown().split(f'## `{plot_type}`', 1)[1]
    section = section.split('\n## ', 1)[0]
    common = set(plot_config.PlotBase.model_fields)
    described, own, shared = [], [], []
    for line in section.splitlines():
        match = _ROW.match(line)
        if match is None or match.group(1) == 'plot_type':
            continue
        name, kind = match.group(1), _short_type(match.group(2))
        meaning = match.group(3).replace(r'\|', '/').strip()
        if meaning:
            described.append(f'{name} ({kind}): {meaning}')
        else:
            (shared if name in common else own).append(f'{name} ({kind})')
    lines = [f'{plot_type} fields:'] + described
    if own:
        lines += _wrapped(f'{plot_type} only:', own)
    if shared:
        lines += _wrapped('common to all plot types:', shared)
    return compact('\n'.join(lines))
