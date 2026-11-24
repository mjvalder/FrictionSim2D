"""Analysis tools: find result sets, summarise one, describe the plot types.

Read-only. Two kinds of result set exist in a workspace:

* a **dataset** -- a folder with `outputs/output_full_*.json` (the rule the
  Plots tab uses, `frictionsim2d.gui.datasets.is_dataset`), which is what plots read;
* a **simulation root** -- a folder whose `<model>/` subtrees hold runs
  (`frictionsim2d.run.tree.discover_runs`), built but maybe not post-processed yet.

A folder can be both: `postprocess read` writes `outputs/` into the root.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...gui.datasets import DATASETS_DIRNAME, dataset_files, is_dataset
from ...postprocessing import plot_config
from ...postprocessing.integrity_rulings import format_point_id
from ...postprocessing.run_status import STATUSES, expected_chains, read_run_config
from ...run.naming import MODELS
from ...run.tree import discover_runs
from ..changes import ToolInputError
from ..sandbox import display_path, resolve_readable
from . import RESULT_LIMIT, compact

#: Largest `output_full_*.json` `summarise_results` will parse ...
MAX_JSON_MB = 200
#: ... and the most it parses in one call, all files of a dataset together;
#: the files past it are listed as "not read (budget)".
SUMMARY_BUDGET_MB = 400
#: Largest file `find_result_sets` parses just to count its runs ...
COUNT_JSON_MB = 20
#: ... and the most it parses for counting in one call, all files together.
COUNT_BUDGET_MB = 50
MAX_FIND_DEPTH = 4
#: Result-set lines `find_result_sets` collects: about what fits in one result.
MAX_FIND_HITS = 15
#: `provenance.json` files counted per simulation root before saying "N+".
MAX_ROOT_RUNS = 1000
#: At most this many scalar `post` entries shown per row.
MAX_POST = 3
#: Kept free at the end of a result for the "… N more" line.
TAIL_RESERVE = 60


# ---- finding -----------------------------------------------------------------

def _children(folder: Path, real_root: str) -> List[Path]:
    """Sub-folders, without dot-folders or symlinks that leave the workspace."""
    try:
        children = sorted(p for p in folder.iterdir()
                          if p.is_dir() and not p.name.startswith('.'))
    except OSError:
        return []
    kept = []
    for child in children:
        if child.is_symlink():
            real = os.path.realpath(child)
            if real != real_root and not real.startswith(real_root + os.sep):
                continue
        kept.append(child)
    return kept


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


def _dataset_line(deps, folder: Path, budget: List[float]) -> str:
    """One dataset line; parses for run counts while ``budget[0]`` MB remain."""
    files = dataset_files(folder)
    sizes = ', '.join(_size_of(p) for p in files)
    total = sum(_megabytes(p) for p in files)
    counts: Optional[List[int]] = None
    if total <= budget[0] and all(_megabytes(p) <= COUNT_JSON_MB for p in files):
        budget[0] -= total
        counts = []
        for path in files:
            doc, _ = _load(path, COUNT_JSON_MB)
            if doc is None:
                counts = None
                break
            counts.append(len(doc.get('runs') or []))
    tail = f'{sum(counts)} runs' if counts is not None else f'{total:.0f} MB'
    return f'{display_path(folder, deps.roots)} — dataset (sizes: {sizes}; {tail})'


def _count_runs(folder: Path, cap: int = MAX_ROOT_RUNS) -> Tuple[Counter, bool]:
    """`provenance.json` files per model under ``folder``, stopping at ``cap``."""
    counts: Counter = Counter()
    for model in MODELS:
        model_dir = folder / model
        if not model_dir.is_dir():
            continue
        for _base, _dirs, names in os.walk(model_dir):
            if 'provenance.json' in names:
                counts[model] += 1
                if sum(counts.values()) >= cap:
                    return counts, True
    return counts, False


def _root_line(deps, folder: Path) -> Optional[str]:
    counts, capped = _count_runs(folder)
    if not counts:
        return None
    shown = [f'{model}: {counts[model]} runs' for model in MODELS if model in counts]
    if capped:
        shown[-1] = shown[-1].replace(' runs', '+ runs')
    return f"{display_path(folder, deps.roots)} — simulation root ({', '.join(shown)})"


def find_result_sets(deps, max_hits: int = MAX_FIND_HITS,
                     count_budget_mb: float = COUNT_BUDGET_MB) -> str:
    """Datasets and simulation roots in the workspace (4 levels deep at most)."""
    root = Path(deps.roots.workspace)
    real_root = os.path.realpath(root)
    queue = deque([(root, 0)])
    exported = root / '.preview' / DATASETS_DIRNAME
    if exported.is_dir():
        queue.append((exported, 2))
    budget = [float(count_budget_mb)]
    lines: List[str] = []
    while queue and len(lines) <= max_hits:
        folder, depth = queue.popleft()
        if is_dataset(folder):
            lines.append(_dataset_line(deps, folder, budget))
        has_runs = False
        if any((folder / model).is_dir() for model in MODELS):
            line = _root_line(deps, folder)
            if line is not None:
                lines.append(line)
                has_runs = True
        if depth >= MAX_FIND_DEPTH:
            continue
        for child in _children(folder, real_root):
            if child.name == 'outputs' or (has_runs and child.name in MODELS):
                continue
            queue.append((child, depth + 1))
    if not lines:
        return ('no datasets or simulation roots in the workspace (searched '
                f'{MAX_FIND_DEPTH} levels deep)')
    if len(lines) > max_hits:
        lines = lines[:max_hits] + [f'… more result sets; stopped after {max_hits}']
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
    counts = meta.get('status_counts') or Counter(
        run.get('status') for run in doc.get('runs') or [] if isinstance(run, dict))
    status = ', '.join(f'{s} {counts[s]}' for s in STATUSES if counts.get(s))
    lines = [f"{path.name}: {len(doc.get('runs') or [])} runs — {status or 'no status'}"]
    lines.append(f"materials: {', '.join(meta.get('materials') or [])}; "
                 f"T: {', '.join(_num(t) for t in meta.get('temperatures') or [])}; "
                 f"loads: {_loads(meta) or '-'}")
    return lines


def _ruling_line(run: dict) -> str:
    """A quarantined point: its ruling id, then why."""
    try:
        pid = format_point_id(str(run.get('rel_path')), str(run.get('load_kind')),
                              run.get('load'), run.get('angle'), run.get('speed'))
    except (TypeError, ValueError):
        pid = _row(run)
    reason = ' '.join(str(run.get('status_detail') or '').split())
    return f'{pid} -- {reason}' if reason else pid


def _summarise_dataset(deps, folder: Path, files: List[Path], filters: Dict[str, Any],
                       max_rows: int, max_json_mb: float, budget_mb: float) -> str:
    lines = [f'{display_path(folder, deps.roots)} — dataset, {len(files)} file(s)']
    rows: List[Tuple[int, str]] = []
    awaiting: List[str] = []
    overridden: List[str] = []
    for path in files:
        if filters['size'] and _norm_size(_size_of(path)) != _norm_size(filters['size']):
            continue
        size = _megabytes(path)
        if size <= max_json_mb:  # an oversized file is refused below, at no cost
            if size > budget_mb:
                lines.append(f'{path.name}: not read (budget)')
                continue
            budget_mb -= size
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
                if status == 'quarantined':
                    awaiting.append(_ruling_line(run))
                integrity = run.get('integrity')
                if isinstance(integrity, dict) and integrity.get('overridden'):
                    overridden.append(
                        f"{_ruling_line(run)} ({integrity.get('original_status')}"
                        f" -> {status})")
    if not rows:
        return '\n'.join(lines + [_no_match(filters)])
    if awaiting:
        lines += ['awaiting your ruling (add to integrity_rulings.yaml):'] + [
            f'  {a}' for a in awaiting]
    if overridden:
        lines += ['overridden by you (integrity_rulings.yaml):'] + [
            f'  {o}' for o in overridden]
    rows.sort(key=lambda r: r[0])  # stable: complete first, file order kept
    max_rows = max(1, int(max_rows))
    return _fit(lines, [row for _, row in rows[:max_rows]], len(rows[max_rows:]), 'rows')


def _fit(header: List[str], rows: List[str], more: int, noun: str) -> str:
    """``header`` + as many ``rows`` as fit in a result, then "… N more {noun}"."""
    room = RESULT_LIMIT - TAIL_RESERVE - sum(len(line) + 1 for line in header)
    shown: List[str] = []
    for row in rows:
        if len(row) + 1 > room:
            break
        shown.append(row)
        room -= len(row) + 1
    more += len(rows) - len(shown)
    tail = [f'… {more} more {noun}'] if more else []
    return '\n'.join(header + shown + tail)


def _no_match(filters: Dict[str, Any]) -> str:
    wanted = ', '.join(f'{k}={v}' for k, v in filters.items() if v)
    return f'no runs match {wanted}' if wanted else 'no runs'


def _token_row(run) -> str:
    tokens = run.tokens
    parts = [tokens.model if tokens.family == tokens.model
             else f'{tokens.model}/{tokens.family}',
             '+'.join(tokens.materials), f'{tokens.size_x:g}x{tokens.size_y:g}y',
             f'T{tokens.temperature:g}']
    if tokens.layers is not None:
        parts.append(f'L{tokens.layers}')
    if tokens.pot_token:
        parts.append(f'pot {tokens.pot_token}')
    return ' '.join(parts)


def _summarise_root(deps, folder: Path, runs, filters: Dict[str, Any], max_rows: int,
                    config_reader: Callable) -> str:
    """Rows from path tokens; config and results/ are looked at for shown rows only."""
    header = [f'{display_path(folder, deps.roots)} — simulation root, {len(runs)} runs '
              '(not post-processed: run `FrictionSim2D postprocess read` on it for a dataset)']
    matching = []
    for run in runs:
        tokens = run.tokens
        fake = {'material': tokens.materials[0] if tokens.materials else None,
                'materials': tokens.materials,
                'size': f'{tokens.size_x:g}x{tokens.size_y:g}y'}
        if _matches(fake, filters['material'], filters['size']):
            matching.append(run)
    if not matching:
        return '\n'.join(header + [_no_match(filters)])
    max_rows = max(1, int(max_rows))
    rows = []
    for run in matching[:max_rows]:
        chains = expected_chains(config_reader(run), run.tokens.family)
        expected = ('grid unknown' if chains is None
                    else f'{sum(len(c) for c in chains)} points expected')
        try:
            n_files = sum(1 for p in run.results_dir.iterdir() if p.is_file())
        except OSError:
            n_files = 0
        rows.append(f'{_token_row(run)}: {expected}, {n_files} results files')
    return _fit(header, rows, len(matching[max_rows:]), 'runs not inspected')


def summarise_results(deps, path: str, material: Optional[str] = None,
                      size: Optional[str] = None, max_rows: int = 15,
                      max_json_mb: float = MAX_JSON_MB,
                      config_reader: Callable = read_run_config, *,
                      summary_budget_mb: float = SUMMARY_BUDGET_MB) -> str:
    """Status counts, materials, loads and a run table for a dataset or simulation root.

    A dataset's files are parsed while ``summary_budget_mb`` lasts, each one
    no larger than ``max_json_mb``; the rest are named but not read.
    """
    target = resolve_readable(path, deps.roots)
    filters = {'material': material, 'size': size}
    if target.is_file() and target.match('output_full_*.json'):
        return compact(_summarise_dataset(deps, target.parent.parent, [target], filters,
                                          max_rows, max_json_mb, summary_budget_mb))
    if is_dataset(target):
        return compact(_summarise_dataset(deps, target, dataset_files(target), filters,
                                          max_rows, max_json_mb, summary_budget_mb))
    runs = discover_runs(target) if target.is_dir() else []
    if runs:
        return compact(_summarise_root(deps, target, runs, filters, max_rows,
                                       config_reader))
    raise ToolInputError(f'{display_path(target, deps.roots)}: neither a dataset nor a '
                         'simulation root; use find_result_sets to list them')


# ---- plots -------------------------------------------------------------------

def _plot_models() -> Dict[str, type]:
    """plot_type -> its model, read off the PlotBase subclasses."""
    models = {}
    pending = list(plot_config.PlotBase.__subclasses__())
    while pending:  # every descendant: some types share an intermediate base
        model = pending.pop()
        pending.extend(model.__subclasses__())
        field = model.model_fields.get('plot_type')
        if field is not None and isinstance(field.default, str):
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
    text = annotation.replace('frictionsim2d.postprocessing.plot_config.', '')
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
    """The fields one plot type accepts, from its plot model; its own fields first."""
    if plot_type not in plot_config.PLOT_TYPES:
        raise ToolInputError(f'unknown plot type {plot_type!r}; one of: '
                             + ', '.join(plot_config.PLOT_TYPES))
    common = set(plot_config.PlotBase.model_fields)
    described, own, shared = [], [], []
    for name, info in _plot_models()[plot_type].model_fields.items():
        if name == 'plot_type':
            continue
        kind = _short_type(str(info.annotation).replace('typing.', ''))
        meaning = (info.description or '').replace('|', '/').strip()
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
