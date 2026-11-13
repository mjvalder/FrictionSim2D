"""Acceptance of the ported integrity checker against ML_taxonomy's reference.

    python tools/integrity/compare_reference.py afm   <run_root> <parquet> [--jobs 8]
    python tools/integrity/compare_reference.py sheet <run_root> <parquet> [--jobs 8]

`<run_root>` is a simulation root (the directory holding `afm/` or
`sheetonsheet/`). It is read, never written: every material directory is
mirrored into a scratch root (real directories, symlinked files), and one
`DataReader` runs per material there (DataReader creates `<root>/outputs/`).
Sharding by material changes nothing the checker sees -- chains, multi-speed
and end-snapshot detection are all per run -- except the legacy truncation
rule for runs whose config gives no expected length; the tool counts such
points and prints the count.

Joins:
  - afm:   s48 `run_id` = `material/l{layer}/f{F:g}/a0` (angle 0 only; s48
           writes the material with `-` replaced by `_`);
  - sheet: `(material, pressure_GPa, angle)`, against `verdict_variant` /
           `reasons_variant` (s51's variant: settled heating peak, lasting gap
           shift).

Prints the confusion table, every disagreement with both sides' measures
(flagging the ones the known sigma_T difference explains), a measure-agreement
summary and the margin table: for each rule and margin m, how many points the
port passes would be quarantined (value in [m * threshold, threshold)).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.config import IntegritySettings
from src.postprocessing.read_data import DataReader

MODELS = {'afm': 'afm', 'sheet': 'sheetonsheet'}
MARGINS = (0.5, 0.6, 0.7, 0.8, 0.9)
#: (rule, our measure key, IntegritySettings attribute): the rules a margin
#: can apply to (`verdict` never applies one to boolean rules or collapse).
MARGIN_RULES = (('hole', 'hole', 'hole_bonds'),
                ('isolated_atoms', 'isolated', 'isolated_atoms'),
                ('ejected_atoms', 'ejected', 'ejected_atoms'),
                ('sheet_heating', 'T_sigma', 'heating_sigma'),
                ('gap_rise', 'gap_rise', 'gap_rise'))
#: Our measure key -> reference column, per campaign.
REF_MEASURES = {
    'afm': {'hole': 'hole', 'isolated': 'isolated', 'ejected': 'ejected',
            'T_sigma': 'T_sigma', 'gap_rise': 'gap_rise'},
    'sheet': {'hole': 'hole', 'isolated': 'isolated', 'ejected': 'ejected',
              'T_sigma': 'T_sigma_settled', 'gap_rise': 'gap_rise_variant',
              'gap_min_any': 'gap_min_any', 'n_heat': 'n_center'},
}
#: s48 spells our `no_output` gate `crashed` (s48 :46).
GATE_ALIAS = {'crashed': 'no_output'}
OUR_COMPLETE = ('complete', 'failed_integrity', 'quarantined')


def _norm(material: str) -> str:
    return material.replace('-', '_')


def _mirror(src: Path, dst: Path) -> None:
    """Rebuild ``src``'s directories under ``dst`` with every file a symlink.

    Directories are real, not symlinks, because ``discover_runs`` walks with
    ``Path.rglob``, which does not descend into a symlinked directory.
    """
    for dirpath, _dirnames, filenames in os.walk(src):
        target = dst / Path(dirpath).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (target / name).symlink_to(Path(dirpath) / name)


def _shard(args) -> List[dict]:
    """One DataReader on one material, through a symlinked scratch root."""
    model, mat_dir, scratch = args
    logging.disable(logging.WARNING)
    root = Path(tempfile.mkdtemp(prefix=f'{Path(mat_dir).name}_', dir=scratch))
    _mirror(Path(mat_dir), root / model / Path(mat_dir).name)
    reader = DataReader(str(root), integrity_settings=IntegritySettings())
    rows = []
    for p in reader.points:
        v = p.integrity
        tok = p.tokens
        if v is None:
            file_status = p.status
        elif v.status in OUR_COMPLETE:
            file_status = 'complete'
        else:
            file_status = v.status
        rules = sorted({r.split()[0] for r in (v.reasons if v else [])} - {file_status})
        row = dict(material=_norm(tok.materials[0]),  # pylint: disable=use-dict-literal
                   layer=int(tok.layers or p.n_layers),
                   load=float(p.load), angle=float(p.angle), speed=float(p.speed),
                   rel_path=tok.rel_path, file_status=file_status,
                   status=v.status if v else p.status, rules=';'.join(rules),
                   quarantine='; '.join(v.quarantine) if v else '',
                   not_applicable='; '.join(v.not_applicable) if v else '',
                   no_expected_rows=not p.expected_rows,
                   T_target=float(tok.temperature))
        for k, val in ((v.measures if v else {}) or {}).items():
            if not isinstance(val, dict):
                row[k] = float(val) if val is not None else math.nan
        rows.append(row)
    return rows


def read_ours(model: str, run_root: Path, jobs: int, scratch: Path,
              materials: Optional[List[str]]) -> pd.DataFrame:
    """Every point of the campaign through the port, one row each."""
    mats = sorted(d for d in (run_root / model).iterdir() if d.is_dir())
    if materials:
        mats = [d for d in mats if d.name in materials]
    tasks = [(model, str(d.resolve()), str(scratch)) for d in mats]
    if jobs > 1:
        with Pool(jobs) as pool:
            parts = pool.map(_shard, tasks, chunksize=1)
    else:
        parts = [_shard(t) for t in tasks]
    return pd.DataFrame([r for part in parts for r in part])


def _ref_rules(reasons) -> str:
    rs = [GATE_ALIAS.get(r, r) for r in str(reasons or '').split(';') if r]
    return ';'.join(sorted(rs))


def join(kind: str, ours: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Outer join of the port's rows to the reference's, on the campaign key."""
    ref = ref.copy()
    if kind == 'afm':
        ours = ours[ours['angle'] == 0.0].copy()
        ours['key'] = (ours['material'] + '/l' + ours['layer'].astype(str) + '/f'
                       + ours['load'].map(lambda x: f'{x:g}') + '/a0')
        ref['key'] = ref['run_id']
        ref['ref_verdict'] = ref['verdict']
        ref['ref_rules'] = [_ref_rules(r) for r in ref['reasons']]
        ref_cols = ['key', 'status', 'ref_verdict', 'ref_rules', 'hole', 'isolated',
                    'ejected', 'T_sigma', 'gap_rise', 'non_finite', 'frozen']
    else:
        ours = ours.copy()
        ref['material'] = ref['material'].map(_norm)
        ours['key'] = (ours['material'] + '|p' + ours['load'].map(lambda x: f'{x:g}')
                       + '|a' + ours['angle'].map(lambda x: f'{x:g}'))
        ref['key'] = (ref['material'] + '|p' + ref['pressure_GPa'].map(lambda x: f'{x:g}')
                      + '|a' + ref['angle'].map(lambda x: f'{x:g}'))
        ref['ref_verdict'] = ref['verdict_variant']
        ref['ref_rules'] = [_ref_rules(r) for r in ref['reasons_variant']]
        ref['gap_rise_variant'] = ref['gap_shift_end'].abs()
        ref_cols = ['key', 'status', 'ref_verdict', 'ref_rules', 'hole', 'isolated',
                    'ejected', 'T_sigma_settled', 'gap_rise_variant', 'gap_min_any',
                    'n_center', 'non_finite', 'frozen']
    ref = ref[ref_cols].rename(columns={c: f'ref_{c}' for c in ref_cols
                                        if c not in ('key', 'ref_verdict', 'ref_rules')})
    return ours.merge(ref, on='key', how='outer', indicator=True)


def _our_rule_set(row) -> str:
    rules = [r for r in str(row['rules'] or '').split(';') if r]
    if row['file_status'] != 'complete':
        rules.append(row['file_status'])
    return ';'.join(sorted(rules))


def known_heating_ref(df: pd.DataFrame, kind: str) -> pd.Series:
    """Our T_sigma recomputed the reference's way: N = the per-material
    median (s48 :190, s51 :168) and T_target = 300 K (s48 :44, s51 :49)."""
    group = ['material', 'layer'] if kind == 'afm' else ['material']
    # The reference's N comes from the end snapshots only (s48 `n`, s51
    # `n_center`); a point without one has no N there.
    n_snap = df['n_heat'].where(pd.to_numeric(df['hole'], errors='coerce').notna())
    n_med = n_snap.groupby([df[g] for g in group]).transform('median')
    return df['T_sigma'] * np.sqrt(n_med / df['n_heat']) * (df['T_target'] / 300.0)


def classify(df: pd.DataFrame, kind: str, settings: IntegritySettings) -> pd.DataFrame:
    """Agreement per joined point, and whether a known difference explains it."""
    both = df['_merge'] == 'both'
    df = df[both].copy()
    df['our_rules'] = [_our_rule_set(r) for _, r in df.iterrows()]
    df['T_sigma_refstyle'] = known_heating_ref(df, kind)
    explained = []
    for _, r in df.iterrows():
        ours = set(filter(None, r['our_rules'].split(';')))
        ref = set(filter(None, r['ref_rules'].split(';')))
        if r['status'] == 'quarantined':
            ours.add('QUARANTINED')
        if ours == ref:
            explained.append('agree')
            continue
        diff = ours ^ ref
        gate = r['file_status']
        if ours == {'not_reached'} and ref == {'no_output'}:
            explained.append('gate spelling: port not_reached, reference no_output '
                             '(both failed)')
            continue
        if (gate != 'complete' and gate == GATE_ALIAS.get(r['ref_status'], r['ref_status'])
                and ref == {gate} and gate in ours):
            explained.append('gate agrees; port also measured the window '
                             '(reference had no frames)')
            continue
        if diff == {'sheet_heating'} and not math.isnan(r['T_sigma_refstyle']):
            refstyle_fires = r['T_sigma_refstyle'] >= settings.heating_sigma
            if refstyle_fires == ('sheet_heating' in ref):
                explained.append('known: sigma_T N per point vs material median')
                continue
        explained.append('UNEXPLAINED')
    df['agreement'] = explained
    return df


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        return 'nan' if math.isnan(x) else f'{x:.4g}'
    return str(x)


def report(kind: str, df: pd.DataFrame, joined: pd.DataFrame,
           settings: IntegritySettings) -> List[str]:
    """The printed report, as markdown lines."""
    out = [f'# Integrity acceptance: {kind}', '']
    only_ours = joined[joined['_merge'] == 'left_only']
    only_ref = joined[joined['_merge'] == 'right_only']
    out += [f'Points: port {int((joined["_merge"] != "right_only").sum())}, reference '
            f'{int((joined["_merge"] != "left_only").sum())}, joined {len(df)}; '
            f'port only {len(only_ours)}, reference only {len(only_ref)}.', '']
    if len(only_ours) or len(only_ref):
        out += ['Unjoined keys (first 20 each):', '']
        out += [f'- port only: {k}' for k in only_ours['key'].head(20)]
        out += [f'- reference only: {k}' for k in only_ref['key'].head(20)]
        out.append('')
    n_legacy = int(df['no_expected_rows'].astype(bool).sum())
    out += [f'Points without expected_rows (legacy truncation rule, sharding-sensitive): '
            f'{n_legacy}.', '']

    out += ['## Confusion (port status x reference verdict)', '']
    tab = pd.crosstab(df['status'], df['ref_verdict'], margins=True)
    out += _md_table(tab.reset_index().rename(columns={'status': 'port \\ ref'}))
    out += ['', '## Confusion (file-level status: port x reference)', '']
    tab = pd.crosstab(df['file_status'], df['ref_status'], margins=True)
    out += _md_table(tab.reset_index().rename(columns={'file_status': 'port \\ ref'}))

    failed = df[df['status'].isin(['failed_integrity'])
                | ~df['status'].isin(OUR_COMPLETE + ('not_reached',))]
    after = df[df['status'] == 'failed_integrity']
    n_nr = int((df['status'] == 'not_reached').sum())
    out += ['', f'Port: {len(failed)} failed points, not counting not_reached '
            f'({len(after)} failed after the run, i.e. failed_integrity), in '
            f'{failed["material"].nunique()} materials. not_reached: {n_nr} '
            '(verdict failed=True, observed=False; s48 counts it apart, s51 calls it no_output).']
    if kind == 'afm':
        out.append('Failed after the run by layer: ' + ', '.join(
            f'l{k}: {v}' for k, v in after.groupby('layer').size().items()))
    ref_fail = df[df['ref_verdict'] == 'fail']
    out.append(f'Reference: {len(ref_fail)} failed points in '
               f'{ref_fail["material"].nunique()} materials.')
    out += ['', '## Rule counts among failed points (port / reference)', '',
            '| rule | port | reference |', '|---|---|---|']
    for rule in ('hole', 'isolated_atoms', 'ejected_atoms', 'sheet_heating', 'gap_rise',
                 'collapse', 'non_finite', 'frozen', 'no_output', 'halted', 'truncated',
                 'unreadable'):
        p = int(df['our_rules'].str.split(';').apply(lambda s, k=rule: k in s).sum())
        r = int(df['ref_rules'].str.split(';').apply(lambda s, k=rule: k in s).sum())
        out.append(f'| {rule} | {p} | {r} |')

    out += ['', '## Agreement', '']
    out += [f'- {k}: {v}' for k, v in df['agreement'].value_counts().items()]
    both_complete = df[(df['file_status'] == 'complete') & (df['ref_status'] == 'complete')]
    out.append(f'- of which both complete: {len(both_complete)}, rule sets identical '
               f'{int((both_complete["agreement"] == "agree").sum())}')

    out += ['', '## Measure agreement (joined points with both values)', '',
            '| measure | n | max abs diff | points differing > 1e-4 x max(abs(ref), 1) |',
            '|---|---|---|---|']
    for ours_key, ref_col in REF_MEASURES[kind].items():
        if ours_key not in df or f'ref_{ref_col}' not in df:
            continue
        a = pd.to_numeric(df[ours_key], errors='coerce')
        b = pd.to_numeric(df[f'ref_{ref_col}'], errors='coerce')
        if ours_key == 'T_sigma':
            a = df['T_sigma_refstyle']
            ours_key = 'T_sigma (port, reference-style N)'
        ok = a.notna() & b.notna()
        d = (a[ok] - b[ok]).abs()
        rel = d / b[ok].abs().clip(lower=1.0)
        out.append(f'| {ours_key} vs {ref_col} | {int(ok.sum())} | '
                   f'{_fmt(d.max() if len(d) else math.nan)} | {int((rel > 1e-4).sum())} |')

    dis = df[df['agreement'] != 'agree']
    out += ['', f'## Disagreements ({len(dis)})', '']
    if len(dis):
        cols = ['key', 'file_status', 'ref_status', 'status', 'our_rules', 'ref_rules',
                'agreement', 'quarantine']
        for ours_key, ref_col in REF_MEASURES[kind].items():
            cols += [c for c in (ours_key, f'ref_{ref_col}') if c in dis]
        cols.append('T_sigma_refstyle')
        out += _md_table(dis.sort_values('key')[cols])

    out += ['', '## Margin table', '',
            'Points the port passes (status `complete`) whose value lies in '
            '[m x threshold, threshold): each would become `quarantined` with that margin.', '',
            '| rule | threshold | ' + ' | '.join(f'm={m:g}' for m in MARGINS) + ' |',
            '|---|---|' + '---|' * len(MARGINS)]
    passing = df[df['status'] == 'complete']
    any_hit = {m: pd.Series(False, index=passing.index) for m in MARGINS}
    for rule, key, attr in MARGIN_RULES:
        th = getattr(settings, attr)
        cells = []
        for m in MARGINS:
            val = pd.to_numeric(passing.get(key), errors='coerce')
            hit = (val >= m * th) & (val < th)
            any_hit[m] |= hit.fillna(False)
            cells.append(str(int(hit.sum())))
        out.append(f'| {rule} | {th:g} | ' + ' | '.join(cells) + ' |')
    out.append('| any rule (same m for all) | | '
               + ' | '.join(str(int(any_hit[m].sum())) for m in MARGINS) + ' |')
    out.append(f'\nPassing points: {len(passing)}.')
    return out


def _md_table(frame: pd.DataFrame) -> List[str]:
    cols = [str(c) for c in frame.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row in frame.itertuples(index=False):
        lines.append('| ' + ' | '.join(_fmt(x) for x in row) + ' |')
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    ap.add_argument('kind', choices=sorted(MODELS))
    ap.add_argument('run_root', type=Path)
    ap.add_argument('parquet', type=Path)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--scratch', type=Path, default=None,
                    help='where the symlinked shard roots go (default: a temp dir)')
    ap.add_argument('--materials', nargs='*', default=None,
                    help='restrict to these material directory names')
    ap.add_argument('--csv', type=Path, default=None, help='write the joined table here')
    args = ap.parse_args(argv)

    settings = IntegritySettings()
    with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
        ours = read_ours(MODELS[args.kind], args.run_root.expanduser(), args.jobs,
                         Path(scratch), args.materials)
    ref = pd.read_parquet(args.parquet.expanduser())
    if args.materials:
        names = {_norm(m) for m in args.materials}
        ref = ref[ref['material'].map(_norm).isin(names)]
    joined = join(args.kind, ours, ref)
    df = classify(joined, args.kind, settings)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print('\n'.join(report(args.kind, df, joined, settings)))
    print('\nReference measures are float32 (T_sigma, gap_rise): a difference '
          'below 1e-4 x max(|ref|, 1) is rounding.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
