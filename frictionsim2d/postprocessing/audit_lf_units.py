#!/usr/bin/env python
"""Census of sheet results files affected by the pre-schema-2 lateral-force defect.

A pre-schema-2 sheet run on the LJ branch emitted ``xfrict`` as ``compute
group/group``'s force in eV/A, while the load it is divided by was already nN.
Its COF is therefore a factor 1.602176565 too SMALL. The many-body branch
defined ``xfrict`` AS ``v_sx``, so those two columns are bit-identical there
and the file is unaffected. That exact equality is the discriminator.

Usage:  python -m frictionsim2d.postprocessing.audit_lf_units ~/mnt/PhD/results
"""
import sys
from collections import Counter
from pathlib import Path

from frictionsim2d.postprocessing.read_data import is_lj_branch, parse_lf_schema

VERDICTS = ('affected', 'unaffected', 'unparseable')


def classify_file(path) -> str:
    """Return 'affected' | 'unaffected' | 'unparseable' for one results file."""
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            head = [next(fh, ''), next(fh, ''), next(fh, '')]
    except OSError:
        return 'unparseable'
    # Parsed as an integer via the reader's own helper, so the audit and the
    # corrector cannot drift: any schema >= 2 is post-fix and already nN.
    if parse_lf_schema(head[0]) >= 2:
        return 'unaffected'
    cols = head[1].lstrip('#').split()
    if 'v_xfrict' not in cols or 'v_sx' not in cols:
        return 'unparseable'
    ix, isx = cols.index('v_xfrict'), cols.index('v_sx')
    values = head[2].split()
    if len(values) <= max(ix, isx):
        return 'unparseable'
    try:
        # Same rule as the reader's correction path, by construction.
        return ('affected'
                if is_lj_branch([float(values[ix])], [float(values[isx])])
                else 'unaffected')
    except ValueError:
        return 'unparseable'


def main(root: str) -> int:
    root_path = Path(root).expanduser()
    counts: Counter = Counter()
    by_set: dict = {}
    for path in root_path.rglob('*'):
        if not path.is_file() or 'sheet' not in str(path).lower():
            continue
        if not (path.name.startswith('fc_ave_slide')
                or path.name.startswith('friction_')):
            continue
        verdict = classify_file(path)
        counts[verdict] += 1
        try:
            top = path.relative_to(root_path).parts[0]
        except ValueError:
            top = str(path.parent)
        by_set.setdefault(top, Counter())[verdict] += 1

    for verdict in VERDICTS:
        print(f"{verdict:>12}: {counts[verdict]}")
    print("\nper result set:")
    for name in sorted(by_set):
        c = by_set[name]
        mixed = ' <-- MIXED' if c['affected'] and c['unaffected'] else ''
        print(f"  {name:<28} affected={c['affected']:<6} "
              f"unaffected={c['unaffected']:<5} "
              f"unparseable={c['unparseable']}{mixed}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '.'))
