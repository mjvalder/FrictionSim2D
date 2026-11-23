"""Reconstruct ``provenance.json`` for legacy run trees, so ``aiida import`` takes them.

Runs built before provenance records existed carry only
``provenance/{config,manifest}.json``. The import needs one ``provenance.json`` per
run, so this writes one from what the tree does record, and leaves every field
it cannot know as ``None`` (or a ``legacy`` marker where the model needs a value).

An MLIP run whose directory uses the old ``K300_<framework>-native-<d3|base>``
token is moved to the path today's builder would give it
(:func:`frictionsim2d.run.naming.output_dir_for`), and the old relative path is rewritten
in its ``lammps/`` text files, which embed it.

Meant for COPIES of legacy trees (the publication sets are staged under
``~/mnt/aiida_pubdata``); never run it on an original.

    python tools/publication/legacy_provenance.py ROOT [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from frictionsim2d.aiida.block_schema import SPRING_METHODS
from frictionsim2d.data.models import NM_TO_EV_A2
from frictionsim2d.data.provenance import (MaterialProvenance, ProvenanceRecord, config_hash,
                                 new_run_uuid)
from frictionsim2d.run.naming import output_dir_for, parse_run_path

#: Values for required record fields that a legacy tree never recorded.
LEGACY_GIT_COMMIT = 'unknown-legacy'
LEGACY_CUTOFF_SOURCE = 'not_recorded_legacy'
LEGACY_SCHEMA = '2.1.0'

_GROUP_RE = re.compile(r'^\s*group\s+(layer_\d+)\s+type\s+(.+?)\s*(?:#.*)?$', re.M)


@dataclass
class ReconstructResult:
    """What :func:`reconstruct` did to one run directory."""
    run_dir: Path
    moved_to: Optional[Path] = None
    wrote: Optional[Path] = None
    notes: List[str] = field(default_factory=list)

    def patch_lines(self, root: Path) -> List[str]:
        """``PATCHES.md`` lines for this run."""
        rel = self.run_dir.relative_to(root).as_posix()
        lines = []
        if self.moved_to is not None:
            lines.append(f'- moved `{rel}` -> `{self.moved_to.relative_to(root).as_posix()}`, '
                         'old path rewritten in lammps/*: the MLIP path token predates '
                         'the parser (from: output_dir_for of its config.json)')
        if self.wrote is not None:
            lines.append(f'- wrote `{self.wrote.relative_to(root).as_posix()}`: legacy run had '
                         'none (from: provenance/config.json, manifest.json, '
                         'lammps/system.in.settings, deck)')
        lines += [f'- `{rel}`: {n}' for n in self.notes]
        return lines


def find_legacy_runs(root: Path) -> List[Path]:
    """Run dirs under ``root`` with ``provenance/config.json`` but no ``provenance.json``."""
    root = Path(root)
    return sorted(p.parent.parent for p in root.rglob('provenance/config.json')
                  if 'sync-conflict' not in str(p)
                  and not (p.parent.parent / 'provenance.json').exists())


def _layer_type_map(settings_text: str) -> Dict[str, List[int]]:
    return {name: [int(t) for t in types.split()]
            for name, types in _GROUP_RE.findall(settings_text)}


def _thermostat_scheme(deck_text: str) -> str:
    if 'langevin' in deck_text:
        return 'langevin_center_nve'
    if re.search(r'\bnvt\b', deck_text):
        return 'nvt_center'
    return 'unknown_legacy'


def _materials(config: dict, manifest: dict) -> List[MaterialProvenance]:
    sheet = config['sheet']
    pot_sums = [f['checksum'] for f in manifest.get('files', [])
                if f.get('category') == 'potential']
    return [MaterialProvenance(
        name=str(sheet['mat']), role='sheet', pot_path=sheet.get('pot_path'),
        pot_type=sheet.get('pot_type'), pot_file_sha256=pot_sums[0] if pot_sums else None)]


def _record(run_dir: Path, family: str) -> ProvenanceRecord:
    prov = run_dir / 'provenance'
    config = json.loads((prov / 'config.json').read_text(encoding='utf-8'))
    manifest_file = prov / 'manifest.json'
    manifest = (json.loads(manifest_file.read_text(encoding='utf-8'))
                if manifest_file.is_file() else {})
    lammps = run_dir / 'lammps'
    settings = lammps / 'system.in.settings'
    decks = sorted(lammps.glob('slide*.in'))
    sim = config.get('settings', {}).get('simulation', {})
    spring = config['general'].get('driving_spring')
    drive_k = (float(spring) / NM_TO_EV_A2
               if spring is not None and sim.get('drive_method') in SPRING_METHODS else None)
    return ProvenanceRecord(
        schema_version=LEGACY_SCHEMA,
        run_uuid=new_run_uuid(),
        run_uuid_stamped_at=date.today().isoformat(),
        family=family, model=family,
        materials=_materials(config, manifest),
        seed_thermal_tip=None, seed_thermal_sub=None, seed_velocity=None,
        substrate_realization_id=None, damping_ratio=None, damp_ev=None,
        git_commit=LEGACY_GIT_COMMIT, git_dirty=False,
        config_hash=config_hash(config),
        layer_type_map=(_layer_type_map(settings.read_text(encoding='utf-8'))
                        if settings.is_file() else {}),
        contact_cutoff=0.0, cutoff_source=LEGACY_CUTOFF_SOURCE,
        thermostat_scheme=(_thermostat_scheme(decks[0].read_text(encoding='utf-8'))
                           if decks else 'unknown_legacy'),
        run_role='label', load_requested=None,
        twist_angle=config['general'].get('twist_angle'),
        drive_k=drive_k,
    )


def _modern_rel(run_dir: Path, family: str) -> str:
    config = json.loads((run_dir / 'provenance' / 'config.json').read_text(encoding='utf-8'))
    return output_dir_for({'general': config['general'], '2D': config['sheet']},
                          family).as_posix()


def _move(run_dir: Path, root: Path, new_rel: str) -> Path:
    old_rel = run_dir.relative_to(root).as_posix()
    target = root / new_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    for text_file in (target / 'lammps').iterdir():
        if text_file.is_file():
            text = text_file.read_text(encoding='utf-8')
            if old_rel + '/' in text:
                text_file.write_text(text.replace(old_rel + '/', new_rel + '/'),
                                     encoding='utf-8')
    return target


def reconstruct(run_dir: Path, root: Path, *, dry_run: bool = False) -> ReconstructResult:
    """Make one legacy run importable: move it if its path does not parse, then
    write its ``provenance.json``. A run that already has one, or has no
    ``provenance/config.json``, is left alone (with a note for the latter)."""
    run_dir, root = Path(run_dir), Path(root)
    result = ReconstructResult(run_dir=run_dir)
    if (run_dir / 'provenance.json').exists():
        return result
    if not (run_dir / 'provenance' / 'config.json').is_file():
        result.notes.append('skipped: no provenance/config.json')
        return result
    rel = run_dir.relative_to(root).as_posix()
    family = rel.split('/', 1)[0]
    try:
        parse_run_path(rel)
    except ValueError:
        new_rel = _modern_rel(run_dir, family)
        parse_run_path(new_rel)  # a path today's parser still rejects is a bug here
        result.moved_to = root / new_rel
        if not dry_run:
            run_dir = _move(run_dir, root, new_rel)
    record = _record(run_dir, family)
    result.wrote = run_dir / 'provenance.json'
    if not dry_run:
        result.wrote.write_text(record.model_dump_json(indent=2), encoding='utf-8')
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: reconstruct every legacy run under ROOT, print PATCHES.md lines."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    runs = find_legacy_runs(args.root)
    if not runs:
        print(f'{args.root}: no legacy runs')
        return 0
    for run_dir in runs:
        for line in reconstruct(run_dir, args.root, dry_run=args.dry_run).patch_lines(args.root):
            print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
