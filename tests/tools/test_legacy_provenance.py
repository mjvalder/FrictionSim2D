"""tools/publication/legacy_provenance.py: legacy run trees made importable.

The fixtures under tests/fixtures/legacy_trees/ are verbatim copies of the
provenance/ and lammps/ files of two publication runs: 260227 REBOMoS (a
legacy path that already parses) and 260830 DeePMD (an MLIP path,
``K300_deepmd-native-d3``, that today's parser rejects).
"""

import json
import shutil
import uuid
from pathlib import Path

import pytest

from frictionsim2d.data.provenance import ProvenanceRecord
from frictionsim2d.run.naming import parse_run_path
from frictionsim2d.run.tree import discover_runs
from tools.publication.legacy_provenance import find_legacy_runs, reconstruct

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'legacy_trees'
REBOMOS_REL = 'sheetonsheet/h-MoS2/100x_100y/K300'
DEEPMD_REL = 'sheetonsheet/h-MoS2/30x_30y/K300_deepmd-native-d3'


def _tree(tmp_path: Path, fixture: str, rel: str) -> tuple:
    root = tmp_path / 'simulation_20260227_164626'
    run = root / rel
    shutil.copytree(FIXTURES / fixture, run)
    (run / 'results').mkdir()
    (run / 'results' / 'friction_p1_a90_s20').write_text('# header\n', encoding='utf-8')
    pot = run / 'provenance' / 'potentials'
    pot.mkdir(parents=True)
    return root, run


def test_reconstruct_writes_valid_record(tmp_path):
    root, run = _tree(tmp_path, 'rebomos', REBOMOS_REL)
    result = reconstruct(run, root)
    record = ProvenanceRecord.model_validate_json(result.wrote.read_text(encoding='utf-8'))
    manifest = json.loads((run / 'provenance' / 'manifest.json').read_text(encoding='utf-8'))
    pot_sum = next(f['checksum'] for f in manifest['files'] if f['category'] == 'potential')
    assert record.family == 'sheetonsheet'
    assert record.materials[0].pot_type == 'rebomos'
    assert record.materials[0].pot_file_sha256 == pot_sum
    assert uuid.UUID(record.run_uuid)
    assert record.run_uuid_stamped_at
    assert record.layer_type_map == {f'layer_{i}': [i, i + 5] for i in range(1, 6)}
    assert record.drive_k is None  # fix_move: no spring
    assert result.moved_to is None


def test_reconstruct_moves_to_parseable_path(tmp_path):
    root, run = _tree(tmp_path, 'deepmd', DEEPMD_REL)
    with pytest.raises(ValueError):
        parse_run_path(DEEPMD_REL)
    result = reconstruct(run, root)
    assert result.moved_to is not None
    new_rel = result.moved_to.relative_to(root).as_posix()
    tokens = parse_run_path(new_rel)
    assert (tokens.mlip_framework, tokens.dispersion) == ('deepmd', 'd3')
    runs = discover_runs(root)
    assert [r.tokens.rel_path for r in runs] == [new_rel]
    deck = (result.moved_to / 'lammps' / 'slide_p1gpa.in').read_text(encoding='utf-8')
    assert DEEPMD_REL not in deck and new_rel in deck


def test_reconstruct_skips_run_without_config(tmp_path):
    root = tmp_path / 'simulation_x'
    run = root / REBOMOS_REL
    (run / 'results').mkdir(parents=True)
    assert find_legacy_runs(root) == []
    result = reconstruct(run, root)
    assert result.wrote is None
    assert any('config.json' in n for n in result.notes)


def test_existing_provenance_untouched(tmp_path):
    root, run = _tree(tmp_path, 'rebomos', REBOMOS_REL)
    (run / 'provenance.json').write_text('{"family": "x"}', encoding='utf-8')
    before = (run / 'provenance.json').read_bytes()
    assert find_legacy_runs(root) == []
    result = reconstruct(run, root)
    assert result.wrote is None
    assert (run / 'provenance.json').read_bytes() == before


def test_single_deck_named_slide_in(tmp_path):
    """260312 runs have one ``slide.in`` looping over pressures, not ``slide_p*.in``."""
    root, run = _tree(tmp_path, 'rebomos', REBOMOS_REL)
    (run / 'lammps' / 'slide_p1gpa.in').rename(run / 'lammps' / 'slide.in')
    result = reconstruct(run, root)
    record = ProvenanceRecord.model_validate_json(result.wrote.read_text(encoding='utf-8'))
    assert record.thermostat_scheme == 'nvt_center'


def test_a_langevin_deck_is_recognised(tmp_path):
    root, run = _tree(tmp_path, 'rebomos', REBOMOS_REL)
    deck = run / 'lammps' / 'slide_p1gpa.in'
    deck.write_text(deck.read_text(encoding='utf-8') + '\nfix lg all langevin 300 300 0.1 42\n',
                    encoding='utf-8')
    result = reconstruct(run, root)
    record = ProvenanceRecord.model_validate_json(result.wrote.read_text(encoding='utf-8'))
    assert record.thermostat_scheme == 'langevin_center_nve'


def test_a_sync_conflict_copy_is_not_a_legacy_run(tmp_path):
    root, run = _tree(tmp_path, 'rebomos', REBOMOS_REL)
    conflict = root / 'sheetonsheet' / 'h-MoS2' / '100x_100y' / 'K300.sync-conflict-20260301'
    shutil.copytree(run, conflict)
    assert find_legacy_runs(root) == [run]


@pytest.mark.parametrize('fixture, rel', [('rebomos', REBOMOS_REL), ('deepmd', DEEPMD_REL)])
def test_dry_run_writes_and_moves_nothing(tmp_path, fixture, rel):
    root, run = _tree(tmp_path, fixture, rel)
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))

    result = reconstruct(run, root, dry_run=True)

    assert result.wrote is not None            # reports what it would write ...
    assert not result.wrote.exists()           # ... but writes nothing
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob('*')) == before
