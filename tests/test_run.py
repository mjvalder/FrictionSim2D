"""Tests for run-level sweep expansion and orchestration utilities."""

from pathlib import Path

import pytest

from src.core.config import load_settings as real_load_settings
from src.run.expand import _validate_runtime_sweep_ordering, expand_config_sweeps, resolve_lj_override_file
from src.run.hpc_dispatch import _build_hpc_manifest_entries, collect_hpc_simulation_paths
from src.run.naming import output_dir_for
from src.run.simulate import run_simulations

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


def test_output_dir_for_matches_run_layout() -> None:
    """A pure extraction of the naming block in run_simulations -- the path
    it returns (relative to the simulation root) must match what the AFM
    dispatch test above observes run_simulations actually build."""
    run_dict = {
        '2D': {'mat': 'h_MoS2', 'x': 100, 'y': 100},
        'general': {'temp': 300},
        'tip': {'mat': 'Si', 'amorph': 'a', 'r': 25},
        'sub': {'mat': 'Si', 'amorph': 'a'},
    }

    path = output_dir_for(run_dict, 'afm')

    assert str(path) == 'afm/h_MoS2/100x_100y/sub_aSi_tip_aSi_r25/K300'


def test_output_dir_for_sheet_like_model_has_no_tip_or_sub_segment() -> None:
    """Sheet-on-sheet layouts skip the tip/substrate naming segment entirely."""
    run_dict = {
        '2D': {'mat': 'h_MoS2', 'x': 100, 'y': 100},
        'general': {'temp': 300},
    }

    path = output_dir_for(run_dict, 'sheetonsheet')

    assert str(path) == 'sheetonsheet/h_MoS2/100x_100y/K300'


def test_expand_config_sweeps_does_not_expand_2d_layers() -> None:
    """Layer lists are not expanded in run-level sweeps."""
    base_config = {
        "general": {
            "temp": 300.0,
            "pressure": [1.0, 2.0],
        },
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": "dummy.sw",
            "cif_path": "dummy.cif",
            "x": 20.0,
            "y": 20.0,
            "layers": [3, 4],
        },
    }

    expanded = expand_config_sweeps(base_config)

    assert len(expanded) == 1
    assert expanded[0]["2D"]["layers"] == [3, 4]


def test_runtime_validation_rejects_unsorted_pressure() -> None:
    """Runtime validation should reject non-ascending pressure lists."""
    with pytest.raises(ValueError, match="pressure must be in ascending order"):
        _validate_runtime_sweep_ordering({"pressure": [1.0, 0.1]})


def test_runtime_validation_rejects_unsorted_scan_angle_force() -> None:
    """Runtime validation should reject non-ascending selector lists."""
    with pytest.raises(ValueError, match="scan_angle_force must be in ascending order"):
        _validate_runtime_sweep_ordering(
            {"pressure": [0.1, 1.0, 10.0], "scan_angle_force": [10.0, 0.1]}
        )


def test_runtime_validation_rejects_selector_not_in_target_order() -> None:
    """Selector values must follow pressure/force order as a subsequence."""
    with pytest.raises(ValueError, match="scan_angle_force must follow the same order as pressure values"):
        _validate_runtime_sweep_ordering(
            {"pressure": [0.1, 1.0, 10.0], "scan_angle_force": [0.1, 5.0]}
        )


class TestRoutingViewMatchesTheModel:
    """The view exists to answer routing questions exactly as the validated
    config would. `.ini` values arrive as the strings configparser produced,
    so the view has to coerce them the way pydantic does -- `finite_sheet =
    false` is the string 'false', which is truthy."""

    @staticmethod
    def _hetero_afm_raw(finite_sheet):
        return {
            'general': {'temp': 300, 'finite_sheet': finite_sheet},
            'tip': {'mat': 'Si', 'r': 25},
            'sub': {'mat': 'Si', 'amorph': 'a'},
            '2D': {'mat': 'h-MoS2'},
            '2D-1': {'mat': 'h-MoS2'},
            '2D-2': {'mat': 'h-WS2'},
        }

    def test_a_false_string_is_not_a_finite_sheet(self):
        from src.builders.hetero_afm import HeteroAFMSimulation
        from src.run.expand import _builder_cls_for_count

        assert _builder_cls_for_count(
            self._hetero_afm_raw('false'), 'afm') is HeteroAFMSimulation

    def test_a_true_string_still_routes_to_the_finite_builder(self):
        from src.builders.hetero_afm import FiniteHeteroAFMSimulation
        from src.run.expand import _builder_cls_for_count

        assert _builder_cls_for_count(
            self._hetero_afm_raw('true'), 'afm') is FiniteHeteroAFMSimulation

    def test_a_real_bool_is_unaffected(self):
        from src.builders.hetero_afm import (
            FiniteHeteroAFMSimulation, HeteroAFMSimulation,
        )
        from src.run.expand import _builder_cls_for_count

        assert _builder_cls_for_count(
            self._hetero_afm_raw(False), 'afm') is HeteroAFMSimulation
        assert _builder_cls_for_count(
            self._hetero_afm_raw(True), 'afm') is FiniteHeteroAFMSimulation

    def test_a_blank_twist_angle_is_not_a_twist(self):
        """An empty `.ini` key reads as None, but a blank string reaching the
        view (a half-typed GUI field, a JSON config) must not route a
        single-material sheet config to the hetero builder."""
        from src.builders.sheetonsheet import SheetOnSheetSimulation
        from src.run.expand import _builder_cls_for_count

        raw = {'general': {'temp': 300, 'twist_angle': ''}, '2D': {'mat': 'h-MoS2'}}
        assert _builder_cls_for_count(raw, 'sheetonsheet') is SheetOnSheetSimulation


class TestAngleSplitWithAForceGate:
    """``outer_loop='scan_angle'`` cannot carry ``scan_angle_force`` -- the
    split applies its one angle to every force, silently dropping the gate.

    But only where the split actually happens. Every other builder keeps the
    in-script angle loop, which honours the gate (hetero/slide.lmp and
    flake/slide.lmp really do gate on it), so the same config is correct
    there and must still build. The check therefore has to know which builder
    the config routes to -- a rule stated without that knowledge rejects
    working configs.
    """

    @staticmethod
    def _write(raw, tmp_path, name):
        from src.core.config_io import write_config
        raw['general']['scan_angle'] = [0, 30]
        raw['general']['scan_angle_force'] = 30
        raw['general']['outer_loop'] = 'scan_angle'
        return write_config(raw, tmp_path / name)

    def test_rejected_before_anything_is_built(self, tmp_path, monkeypatch):
        """The AFM slide really does split, so this one is the conflict --
        and it must be refused before the geometry build, not from
        write_inputs with a half-written tree already on disk."""
        from src.core.config import parse_config

        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'afm_flake_config.ini')
        config_path = self._write(raw, tmp_path, 'afm_gate.ini')

        with pytest.raises(ValueError, match='scan_angle_force'):
            run_simulations(config_file=str(config_path), model='afm',
                            settings_file=None, output_root=tmp_path / 'out',
                            generate_hpc=False)

        assert not (tmp_path / 'out').exists()

    def test_the_indent_still_builds(self, tmp_path, monkeypatch):
        """IndentSimulation never splits by angle (SPLIT_OUTER_LOOPS), so the
        combination is inert there -- it built before and must keep building."""
        from src.core.config import parse_config

        monkeypatch.chdir(REPO_ROOT)
        raw = parse_config(EXAMPLES / 'indent_config.ini')
        config_path = self._write(raw, tmp_path, 'indent_gate.ini')

        created, _, _, _ = run_simulations(
            config_file=str(config_path), model='indent', settings_file=None,
            output_root=tmp_path / 'out', generate_hpc=False,
        )
        assert len(created) == 1

    def test_a_hetero_sheet_config_still_builds(self, tmp_path, monkeypatch):
        """The hetero slide ignores outer_loop and honours the gate in-script,
        so a stray outer_loop next to a real gate must not fail the build."""
        from src.core.config import parse_config  # noqa: F401  (parity with above)
        from src.core.config_io import write_config

        monkeypatch.chdir(REPO_ROOT)
        sheet_1 = {
            'mat': 'h-MoS2', 'cif_path': 'examples/materials/h-MoS2.cif',
            'pot_path': 'examples/potentials/sw/MoS2_wen.sw', 'pot_type': 'sw',
            'x': 12.0, 'y': 12.0, 'layers': [2],
        }
        raw = {
            '2D': dict(sheet_1),
            '2D-1': dict(sheet_1),
            '2D-2': {
                'mat': 'h-WS2', 'cif_path': 'examples/materials/h-WS2.cif',
                'pot_path': 'examples/potentials/sw/sw_lammps/t-WS2.sw',
                'pot_type': 'sw', 'x': 12.0, 'y': 12.0, 'layers': [2],
            },
            'general': {
                'temp': 300, 'scan_speed': 1, 'pressure': [0.1, 30],
                'scan_angle': [0, 30], 'scan_angle_force': 30,
                'outer_loop': 'scan_angle',
            },
        }
        config_path = write_config(raw, tmp_path / 'hetero_gate.ini')

        created, _, _, _ = run_simulations(
            config_file=str(config_path), model='sheetonsheet', settings_file=None,
            output_root=tmp_path / 'out', generate_hpc=False,
        )
        assert len(created) == 1


def test_expand_config_sweeps_expands_general_non_lammps_lists() -> None:
    """General non-LAMMPS-loop list parameters should expand combinations."""
    base_config = {
        "general": {
            "temp": [300.0, 350.0],
            "scan_speed": [1.0, 2.0],
        },
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": "dummy.sw",
            "cif_path": "dummy.cif",
            "x": 20.0,
            "y": 20.0,
            "layers": [3],
        },
    }

    expanded = expand_config_sweeps(base_config)

    # Only temp expands; scan_speed is intentionally handled at LAMMPS level.
    assert len(expanded) == 2
    temps = sorted(conf["general"]["temp"] for conf in expanded)
    assert temps == [300.0, 350.0]
    for conf in expanded:
        assert conf["general"]["scan_speed"] == [1.0, 2.0]


class TestOrderedPairSweeps:
    """A materials_list of ordered pairs ('mat1 + mat2') drives both the
    primary sheet and '2D-2' -- one heterostructure combination per line,
    never sorted or deduped across order (h-MoS2+h-WS2 != h-WS2+h-MoS2)."""

    def test_pair_line_requires_second_section(self, tmp_path: Path) -> None:
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\n')
        config = {'2D': {'mat': '{mat}', 'cif_path': 'c/{mat}.cif',
                         'pot_path': 'p/{mat}.sw', 'materials_list': str(f)},
                  'general': {}}
        with pytest.raises(ValueError, match='2D-2'):
            expand_config_sweeps(config, base_dir=tmp_path)

    def test_single_name_line_rejected_when_config_expects_a_pair(
            self, tmp_path: Path) -> None:
        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2\n')
        config = {
            '2D-1': {'mat': '{mat1}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw', 'materials_list': str(f)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw'},
            'general': {},
        }
        with pytest.raises(ValueError, match='2D-2'):
            expand_config_sweeps(config, base_dir=tmp_path)

    def test_pair_sweep_expands_both_sections(self, tmp_path: Path) -> None:
        # Real files for both members of both pair orderings, so the preflight
        # that gates a material sweep resolves everything -- the same
        # real-file fixture pattern TestExpansionSkipsUnusableMaterials uses,
        # just for two sections instead of one.
        (tmp_path / 'c').mkdir()
        (tmp_path / 'p').mkdir()
        for mat in ('h-MoS2', 'h-WS2'):
            (tmp_path / 'c' / f'{mat}.cif').write_text('', encoding='utf-8')
            (tmp_path / 'p' / f'{mat}.sw').write_text('', encoding='utf-8')

        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\nh-WS2 + h-MoS2\n')
        config = {
            '2D-1': {'mat': '{mat1}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw', 'materials_list': str(f)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw'},
            'general': {},
        }
        out = expand_config_sweeps(config, base_dir=tmp_path)
        assert len(out) == 2
        assert out[0]['2D-1']['mat'] == 'h-MoS2' and out[0]['2D-2']['mat'] == 'h-WS2'
        assert out[1]['2D-1']['mat'] == 'h-WS2' and out[1]['2D-2']['mat'] == 'h-MoS2'
        assert 'materials_list' not in out[0]['2D-1']
        assert out[0]['2D-1']['cif_path'] == 'c/h-MoS2.cif'
        assert out[0]['2D-2']['cif_path'] == 'c/h-WS2.cif'

    def test_pair_line_requires_a_template_in_the_second_section(
            self, tmp_path: Path) -> None:
        """A literal [2D-2] against a pair list builds the SAME second material
        for every member, under a directory named after the pair the list asked
        for -- wrong results that look plausible and do not collide. The files
        it names exist, so the preflight is happy and validation is clean; only
        the missing placeholder gives it away."""
        (tmp_path / 'c').mkdir()
        (tmp_path / 'p').mkdir()
        for mat in ('h-MoS2', 'h-MoSe2', 'h-WS2', 'h-WSe2'):
            (tmp_path / 'c' / f'{mat}.cif').write_text('', encoding='utf-8')
            (tmp_path / 'p' / f'{mat}.sw').write_text('', encoding='utf-8')

        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\nh-MoSe2 + h-WSe2\n')
        config = {
            '2D': {'mat': '{mat}', 'cif_path': 'c/{mat}.cif',
                   'pot_path': 'p/{mat}.sw', 'materials_list': str(f)},
            '2D-2': {'mat': 'h-WS2', 'cif_path': 'c/h-WS2.cif',
                     'pot_path': 'p/h-WS2.sw'},
            'general': {},
        }
        with pytest.raises(ValueError, match=r'\[2D-2\].*\{mat2\}'):
            expand_config_sweeps(config, base_dir=tmp_path)

    def test_pair_line_requires_a_template_in_the_primary_section(
            self, tmp_path: Path) -> None:
        """The mirror hole: a literal primary sheet pins mat1 across the sweep
        while [2D-2] varies, so the runs differ in directory name but every one
        of them builds the same bottom material."""
        (tmp_path / 'c').mkdir()
        (tmp_path / 'p').mkdir()
        for mat in ('h-MoS2', 'h-MoSe2', 'h-WS2', 'h-WSe2'):
            (tmp_path / 'c' / f'{mat}.cif').write_text('', encoding='utf-8')
            (tmp_path / 'p' / f'{mat}.sw').write_text('', encoding='utf-8')

        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2 + h-WS2\nh-MoSe2 + h-WSe2\n')
        config = {
            '2D-1': {'mat': 'h-MoS2', 'cif_path': 'c/h-MoS2.cif',
                     'pot_path': 'p/h-MoS2.sw', 'materials_list': str(f)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw'},
            'general': {},
        }
        with pytest.raises(ValueError, match=r'\[2D-1\].*\{mat1\}'):
            expand_config_sweeps(config, base_dir=tmp_path)

    @staticmethod
    def _pair_files(tmp_path: Path) -> Path:
        """Real files for every member, so the preflight resolves and only the
        template check can be what refuses the config."""
        (tmp_path / 'c').mkdir()
        (tmp_path / 'p').mkdir()
        for mat in ('h-MoS2', 'h-MoSe2', 'h-WS2', 'h-WSe2'):
            (tmp_path / 'c' / f'{mat}.cif').write_text('', encoding='utf-8')
            (tmp_path / 'p' / f'{mat}.sw').write_text('', encoding='utf-8')
        listing = tmp_path / 'm.txt'
        listing.write_text('h-MoS2 + h-WS2\nh-MoSe2 + h-WSe2\n')
        return listing

    def test_second_section_pasted_from_the_first_is_refused(
            self, tmp_path: Path) -> None:
        """The most plausible edit in the whole format: copy the [2D-1] block,
        paste it as [2D-2], change `mat` and forget cif_path/pot_path. Both
        sheets then load the FIRST material's files -- a homostructure under a
        correctly named heterostructure directory. The section does carry
        '{mat2}', so presence alone cannot catch this; the wrong marker has to
        be refused wherever it appears."""
        listing = self._pair_files(tmp_path)
        config = {
            '2D-1': {'mat': '{mat1}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw', 'materials_list': str(listing)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw'},
            'general': {},
        }
        with pytest.raises(ValueError) as excinfo:
            expand_config_sweeps(config, base_dir=tmp_path)
        message = str(excinfo.value)
        assert '[2D-2]' in message and "'{mat1}'" in message
        assert "'{mat2}'" in message      # says which marker belongs there

    def test_second_section_templated_on_the_first_marker_is_refused(
            self, tmp_path: Path) -> None:
        listing = self._pair_files(tmp_path)
        config = {
            '2D-1': {'mat': '{mat1}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw', 'materials_list': str(listing)},
            '2D-2': {'mat': '{mat1}', 'cif_path': 'c/{mat1}.cif',
                     'pot_path': 'p/{mat1}.sw'},
            'general': {},
        }
        with pytest.raises(ValueError, match=r"\[2D-2\].*'\{mat2\}'"):
            expand_config_sweeps(config, base_dir=tmp_path)

    def test_primary_section_templated_on_the_second_marker_is_refused(
            self, tmp_path: Path) -> None:
        listing = self._pair_files(tmp_path)
        config = {
            '2D-1': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw', 'materials_list': str(listing)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw'},
            'general': {},
        }
        with pytest.raises(ValueError, match=r"\[2D-1\].*'\{mat1\}'"):
            expand_config_sweeps(config, base_dir=tmp_path)

    def test_the_primary_may_still_spell_its_marker_mat(
            self, tmp_path: Path) -> None:
        """`{mat}` and `{mat1}` are the same placeholder (material_templates),
        so a single-material config that later gains a second sheet must not
        have to be rewritten to keep sweeping."""
        listing = self._pair_files(tmp_path)
        config = {
            '2D-1': {'mat': '{mat}', 'cif_path': 'c/{mat}.cif',
                     'pot_path': 'p/{mat}.sw', 'materials_list': str(listing)},
            '2D-2': {'mat': '{mat2}', 'cif_path': 'c/{mat2}.cif',
                     'pot_path': 'p/{mat2}.sw'},
            'general': {},
        }
        out = expand_config_sweeps(config, base_dir=tmp_path)
        assert [(c['2D-1']['mat'], c['2D-2']['mat']) for c in out] == [
            ('h-MoS2', 'h-WS2'), ('h-MoSe2', 'h-WSe2')]
        assert [c['2D-1']['cif_path'] for c in out] == [
            'c/h-MoS2.cif', 'c/h-MoSe2.cif']
        assert [c['2D-2']['cif_path'] for c in out] == [
            'c/h-WS2.cif', 'c/h-WSe2.cif']

    def test_a_single_material_sweep_needs_no_pair_template(
            self, tmp_path: Path) -> None:
        """The installed base: a one-name-per-line list with no [2D-2] is not
        a pair sweep, so the pair-template rule must never fire on it."""
        (tmp_path / 'c').mkdir()
        (tmp_path / 'p').mkdir()
        for mat in ('h-MoS2', 'h-WS2'):
            (tmp_path / 'c' / f'{mat}.cif').write_text('', encoding='utf-8')
            (tmp_path / 'p' / f'{mat}.sw').write_text('', encoding='utf-8')

        f = tmp_path / 'm.txt'
        f.write_text('h-MoS2\nh-WS2\n')
        config = {
            '2D': {'mat': '{mat}', 'cif_path': 'c/{mat}.cif',
                   'pot_path': 'p/{mat}.sw', 'materials_list': str(f)},
            'general': {},
        }
        out = expand_config_sweeps(config, base_dir=tmp_path)
        assert [conf['2D']['mat'] for conf in out] == ['h-MoS2', 'h-WS2']

    def test_the_shipped_pair_sweep_still_expands(self, monkeypatch) -> None:
        """examples/hetero_pair_sweep.ini is the correctly-templated reference:
        three distinct ordered pairs, unaffected by the guard."""
        from src.core.config import parse_config

        monkeypatch.chdir(REPO_ROOT)
        out = expand_config_sweeps(parse_config(EXAMPLES / 'hetero_pair_sweep.ini'))

        assert [(c['2D-1']['mat'], c['2D-2']['mat']) for c in out] == [
            ('h-MoS2', 'h-WS2'), ('h-MoSe2', 'h-WSe2'), ('h-MoTe2', 'h-WTe2')]

    def test_numbered_only_primary_still_uses_the_2d_note_prefix(
            self, tmp_path: Path) -> None:
        """The GUI filters expansion notes on the literal '[2D] materials_list'
        prefix (app.py); a numbered-only primary section must not change it."""
        from src.run.expand import count_expansion

        listing = tmp_path / 'mats.csv'
        listing.write_text('h-MoS2\n', encoding='utf-8')
        config = {
            '2D-1': {'mat': '{mat1}', 'materials_list': str(listing)},
            'general': {},
        }
        summary = count_expansion(config, 'sheetonsheet')
        assert any(n.startswith('[2D] materials_list') for n in summary.notes)


def test_lj_file_merges_inline_wins(tmp_path: Path) -> None:
    """A shared overrides file fills in pairs the section did not already
    name; an inline entry with the same key is left untouched."""
    lj_file = tmp_path / 'lj.txt'
    lj_file.write_text('# master table\nMo-S  0.4124  3.75114\nS-S, 0.198, 3.62\n')
    section = {'file': str(lj_file), 'Mo-S': [9.9, 9.9]}

    out = resolve_lj_override_file(section, tmp_path)

    assert out['Mo-S'] == [9.9, 9.9]          # inline wins
    assert out['S-S'] == [0.198, 3.62]
    assert 'file' not in out


def test_lj_file_missing_is_skipped_silently(tmp_path: Path) -> None:
    """An unreadable/missing shared file is advisory, not fatal -- same
    tolerance as materials_list."""
    section = {'file': str(tmp_path / 'does-not-exist.txt'), 'Mo-S': [1.0, 2.0]}

    out = resolve_lj_override_file(section, tmp_path)

    assert out == {'Mo-S': [1.0, 2.0]}


def test_expansion_inlines_lj_file_on_the_no_sweep_path(tmp_path: Path) -> None:
    """expand_config_sweeps' early `return [base_config]` -- taken whenever
    nothing else is swept -- is the common case, so the [lj_override] file
    has to be resolved there too, not only on the itertools.product path
    below it."""
    lj_file = tmp_path / 'lj.txt'
    lj_file.write_text('Mo-S  0.4124  3.75114\n')
    base_config = {
        "general": {"temp": 300.0},
        "2D": {
            "mat": "h-MoS2", "pot_type": "sw",
            "pot_path": "dummy.sw", "cif_path": "dummy.cif",
            "x": 20.0, "y": 20.0,
        },
        "lj_override": {"file": str(lj_file)},
    }

    expanded = expand_config_sweeps(base_config, base_dir=tmp_path)

    assert len(expanded) == 1
    assert expanded[0]["lj_override"] == {"Mo-S": [0.4124, 3.75114]}
    assert "file" not in expanded[0]["lj_override"]


def test_expansion_inlines_lj_file_on_the_swept_path(tmp_path: Path) -> None:
    """The same file resolution also has to apply to every config produced
    by the itertools.product expansion, not just the no-sweep case."""
    lj_file = tmp_path / 'lj.txt'
    lj_file.write_text('Mo-S  0.4124  3.75114\n')
    base_config = {
        "general": {"temp": [300.0, 350.0]},
        "2D": {
            "mat": "h-MoS2", "pot_type": "sw",
            "pot_path": "dummy.sw", "cif_path": "dummy.cif",
            "x": 20.0, "y": 20.0,
        },
        "lj_override": {"file": str(lj_file)},
    }

    expanded = expand_config_sweeps(base_config, base_dir=tmp_path)

    assert len(expanded) == 2
    for conf in expanded:
        assert conf["lj_override"] == {"Mo-S": [0.4124, 3.75114]}


def test_collect_hpc_simulation_paths_detects_valid_inputs(tmp_path: Path) -> None:
    """Collect only simulation dirs with recognized LAMMPS input scripts."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "system.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "slide_custom.in").write_text("", encoding="utf-8")

    (tmp_path / "simC" / "lammps").mkdir(parents=True)
    (tmp_path / "simC" / "lammps" / "notes.txt").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path)

    assert paths == ["simA", "simB"]


def test_collect_hpc_simulation_paths_orders_layers_numerically(tmp_path: Path) -> None:
    """Layer-like paths should use numeric order (L2 before L10)."""
    for layer in ("L1", "L2", "L10"):
        (tmp_path / layer / "lammps").mkdir(parents=True)
        (tmp_path / layer / "lammps" / "slide.in").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path)

    assert paths == ["L1", "L2", "L10"]


def test_collect_hpc_simulation_paths_accepts_any_in_when_scripts_empty(tmp_path: Path) -> None:
    """An empty script filter should include dirs containing any .in file."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "custom_input.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "notes.txt").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path, lammps_scripts=[])

    assert paths == ["simA"]


def test_build_hpc_manifest_entries_prefers_slide_when_available(tmp_path: Path) -> None:
    """Manifest should include slide scripts when present, else system scripts."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "system.in").write_text("", encoding="utf-8")
    (tmp_path / "simA" / "lammps" / "slide_p1.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "system_init.in").write_text("", encoding="utf-8")

    entries, resolved = _build_hpc_manifest_entries(
        tmp_path,
        ["simA", "simB"],
        ["system.in", "slide.in"],
    )

    assert entries == [
        "simA/lammps/slide_p1.in",
        "simB/lammps/system_init.in",
    ]
    assert resolved == ["__MANIFEST_PATH__"]


def test_build_hpc_manifest_entries_falls_back_when_no_scripts(tmp_path: Path) -> None:
    """When no scripts are discovered, preserve original simulation paths/scripts."""
    (tmp_path / "simA").mkdir(parents=True)

    entries, resolved = _build_hpc_manifest_entries(
        tmp_path,
        ["simA"],
        ["slide.in"],
    )

    assert entries == ["simA"]
    assert resolved == ["slide.in"]


def test_run_simulations_dispatches_afm_and_generates_hpc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """run_simulations should instantiate AFM builder and call HPC generation."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
        "tip": {"mat": "Si", "amorph": "c", "r": 25},
        "sub": {"mat": "Si", "amorph": "a"},
    }

    settings = real_load_settings()
    settings.hpc.lammps_scripts = ["slide.in"]

    monkeypatch.setattr("src.run.simulate.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.run.simulate.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.run.simulate.expand_config_sweeps", lambda _c: [dict(base_dict)])

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sub = kwargs.get('sub')

        def model_dump_json(self, indent=2):
            return "{}"

    calls = {"build": 0, "base": 0, "hpc": 0}
    seen_output_dirs: list[Path] = []

    class FakeAFMBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            seen_output_dirs.append(Path(_output_dir))
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            calls["base"] += 1

        def build(self):
            calls["build"] += 1

    monkeypatch.setattr("src.run.simulate.AFMSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.run.simulate.AFMSimulation", FakeAFMBuilder)
    monkeypatch.setattr(
        "src.run.simulate.generate_hpc_scripts_for_root",
        lambda _root, _settings: calls.__setitem__("hpc", calls["hpc"] + 1),
    )

    created, root, _configs, returned_settings = run_simulations(
        config_file=str(config_file),
        model="afm",
        output_root=tmp_path,
        generate_hpc=True,
        simulation_root_name="sim_run",
    )

    assert len(created) == 1
    assert root == tmp_path / "sim_run"
    assert calls == {"build": 1, "base": 1, "hpc": 1}
    assert seen_output_dirs[0].as_posix().endswith("afm/MoS2/20x_20y/sub_aSi_tip_Si_r25/K300")
    assert returned_settings.hpc.lammps_scripts == ["system.in", "slide.in"]


def test_run_simulations_sheetonsheet_adjusts_default_scripts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Sheet-on-sheet mode should default to slide-only script set."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
    }

    settings = real_load_settings()
    settings.hpc.lammps_scripts = ["system.in", "slide.in"]

    monkeypatch.setattr("src.run.simulate.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.run.simulate.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.run.simulate.expand_config_sweeps", lambda _c: [dict(base_dict)])

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump_json(self, indent=2):
            return "{}"

    class FakeSheetBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            pass

        def build(self):
            pass

    monkeypatch.setattr("src.run.simulate.SheetOnSheetSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.run.simulate.SheetOnSheetSimulation", FakeSheetBuilder)

    _created, _root, _configs, returned_settings = run_simulations(
        config_file=str(config_file),
        model="sheetonsheet",
        output_root=tmp_path,
        simulation_root_name="sim_run",
    )

    assert returned_settings.hpc.lammps_scripts == ["slide.in"]


def test_run_simulations_continues_after_build_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A failed build should be logged and subsequent configs should still run."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
        "tip": {"mat": "Si", "amorph": "c", "r": 25},
        "sub": {"mat": "Si", "amorph": "a"},
    }

    settings = real_load_settings()
    monkeypatch.setattr("src.run.simulate.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.run.simulate.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr(
        "src.run.simulate.expand_config_sweeps",
        lambda _c: [dict(base_dict), dict(base_dict)],
    )

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sub = kwargs.get('sub')

        def model_dump_json(self, indent=2):
            return "{}"

    call_counter = {"builds": 0}

    class FlakyAFMBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            pass

        def build(self):
            call_counter["builds"] += 1
            if call_counter["builds"] == 1:
                raise RuntimeError("boom")

    monkeypatch.setattr("src.run.simulate.AFMSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.run.simulate.AFMSimulation", FlakyAFMBuilder)

    created, _root, _configs, _returned_settings = run_simulations(
        config_file=str(config_file),
        model="afm",
        output_root=tmp_path,
        simulation_root_name="sim_run",
    )

    assert call_counter["builds"] == 2
    assert len(created) == 1


def test_run_simulations_raises_for_missing_config_file(tmp_path: Path) -> None:
    """Missing config file should raise immediately."""
    with pytest.raises(FileNotFoundError):
        run_simulations(
            config_file=str(tmp_path / "missing.ini"),
            model="afm",
            output_root=tmp_path,
        )


def test_pes_tip_no_substrate_selects_freestanding_builder():
    from src.run.simulate import _select_afm_builder_cls
    from src.builders.afm_freestanding import FreestandingPESTipSimulation
    from src.builders.pes_scan import PESTipSimulation

    class _Cfg:  # minimal stand-in with the only attribute the selector reads
        sub = None
    assert _select_afm_builder_cls('pes_tip', _Cfg()) is FreestandingPESTipSimulation

    class _CfgSub:
        sub = object()
    assert _select_afm_builder_cls('pes_tip', _CfgSub()) is PESTipSimulation


def test_afm_no_substrate_raises():
    from src.run.simulate import _select_afm_builder_cls
    import pytest
    class _Cfg:
        sub = None
    with pytest.raises(ValueError, match="substrate-free"):
        _select_afm_builder_cls('afm', _Cfg())


class TestMaterialListIsNotLostSilently:
    """A materials_list that resolves to nothing used to expand to one config
    with no message at all, so a 40-material sweep silently became one run."""

    def _config(self, path):
        return {
            '2D': {'mat': '{mat}', 'materials_list': str(path), 'layers': [1]},
            'general': {'temp': 300},
        }

    def test_a_non_txt_path_is_reported(self, tmp_path):
        from src.run.expand import count_expansion

        listing = tmp_path / 'mats.csv'
        listing.write_text('h-MoS2\n', encoding='utf-8')
        summary = count_expansion(self._config(listing), 'afm')

        assert summary.configs == 1
        assert any('.txt' in note for note in summary.notes)

    def test_a_txt_path_produces_no_note(self, tmp_path):
        from src.run.expand import count_expansion

        listing = tmp_path / 'mats.txt'
        listing.write_text('h-MoS2\n', encoding='utf-8')
        summary = count_expansion(self._config(listing), 'afm')

        assert not [n for n in summary.notes if '.txt' in n]
