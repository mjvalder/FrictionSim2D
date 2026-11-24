"""Tests for frictionsim2d.builders.components + the plain AFMSimulation orchestration.

Rewritten (Phase 2c review, docs/superpowers/reviews/builders-construction.md
(c), test-weak repair): every test here used to mock the module's own
internals directly and assert only "a mock was called" -- a defect inside the
mocked helper itself would never be caught. These now build real structures
(real CIF + potential fixtures from examples/) into tmp_path and assert on
the produced LAMMPS data / rendered template content. Only the genuine
expensive boundaries stay mocked: ``run_lammps_commands`` (a real LAMMPS
invocation) and ``make_amorphous`` (a real melt-quench) -- both already
allowed by tests/repo/test_mock_targets.py's ALLOWED_TARGET_PREFIXES.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from frictionsim2d.builders import AFMSimulation as ExportedAFMSimulation
from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.builders import SheetOnSheetSimulation as ExportedSheetOnSheetSimulation
from frictionsim2d.builders import components
from frictionsim2d.builders import components as exported_components
from frictionsim2d.builders.construction.components import cifread, get_amorphous_path
from frictionsim2d.builders.utils.lammps_data import get_model_dimensions
from frictionsim2d.core.config import AFMSimulationConfig, SheetConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
SI_CIF, SI_POT = f"{MAT}/Si.cif", f"{POT}/Si.sw"
MOS2_CIF, MOS2_POT = f"{MAT}/h-MoS2.cif", f"{POT}/MoS2_wen.sw"


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _afm_config(build_dir: Path) -> AFMSimulationConfig:
    """A small, real, non-hetero AFM config (real Si tip/sub, real MoS2 sheet).

    x/y=30, r=5 clears validate_tip_sheet_footprint's 4*r=20 floor with
    margin. sub.amorph='a' is required by validate_non_finite_sheet_substrate
    for a non-finite-sheet run; make_amorphous is mocked (a genuine, allowed
    boundary -- see module docstring) rather than actually quenched.
    """
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {
            'mat': 'Si', 'pot_type': 'sw', 'pot_path': SI_POT,
            'cif_path': SI_CIF, 'r': 5.0, 'amorph': 'c', 'dspring': 0.0, 's': 1.0,
        },
        'sub': {
            'mat': 'Si', 'pot_type': 'sw', 'pot_path': SI_POT,
            'cif_path': SI_CIF, 'thickness': 10.0, 'amorph': 'a',
        },
        '2D': {
            'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': MOS2_POT,
            'cif_path': MOS2_CIF, 'x': 30.0, 'y': 30.0, 'layers': [1],
        },
        'lj_override': {},
        'settings': load_settings().model_dump(),
    }
    return AFMSimulationConfig(**data)


@pytest.fixture
def afm_config(temp_dir: Path) -> AFMSimulationConfig:
    return _afm_config(temp_dir)


def _fake_run_lammps_commands(captured: dict):
    """Real ``run_lammps_commands`` boundary stub.

    Reads the REAL base slab (built for real by ``_create_base_slab`` via
    ``create_orthogonal_slab`` -- no atomsk, no LAMMPS) that the rendered
    ``read_data``/``write_data`` commands name, records its real dimensions
    and atom count before the caller unlinks it, then copies it to the
    requested output path so the caller completes normally without needing
    a real ``lmp`` binary for the region-carve/crop step itself.
    """
    def _run(cmds):
        captured['commands'] = list(cmds)
        read_line = next(c for c in cmds if c.startswith('read_data'))
        write_line = next(c for c in cmds if c.startswith('write_data'))
        # First whitespace-separated token only: some templates (e.g.
        # apply_langevin_regions.lmp) append extra read_data args
        # (`extra/atom/types N`) after the path.
        base_path = Path(read_line.split()[1])
        out_path = Path(write_line.split()[1])
        captured['base_dims'] = components.get_model_dimensions(base_path)
        captured['base_text'] = base_path.read_text(encoding='utf-8')
        out_path.write_text(captured['base_text'], encoding='utf-8')
    return _run


def test_component_build_tip(afm_config, temp_dir):
    """build_tip builds a real base slab (real Si CIF, no atomsk/LAMMPS) and
    only the final sphere-carve step (run_lammps_commands) is mocked."""
    settings = afm_config.settings
    captured: dict = {}

    with patch("frictionsim2d.builders.components.run_lammps_commands",
               side_effect=_fake_run_lammps_commands(captured)) as mock_lmp:
        path, radius = components.build_tip(afm_config.tip, temp_dir, settings)

    assert path == temp_dir / "tip.lmp"
    assert path.exists()
    mock_lmp.assert_called_once()

    # Real geometry: a real Si base slab was built (create_orthogonal_slab
    # rounds to the nearest whole-unit-cell replication, so only check it
    # produced a real, positive-sized box -- not an exact target match).
    base_dims = captured['base_dims']
    x_span = base_dims['xhi'] - base_dims['xlo']
    y_span = base_dims['yhi'] - base_dims['ylo']
    assert x_span > 0
    assert y_span > 0
    # Real atoms were written (not a placeholder string).
    assert "Atoms" in captured['base_text']

    # The crystalline tip recomputes its radius from the real built slab
    # (component.py:build_tip), so the returned value must match exactly.
    assert radius == pytest.approx(min(x_span, y_span) / 2.0)
    # And the rendered LAMMPS commands must carry that same real radius,
    # not a hardcoded/mocked one.
    sphere_line = next(c for c in captured['commands'] if c.startswith('region afm_tip'))
    assert f"sphere 0 0 {radius} {radius}" in sphere_line


def test_build_sheet_copies_base_when_multilayer_stacking_disabled(tmp_path: Path) -> None:
    """build_sheet should return the real single-layer build unstacked when
    stack_if_multi=False, even though config.layers names more than one layer
    -- stack_multilayer_sheet (spied, not faked) must never run."""
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True)
    config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[1, 2],
    )

    with patch("frictionsim2d.builders.components.stack_multilayer_sheet",
               wraps=components.stack_multilayer_sheet) as mock_stack:
        out_path, out_dims, out_lat_c = components.build_sheet(
            config,
            build_dir=build_dir,
            stack_if_multi=False,
        )

    assert not mock_stack.called
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Atoms" in text
    assert out_dims['xhi'] - out_dims['xlo'] > 0
    # A single (unstacked) MoS2 layer: 2 atom types (Mo, S), never doubled by
    # a (skipped) stacking pass.
    assert components.get_num_atom_types(out_path) == 2
    assert out_lat_c is None or isinstance(out_lat_c, float)


def test_builders_package_exports():
    """Package-level builder exports should preserve the public API."""
    assert ExportedAFMSimulation is AFMSimulation
    assert ExportedSheetOnSheetSimulation.__name__ == "SheetOnSheetSimulation"
    assert exported_components is components


def test_afm_builder_structure(afm_config, temp_dir):
    """AFMSimulation.build() produces a real sheet/tip/substrate + rendered
    LAMMPS inputs for a one-layer config; only run_lammps_commands (the
    tip/substrate region-carve LAMMPS call) and make_amorphous (the
    substrate melt-quench) are mocked -- both genuine, allowed boundaries.
    Everything else (real CIF reads, potential registration, z-position
    calc, HPC script generation, template rendering) runs for real."""
    afm_config.settings.hpc.modules = ["dummy-module"]
    captured: dict = {}

    def _fake_make_amorphous(mat, cif_path, target_x, target_y, target_z,
                              pot_path, pot_type, output_dir, settings,
                              allow_generate=True, **_kwargs):
        # Stand-in "amorphous" substrate: a real (crystalline) orthogonal
        # slab is good enough here -- this test is not about melt-quench
        # fidelity (test_amorphous_guard.py owns that), only about the
        # build() orchestration around a real substrate file.
        out_path = Path(output_dir) / f"{mat}_amor.lmp"
        components.create_orthogonal_slab(
            Path(cif_path), out_path, target_x=target_x, target_y=target_y,
            target_z=target_z,
        )
        return out_path

    with patch("frictionsim2d.builders.components.run_lammps_commands",
               side_effect=_fake_run_lammps_commands(captured)), \
         patch("frictionsim2d.builders.construction.components.make_amorphous",
               side_effect=_fake_make_amorphous):
        builder = AFMSimulation(afm_config, output_dir=str(temp_dir))
        builder.build()

    assert 1 in builder.output_dir_layer
    layer_dir = builder.output_dir_layer[1]
    assert layer_dir.exists()

    # Real sheet: a genuine h-MoS2 monolayer was built and written.
    sheet_path = builder.sheet_paths[1]
    assert sheet_path.exists()
    sheet_dims = components.get_model_dimensions(sheet_path)
    assert sheet_dims['xhi'] - sheet_dims['xlo'] > 0

    # Real tip/substrate files landed (content stood in by the mocked
    # boundaries above, but the paths and the surrounding orchestration --
    # potential registration, z-positions, template rendering -- are real).
    assert builder.tip_path.exists()
    assert builder.sub_path.exists()

    lammps_dir = layer_dir / "lammps"
    assert lammps_dir.exists()
    system_in = (lammps_dir / "system.in").read_text(encoding="utf-8")
    settings_in = (lammps_dir / "system.in.settings").read_text(encoding="utf-8")
    slide_in = (lammps_dir / "slide.in").read_text(encoding="utf-8")
    # Real rendered content, not an empty/placeholder file: the real
    # PotentialManager registered real sheet/tip/sub atom types.
    assert "atom_style" in system_in
    assert "pair_style" in settings_in
    assert "group" in slide_in

    # Real HPC script generation (settings.hpc.modules set above so
    # _generate_hpc_scripts doesn't refuse): at least one real job script was
    # written, and no manifest.json -- the script list is not a file.
    hpc_dir = builder._get_hpc_output_dir()
    assert not (hpc_dir / "manifest.json").exists()
    assert any(hpc_dir.glob("*.pbs")) or any(hpc_dir.glob("*.sh")) or any(hpc_dir.glob("*.slurm"))


def test_build_monolayer_avoids_atomsk_convert_orthogonalize_duplicate(tmp_path: Path) -> None:
    """build_monolayer builds a real h-MoS2 monolayer via the native
    (non-Atomsk) helpers, called exactly once each -- a regression guard
    for the native pipeline, verified against real produced geometry."""
    config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=50.0, y=50.0, layers=[1],
    )

    with patch("frictionsim2d.builders.components._write_cif_as_lammps_atomic",
               wraps=components._write_cif_as_lammps_atomic) as mock_convert, \
         patch("frictionsim2d.builders.components._orthogonalize_lammps_data",
               wraps=components._orthogonalize_lammps_data) as mock_ortho, \
         patch("frictionsim2d.builders.components.replicate_data_file",
               wraps=components.replicate_data_file) as mock_dup:
        out_path, out_dims, _pot_counts, total_types, _supercell_dims = \
            components.build_monolayer(config)

    assert out_path.exists()
    assert out_dims['xhi'] - out_dims['xlo'] > 0
    assert total_types == 2   # Mo + S
    text = out_path.read_text(encoding="utf-8")
    assert "Atoms" in text and "Masses" in text
    mock_convert.assert_called_once()
    mock_ortho.assert_called_once()
    assert mock_dup.call_count == 1


def test_create_orthogonal_slab_avoids_atomsk_cli(tmp_path: Path) -> None:
    """create_orthogonal_slab builds a real orthogonal Si slab without going
    through the Atomsk CLI at all."""
    out_path = tmp_path / "out.lmp"

    with patch("frictionsim2d.builders.components._write_cif_as_lammps_atomic",
               wraps=components._write_cif_as_lammps_atomic) as mock_convert, \
         patch("frictionsim2d.builders.components.replicate_data_file",
               wraps=components.replicate_data_file) as mock_dup, \
         patch("frictionsim2d.builders.components._orthogonalize_lammps_data",
               wraps=components._orthogonalize_lammps_data) as mock_ortho:
        _, out_dims, duplication = components.create_orthogonal_slab(
            cif_path=Path(SI_CIF),
            output_path=out_path,
            # 12.0 rounds to the same single (2x2 predup) Si cell in x/y
            # (round(12.0 / 5.4437 / 2.0) == 1) and target_z=None skips z
            # replication -- so duplication == [1, 1, 1] and
            # replicate_data_file is called only for the initial 2x2
            # predup step (the "dup==1,1,1" copy branch skips a second call).
            target_x=12.0,
            target_y=12.0,
            target_z=None,
        )

    assert out_path.exists()
    assert out_dims['xhi'] - out_dims['xlo'] > 0
    assert out_dims['yhi'] - out_dims['ylo'] > 0
    assert duplication == [1, 1, 1]
    mock_convert.assert_called_once()
    mock_ortho.assert_called_once()
    assert mock_dup.call_count == 1


def test_build_monolayer_splits_edge_types_when_enabled(tmp_path: Path) -> None:
    """build_monolayer should double the sheet's atom types (core/edge) when
    edge typing is enabled, on a real built monolayer."""
    config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[1],
    )

    with patch("frictionsim2d.builders.components.apply_sheet_edge_types",
               wraps=components.apply_sheet_edge_types) as mock_edge_split:
        out_path, out_dims, _pot_counts, total_types, _supercell_dims = \
            components.build_monolayer(config, edge_mode='fixed', edge_width=4.0)

    assert out_path.exists()
    assert out_dims['xhi'] - out_dims['xlo'] > 0
    assert total_types == 4   # Mo/S core + Mo/S edge
    mock_edge_split.assert_called_once()


# ---------------------------------------------------------------------------
# The three biggest coverage gaps (builders-construction.md (c), findings 1-3)
# ---------------------------------------------------------------------------

def test_make_amorphous_manual_run_writes_real_script_and_raises(tmp_path):
    """Finding 1(a): settings.quench.run_local=False writes a real, non-empty
    melt-quench script (guards a silently-empty Jinja render) naming the real
    drawn melt_seed and the cif's real element order, then raises naming the
    manual-run script -- before anything is written to the package materials
    directory (the RuntimeError fires ahead of the metadata/amor_file write).
    A distinctive material name guarantees no cached hit short-circuits the
    melt-quench path and cannot collide with a real material."""
    settings = load_settings().model_copy(deep=True)
    settings.quench.run_local = False
    mat_name = "zz_test_phase2c_manual"

    with pytest.raises(RuntimeError, match=r"make_amorphous_zz_test_phase2c_manual\.in"):
        components.make_amorphous(
            mat_name, Path(SI_CIF),
            target_x=8.0, target_y=8.0, target_z=8.0,
            pot_path=Path(SI_POT), pot_type="sw",
            output_dir=tmp_path, settings=settings,
        )

    local_lmp_input = tmp_path / f"make_amorphous_{mat_name}.in"
    assert local_lmp_input.exists()
    text = local_lmp_input.read_text(encoding="utf-8")

    velocity_line = next(l for l in text.splitlines() if l.strip().startswith("velocity"))
    seed_token = velocity_line.split()[4]
    assert seed_token.isdigit() and 1 <= int(seed_token) <= 900_000_000

    elements = cifread(Path(SI_CIF))['elements']
    assert elements
    for element in elements:
        assert element in text

    # Nothing landed in the package materials directory.
    assert get_amorphous_path(mat_name) is None


def test_make_amorphous_run_local_success_writes_metadata_and_cleans_base_block(tmp_path):
    """Finding 1(b): settings.quench.run_local=True with only subprocess.run
    mocked (the real external-process boundary) -- a "successful" quench
    writes real metadata (melt_seed, quench_nsteps, pot_type) beside the
    amorphous block, and the crystal base_block is cleaned up afterward.
    This does write to the real package materials directory (make_amorphous's
    actual contract for a real caller), so it tidies up its own output."""
    settings = load_settings().model_copy(deep=True)
    settings.quench.run_local = True
    settings.quench.n_procs = 1
    mat_name = "zz_test_phase2c_auto"

    meta_path = None
    try:
        with patch("subprocess.run") as mock_run:
            amor_file = components.make_amorphous(
                mat_name, Path(SI_CIF),
                target_x=8.0, target_y=8.0, target_z=8.0,
                pot_path=Path(SI_POT), pot_type="sw",
                output_dir=tmp_path, settings=settings,
            )
        mock_run.assert_called_once()

        meta_path = Path(str(amor_file) + '.json')
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta['material'] == mat_name
        assert meta['pot_type'] == "sw"
        assert isinstance(meta['melt_seed'], int) and 1 <= meta['melt_seed'] <= 900_000_000
        assert meta['quench_nsteps'] >= 1

        base_block = tmp_path / f"{mat_name}_crystal_block.lmp"
        assert not base_block.exists()   # unlinked after the "successful" run
    finally:
        if meta_path is not None and meta_path.exists():
            meta_path.unlink()
        stray_amor = get_amorphous_path(mat_name)
        if stray_amor is not None and stray_amor.exists():
            stray_amor.unlink()


def test_calculate_layer_shifts_pair_bonding_n_layers_3(tmp_path):
    """Finding 2: use_pair_bonding=True, n_layers=3 -- the exact shift-tuple
    sequence for the L1-L2/L3-L4 pairing pattern's n_layers==3 special case
    (layer 0 unshifted, layers 1-2 share one shift)."""
    dims = {'xhi': 10.0, 'xlo': 0.0, 'yhi': 30.0, 'ylo': 0.0}
    shifts = components.calculate_layer_shifts(
        "h-MoS2", dims, n_layers=3, use_pair_bonding=True, stacking_type='AB',
    )
    assert shifts == [(0.0, 0.0), (0.0, 10.0), (0.0, 10.0)]


def test_calculate_layer_shifts_pair_bonding_n_layers_5(tmp_path):
    """Finding 2: use_pair_bonding=True, n_layers=5 -- the general (n_layers>3)
    branch: layers 0-1 unshifted (first pair), the middle layer gets one
    shift, and the last pair (layers 3-4) share two shifts."""
    dims = {'xhi': 10.0, 'xlo': 0.0, 'yhi': 30.0, 'ylo': 0.0}
    shifts = components.calculate_layer_shifts(
        "h-MoS2", dims, n_layers=5, use_pair_bonding=True, stacking_type='AB',
    )
    assert shifts == [
        (0.0, 0.0), (0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (0.0, 20.0),
    ]


LMP = shutil.which("lmp_serial")


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_stack_multilayer_sheet_auto_lat_c_recurses_for_n_layers_gt_2(tmp_path):
    """Finding 3: lat_c=None with n_layers=3 first minimizes a 2-layer trial to
    find lat_c, then recurses into itself with the full n_layers and the
    found value (components.py's self-reentry, ~1852-1873). Builds a real
    3-layer h-MoS2 sheet and checks the produced file's atom count is 3x the
    monolayer count and its z-extent grows consistently with the found
    interlayer spacing relative to a real 2-layer build at the same lat_c."""
    settings = load_settings()
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    mono_config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[1],
    )
    mono_path, _dims, _pot_counts, _total_types, _supercell = \
        components.build_monolayer(mono_config)
    mono_atoms = _count_atoms(mono_path)

    config_3l = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[3], lat_c=None,
    )
    stacked_path, _dims3, lat_c = components.build_sheet(
        config_3l, build_dir, stack_if_multi=True, settings=settings,
        n_layers_override=3,
    )
    assert lat_c is not None and lat_c > 0

    stacked_atoms = _count_atoms(stacked_path)
    assert stacked_atoms == 3 * mono_atoms

    # A real 2-layer build at the SAME found lat_c (lat_c no longer None, so
    # this does not re-minimize) isolates one interlayer gap's worth of
    # z-extent to compare against the 3-layer build's two gaps.
    build_dir_2l = tmp_path / "build2"
    build_dir_2l.mkdir()
    config_2l = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[2], lat_c=lat_c,
    )
    two_layer_path, _dims2, lat_c_2l = components.build_sheet(
        config_2l, build_dir_2l, stack_if_multi=True, settings=settings,
        n_layers_override=2,
    )
    assert lat_c_2l == pytest.approx(lat_c)

    z_extent_2l = _z_extent(two_layer_path)
    z_extent_3l = _z_extent(stacked_path)
    # 3 layers span roughly one more interlayer gap (lat_c) than 2 layers.
    assert z_extent_3l > z_extent_2l
    assert (z_extent_3l - z_extent_2l) == pytest.approx(lat_c, rel=0.3, abs=1.0)


def _count_atoms(lmp_path: Path) -> int:
    """Parse the atom count straight out of a LAMMPS data file's header."""
    text = lmp_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().endswith("atoms"):
            return int(line.split()[0])
    raise AssertionError(f"no 'N atoms' header line found in {lmp_path}")


def _z_extent(lmp_path: Path) -> float:
    dims = components.get_model_dimensions(lmp_path)
    return dims['zhi'] - dims['zlo']


class _FakeLmp:
    """Stands in for a LAMMPS instance: only lat_c extraction is meaningful."""

    def __init__(self, lat_c):
        self._lat_c = lat_c

    def command(self, _cmd):
        pass

    def extract_variable(self, _name, _group, _vartype):
        return self._lat_c

    def close(self):
        pass


def _stack_with_fake_lmp(monkeypatch, tmp_path, fake_lat_c, lat_c, info,
                         n_layers=2):
    settings = load_settings()
    config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[n_layers], lat_c=lat_c,
    )
    base, dims, pot_counts, total_types, supercell = \
        components.build_monolayer(config)
    monkeypatch.setattr(components, 'create_lammps_instance',
                        lambda: _FakeLmp(fake_lat_c))
    result = components.stack_multilayer_sheet(
        base_layer_path=base, config=config, output_path=tmp_path / "s.lmp",
        box_dims=dims, n_layers=n_layers, types_per_layer=total_types,
        pot_counts=pot_counts, supercell_dims=supercell, lat_c=lat_c,
        settings=settings, info_out=info)
    return result, settings


def test_stack_reports_minimised_source(monkeypatch, tmp_path):
    info = {}
    result, _ = _stack_with_fake_lmp(monkeypatch, tmp_path, 6.15, None, info)
    assert result == pytest.approx(6.15)
    assert info == {'lat_c_source': 'minimised'}


def test_stack_reports_default_source_when_extraction_fails(monkeypatch, tmp_path):
    info = {}
    result, settings = _stack_with_fake_lmp(monkeypatch, tmp_path, None, None, info)
    assert result == pytest.approx(settings.geometry.lat_c_default)
    assert info == {'lat_c_source': 'default'}


def test_stack_reports_config_source(monkeypatch, tmp_path):
    info = {}
    result, _ = _stack_with_fake_lmp(monkeypatch, tmp_path, 6.15, 6.0, info)
    assert result == pytest.approx(6.0)
    assert info == {'lat_c_source': 'config'}


def test_stack_recursion_for_three_layers_still_reports_minimised(monkeypatch, tmp_path):
    info = {}
    result, _ = _stack_with_fake_lmp(monkeypatch, tmp_path, 6.15, None, info,
                                     n_layers=3)
    assert result == pytest.approx(6.15)
    assert info == {'lat_c_source': 'minimised'}


def test_build_sheet_reports_monolayer_geometry(tmp_path):
    config = SheetConfig(
        mat="h-MoS2", pot_type="sw", pot_path=MOS2_POT, cif_path=MOS2_CIF,
        x=12.0, y=12.0, layers=[1],
    )
    info = {}
    components.build_sheet(config, tmp_path, lat_c_info_out=info)
    assert info['lat_c_source'] == 'not_applicable'
    assert info['atoms_per_layer'] > 0
    assert info['layer_thickness'] > 0


def test_tmp_path_format_and_uniqueness() -> None:
    import re
    first = components._tmp_path("MoS2", "unit")
    second = components._tmp_path("MoS2", "unit")
    assert first != second
    assert re.fullmatch(r"MoS2_[0-9a-f]{8}_unit\.lmp", first.name)
    assert re.fullmatch(r"sheet_2_[0-9a-f]{8}\.in\.settings",
                        components._tmp_path("sheet_2", ext=".in.settings").name)


def test_replicate_or_copy_replicates_only_above_one(tmp_path: Path) -> None:
    src = tmp_path / "src.lmp"
    src.write_text("data")
    with patch("frictionsim2d.builders.components.replicate_data_file") as dup:
        components._replicate_or_copy(src, tmp_path / "same.lmp", 1, 1, 1)
        dup.assert_not_called()
        assert (tmp_path / "same.lmp").read_text() == "data"
        components._replicate_or_copy(src, tmp_path / "big.lmp", 1, 2, 1)
        dup.assert_called_once_with(src, tmp_path / "big.lmp", 1, 2, 1)


def test_carve_flake_passes_shape_centre_and_returns_box(tmp_path: Path) -> None:
    from types import SimpleNamespace
    cfg = SimpleNamespace(shape="circle", edge_length=40, rotation_deg=5, corner_radius=3,
                          center_x=1.0, center_y=2.0)
    dims = dict(xlo=0.0, xhi=10.0, ylo=0.0, yhi=10.0, zlo=0.0, zhi=3.0)
    mod = "frictionsim2d.builders.construction.components"
    with patch(f"{mod}.build_flake") as build, \
         patch(f"{mod}.apply_flake_corner_types") as corner, \
         patch(f"{mod}.get_model_dimensions", return_value=dims):
        out = components.carve_flake(tmp_path / "in.data", tmp_path / "out.data", cfg)
    assert out == dims
    build.assert_called_once_with(
        tmp_path / "in.data", tmp_path / "out.data", shape="circle", edge_length=40.0,
        rotation_deg=5.0, center_xy=(1.0, 2.0))
    corner.assert_called_once_with(tmp_path / "out.data", shape="circle", corner_radius=3.0)


def test_calculate_layer_shifts_phosphorene_shifts_half_the_box_in_x_and_y() -> None:
    dims = {'xhi': 10.0, 'xlo': 0.0, 'yhi': 30.0, 'ylo': 0.0}
    for name in ("p-SnS", "black_phosphorus"):
        assert components.calculate_layer_shifts(name, dims, n_layers=3) == [
            (0.0, 0.0), (5.0, 15.0), (10.0, 30.0)]


def test_calculate_layer_shifts_aa_stacking_is_all_zero() -> None:
    dims = {'xhi': 10.0, 'xlo': 0.0, 'yhi': 30.0, 'ylo': 0.0}
    assert components.calculate_layer_shifts(
        "p-SnS", dims, n_layers=4, stacking_type='aa') == [(0.0, 0.0)] * 4


def _hex_data(tmp_path: Path, a2_over_a1: float) -> Path:
    a, height = 3.0, 0.866 * 3.0
    path = tmp_path / "hex.lmp"
    path.write_text(
        "hex\n\n2 atoms\n2 atom types\n\n"
        f"0.0 {a} xlo xhi\n0.0 {height} ylo yhi\n0.0 5.0 zlo zhi\n"
        f"{a2_over_a1 * a:.6f} 0.0 0.0 xy xz yz\n\n"
        "Masses\n\n1 95.95  # Mo\n2 32.06  # S\n\n"
        "Atoms # atomic\n\n1 1 0.0 0.0 0.0\n2 2 1.5 0.866 1.0\n"
    )
    return path


def test_find_ortho_transform_accepts_a_near_hexagonal_cell() -> None:
    cell = np.array([[3.0, 0.0, 0.0], [-0.50002 * 3.0, 0.866 * 3.0, 0.0], [0.0, 0.0, 5.0]])
    transform = components._find_ortho_transform(cell)
    assert np.abs(transform[1]).tolist() == [1, 2, 0]


def test_orthogonalize_snaps_the_residual_tilt_of_a_near_hexagonal_cell(tmp_path: Path) -> None:
    src = _hex_data(tmp_path, -0.50002)
    out = tmp_path / "ortho.lmp"
    components._orthogonalize_lammps_data(src, out)
    text = out.read_text()
    assert "xy xz yz" not in text
    dims = get_model_dimensions(out)
    assert dims['xhi'] - dims['xlo'] == pytest.approx(3.0, abs=1e-3)
    assert dims['yhi'] - dims['ylo'] == pytest.approx(2 * 0.866 * 3.0, abs=1e-3)


def test_assign_pot_types_matches_uncommented_masses_by_nearest_element(tmp_path: Path) -> None:
    path = tmp_path / "prim.lmp"
    path.write_text(
        "t\n\n3 atoms\n2 atom types\n\n"
        "0.0 3.0 xlo xhi\n0.0 3.0 ylo yhi\n0.0 6.0 zlo zhi\n\n"
        "Masses\n\n1 195.084\n2 78.971\n\n"
        "Atoms # atomic\n\n1 1 0 0 0\n2 2 1 1 1\n3 2 2 2 2\n"
    )
    components._assign_pot_types_by_order(path, ["Pt", "Se"], {"Pt": 1, "Se": 2})
    text = path.read_text()
    atoms = text.split("Atoms # atomic")[1].split()
    assert [int(atoms[i]) for i in (1, 6, 11)] == [1, 2, 3]
    assert "# Pt" in text and text.count("# Se") == 2
