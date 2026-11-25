# Testing

How the suite is organised, how to run each tier, how GUI dialogs are kept from
hanging a run, and how `-m integration` can mislead you.

## Tiers

```bash
pytest                                   # default: 6075 tests (gui and aiida included); bare `pytest` is safe
pytest -m lammps                         # real LAMMPS runs: 126 tests
pytest -m 'integration and not lammps'   # cross-package tier: 139 tests
pytest -m integration                    # the same tier PLUS the LAMMPS smoke: 150 tests
pytest tests/aiida -m aiida              # the AiiDA tier: 380 tests, broker-less SQLite profile
pytest -m scenario                       # fake-LAMMPS end-to-end tier: 17 tests, < 60 s
pytest -m ""                             # everything
pytest tests/builders                    # one package
pytest tests/core/test_config.py::test_name              # one test
pytest --cov=frictionsim2d --cov-report=term-missing tests/core    # coverage for one package
```

`pyproject.toml` sets `testpaths = ["tests"]` and
`addopts = "-v --tb=short -m 'not lammps and not agent_eval and not publication'"`, so the
fast tier is the default. Counts are the collected numbers at the time of writing.

### Two things to know

- **GUI modals are stubbed.** `tests/gui/conftest.py` replaces every `QMessageBox` /
  `QFileDialog` / `QInputDialog` call and `QDialog.exec`: an unpatched one fails the test
  instead of hanging it. A test that expects a dialog patches it itself. A 300 s per-test
  `faulthandler` backstop (`FS2D_GUI_TEST_TIMEOUT`) dumps tracebacks and exits on a hang.
- **`-m integration` replaces the default marker expression**, including the
  `not lammps` part, so it pulls the real-LAMMPS smoke in with it. Use `-m 'integration and not lammps'` when you want only the fast half.

## Markers

The main ones, all registered in `pyproject.toml`:

| marker | meaning |
|---|---|
| `lammps` | executes a real LAMMPS run. Skipped when `lmp_serial` is absent. |
| `integration` | cross-package, artifact-level. Lives in `tests/integration/`. |
| `scenario` | tiny end-to-end runs with a fake LAMMPS. Lives in `tests/scenarios/`; part of the default run. |
| `aiida` | needs aiida-core; runs against a temporary broker-less SQLite profile, never the daemon. |
| `agent_eval` | live-LLM evaluation of the GUI assistant (`AGENT_EVAL_PROVIDER`: `gemini` needs `GEMINI_API_KEY`, `ollama` a running server). Deselected by default; 234 tests. |
| `publication` | read-only checks against the frozen publication tables and the `friction2d_pub` profile. Skips when absent; deselected by default. |

A file may carry both. **`lammps` is derived, not written by hand**: `tests/conftest.py`
reads each test's `skipif(... lmp_serial ...)` guard and applies the marker from it, so a
test cannot claim to be LAMMPS-free while calling LAMMPS.
`tests/repo/test_lammps_marker.py` pins that derivation.

## Layout

`tests/` mirrors `frictionsim2d/`, one directory per package, plus five that do not:

| directory | holds |
|---|---|
| `core/ run/ potentials/ lattice/ builders/ interfaces/ hpc/ cli/ aiida/ data/ postprocessing/ analysis/ agent/ gui/` | tests for the `frictionsim2d` package of the same name |
| `integration/` | cross-package, artifact-level tests — see below |
| `scenarios/` | the scenario tier — see below |
| `repo/` | repo-wide invariants: import surface, import weight, marker derivation, settings isolation, mock-target rule |
| `helpers/` | code shared by more than one test module (a test module never imports another test module) |
| `fixtures/` | data files. `fixtures/examples/` is a frozen copy of the pre-2026-11-25 flat `examples/*.ini`: the deck snapshot and every test that asserts an example's exact values read it, so rewriting `examples/` never moves them |

## The integration tier

`tests/integration/` builds real simulation packages and asserts on what came out.
**Nothing is mocked** — that is the point of the tier. It covers the ground no unit test
can see: that a rendered script and the data file it reads agree, that a manifest names
files the builder actually wrote, and that a setting loaded correctly still reaches the
template.

`generate_package` is the slow part, so every module shares one build per config through
the session-scoped `build_example` fixture in `tests/integration/conftest.py`; the
example set and the build helper live in `tests/helpers/examples.py`.

| module | covers |
|---|---|
| `test_example_configs.py` | every run config in `examples/` (the eight single configs, the two campaigns, the hetero pair sweep) through parse → expand → build → render; the campaigns are built one material wide and their member count is asserted separately |
| `test_examples_mlip.py` | every MLIP example (any `examples/<mode>/*.ini` with an MLIP `pot_type`) validates, including the MLIP cell rules |
| `test_mlip_end_to_end.py` | the MLIP topology that runs, from `run_simulations` to the on-disk tree |
| `test_interlayer_lammps.py` | the interlayer styles against the installed LAMMPS (`lammps` marker) |
| `test_hpc_over_a_real_tree.py` | HPC generation over a builder's own output, both schedulers × all three job shapes |
| `test_settings_layering.py` | a settings value surviving into the rendered script |
| `test_sweep_expansion.py` | one directory per swept value, on disk |
| `test_cli_chain.py` | the CLI driven into a real build, nothing patched |
| `test_failure_paths.py` | bad configs refused by name, with nothing written |
| `test_smoke_lammps.py` | an end-to-end AFM run under real LAMMPS (`lammps` marker) |

One case in `test_example_configs.py` also carries `lammps`: a hetero build shells out to
`lmp_serial` while building, so it cannot run in the LAMMPS-free tier.

## The scenario tier

`tests/scenarios/` runs each row of a small matrix end to end without LAMMPS:
build (build-time LAMMPS replayed from `tests/fixtures/lammps_replay/`) -> every deck
through `fake_lmp.py` -> `FrictionSim2D postprocess read --export` -> `FrictionSim2D
aiida import` into a throwaway profile -> `point_table()`. The rows
(`tests/helpers/deck_snapshot.py::SCENARIO_CASES`, also in the deck snapshot):

| row | family | potential | condition |
|---|---|---|---|
| `afm_sw_load` | afm | sw | load sweep |
| `afm_rebomos_angle_finite` | afm | rebomos | angle sweep, finite sheet, spring edges |
| `afm_hetero_finite` | afm (hetero) | sw + lj interlayer | — |
| `afm_freestanding_tersoff` | pes_tip (freestanding) | tersoff (b-SiGe) | — |
| `indent_sw` | indent | sw | hold + retract |
| `sheet_reaxff_speed` | sheetonsheet | reaxff | speed sweep |
| `sheet_sw_ilp` | sheetonsheet | sw | ilp interlayer |
| `sheet_hetero_twist` | hetero | sw | twist |
| `flake_slide_sw` | flake | sw | — |
| `pes_tip_sw` | pes_tip | sw | — |
| `pes_sheet_ljg` | pes_sheet | sw + lj_g | stands in for kc (C/H only) |
| `sheet_mace_d3` | sheetonsheet | mace (fake model file) | d3 |
| `quench_si` | quench | sw | `quench import` -> `find_block` |

`fake_lmp.py` (`python tests/scenarios/fake_lmp.py -in DECK [-log FILE]`) writes the
files a deck would: each `fix ave/time ... file` (two header lines, five rows), each
`print ... file|append`, each `write_data` (a copy of the last `read_data` input), once
per value of the deck's `index` variables, plus a finished log. Only eight rows also run
`aiida import` (one per family, plus the MLIP row): importing all of them breaks the
60 s budget. The budget is a session check (`tests/scenarios/conftest.py`): the summed
setup + call + teardown time of every selected scenario test, judged only when all of
them finished; over budget, the session exits 1 with a summary line.

## Conventions

- **Name the behaviour, not the function:** `test_<subject>_<condition>_<expected>`.
- **Assert on produced artifacts** — rendered script text, data-file contents, directory
  names — not on `mock.assert_called()`.
- **Mock only expensive boundaries:** LAMMPS, atomsk, subprocess, DB, network. Partially
  enforced by `tests/repo/test_mock_targets.py`, whose `KNOWN_VIOLATIONS` list is a
  ratchet that only shrinks.
- **One reason to fail per test.**
- **Use the conftest fixtures** (`mock_settings`, `materials_path`, `potentials_path`)
  rather than rebuilding `GlobalSettings` locally.
- **Never rely on the cwd.** `Path(__file__).resolve().parents[2]` is the repo root from
  any test package.
- **Hermetic:** `tests/conftest.py` redirects `XDG_CONFIG_HOME` for the whole session, so
  `load_settings()` never reads your real settings file.

`tests/README.md` carries the same conventions next to the code.
