# Testing

How the suite is organised, how to run each tier, and the two ways a naive
`pytest` invocation misleads you.

## Tiers

```bash
pytest tests --ignore=tests/gui          # fast tier: 3689 tests, ~14 min
pytest -m lammps                         # real LAMMPS runs: 80 tests
pytest -m 'integration and not lammps'   # cross-package tier: 105 tests
pytest -m integration                    # the same tier PLUS the LAMMPS smoke: 116 tests
pytest tests/aiida -m aiida              # the AiiDA tier: 291 tests, ~11 min, broker-less SQLite profile
pytest -m ""                             # everything
pytest tests/builders                    # one package
pytest tests/core/test_config.py::test_name              # one test
pytest --cov=frictionsim2d --cov-report=term-missing tests/core    # coverage for one package
```

`pyproject.toml` sets `addopts = "-v --tb=short -m 'not lammps'"`, so the fast tier is
the default.

### Two traps

- **Never run a bare `pytest`.** `tests/gui/test_gui_app_smoke.py` hangs on an unpatched
  modal dialog and the run never finishes. Always name paths or pass
  `--ignore=tests/gui`.
- **`-m integration` replaces the default marker expression**, including the
  `not lammps` part, so it pulls the real-LAMMPS smoke in with it. Use `-m 'integration and not lammps'` when you want only the fast half.

## Markers

Two, both registered in `pyproject.toml`:

| marker | meaning |
|---|---|
| `lammps` | executes a real LAMMPS run. Skipped when `lmp_serial` is absent. |
| `integration` | cross-package, artifact-level. Lives in `tests/integration/`. |

A file may carry both. **`lammps` is derived, not written by hand**: `tests/conftest.py`
reads each test's `skipif(... lmp_serial ...)` guard and applies the marker from it, so a
test cannot claim to be LAMMPS-free while calling LAMMPS.
`tests/repo/test_lammps_marker.py` pins that derivation.

## Layout

`tests/` mirrors `frictionsim2d/`, one directory per package, plus four that do not:

| directory | holds |
|---|---|
| `core/ run/ potentials/ lattice/ builders/ interfaces/ hpc/ cli/` | tests for the `frictionsim2d` package of the same name |
| `integration/` | cross-package, artifact-level tests — see below |
| `repo/` | repo-wide invariants: import surface, import weight, marker derivation, settings isolation, mock-target rule |
| `helpers/` | code shared by more than one test module (a test module never imports another test module) |
| `fixtures/` | data files |

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
| `test_example_configs.py` | all ten `examples/*.ini` through parse → expand → build → render |
| `test_hpc_over_a_real_tree.py` | HPC generation over a builder's own output, both schedulers × all three job shapes |
| `test_settings_layering.py` | a settings value surviving into the rendered script |
| `test_sweep_expansion.py` | one directory per swept value, on disk |
| `test_cli_chain.py` | the CLI driven into a real build, nothing patched |
| `test_failure_paths.py` | bad configs refused by name, with nothing written |
| `test_smoke_lammps.py` | an end-to-end AFM run under real LAMMPS (`lammps` marker) |

One case in `test_example_configs.py` also carries `lammps`: a hetero build shells out to
`lmp_serial` while building, so it cannot run in the LAMMPS-free tier.

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
