# tests/

## Layout (mirrors `src/`)

| directory | tests for |
|---|---|
| `core/ run/ potentials/ lattice/ builders/ interfaces/ hpc/ cli/` | the `src` package of the same name |
| `integration/` | cross-package, artifact-level tests (marker `integration`), Phase 3. Nothing is mocked; `generate_package` builds are shared through the session-scoped `build_example` fixture in its `conftest.py`, with the example set in `helpers/examples.py`. Two files carry `lammps`: `test_smoke_lammps.py` and the one hetero case in `test_example_configs.py` (a hetero build shells out to `lmp_serial`) |
| `repo/` | repo-wide invariants: import surface, import weight, marker derivation, settings isolation, mock-target rule |
| `helpers/` | code shared by more than one test module — a test module never imports another test module |
| `fixtures/` | data files |
| `gui/ aiida/ api/ data/ postprocessing/` | out of scope of the 2026-09 review; moved, not audited |

## Tiers and how to run them

```bash
pytest                      # fast tier: everything except real-LAMMPS (addopts: -m 'not lammps')
pytest -m lammps            # real lmp_serial runs, 56 tests / ~24 min — ask before running on a busy machine
pytest -m integration       # cross-package tier, 103 tests / ~3:20 -- NOTE it overrides the
                            # 'not lammps' default, so the real-LAMMPS smoke runs too
pytest -m 'integration and not lammps'   # the same tier without LAMMPS: 97 tests / ~23 s
pytest -m ""                # everything
pytest tests/builders       # one package
pytest --cov=src --cov-report=term-missing tests/core   # coverage for one package
```

The `lammps` marker is derived in `conftest.py` from each test's `skipif(... lmp_serial ...)`
guard, so the guard and the marker cannot drift apart (`repo/test_lammps_marker.py` pins it).

## Conventions

- **Name the behaviour, not the function:** `test_<subject>_<condition>_<expected>`.
- **Assert on produced artifacts** (rendered script text, data-file contents, directory
  names), not on `mock.assert_called()`.
- **Mock only expensive boundaries:** LAMMPS, atomsk, subprocess, DB, network. Partially
  enforced by `repo/test_mock_targets.py` (string-form targets only; object-form
  `monkeypatch.setattr(obj, 'name', …)` is not checked); its `KNOWN_VIOLATIONS` list is a
  ratchet that only shrinks.
- **One reason to fail per test.**
- **Use conftest fixtures** (`mock_settings`, `materials_path`, `potentials_path`, the
  `*_config` fixtures) instead of rebuilding `GlobalSettings` locally.
- **Paths:** `Path(__file__).resolve().parents[2]` is the repo root from any test package
  (`parents[1]` from tests/conftest.py and tests/helpers/). Never rely on the cwd.
- **Hermetic:** `XDG_CONFIG_HOME` is redirected for the whole session (conftest), so
  `load_settings()` never reads your real settings file.
