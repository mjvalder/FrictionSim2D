# tests/

## Layout (mirrors `frictionsim2d/`)

| directory | tests for |
|---|---|
| `core/ run/ potentials/ lattice/ builders/ interfaces/ hpc/ cli/` | the `frictionsim2d` package of the same name |
| `integration/` | cross-package, artifact-level tests (marker `integration`). Nothing is mocked; `generate_package` builds are shared through the session-scoped `build_example` fixture in its `conftest.py`, with the example set in `helpers/examples.py`. Files carrying `lammps`: `test_smoke_lammps.py`, `test_interlayer_lammps.py` and the one hetero case in `test_example_configs.py` (a hetero build shells out to `lmp_serial`) |
| `scenarios/` | the scenario tier (marker `scenario`): tiny end-to-end runs, build -> `fake_lmp.py` outputs -> `postprocess read` -> `aiida import`, one per row of `helpers/deck_snapshot.SCENARIO_CASES`; configs in `scenarios/configs/` |
| `repo/` | repo-wide invariants: import surface, import weight, marker derivation, settings isolation, mock-target rule |
| `helpers/` | code shared by more than one test module — a test module never imports another test module |
| `fixtures/` | data files. `fixtures/examples/` is a frozen copy of `examples/*.ini` (Task 22): the deck snapshot and every test that asserts an example's exact values read it; the live `examples/` is read only by contract tests -- every example parses and renders (`integration/` through `build_example`, `test_example_configs.py`, `test_examples_mlip.py`, `test_mlip_end_to_end.py`, `gui/test_config_validation.py`), the pair sweep stays runnable (`run/test_run_hetero_naming.py`), the assistant's example defaults match the files (`agent/test_tools_recipes.py`) |
| `gui/ aiida/ data/ postprocessing/ analysis/ agent/ tools/` | the package of the same name (`tools/`: scripts outside the package) |

## Tiers and how to run them

```bash
pytest                      # fast tier, 6075 tests; bare `pytest` is safe (GUI modals are stubbed).
                            # addopts: -m 'not lammps and not agent_eval and not publication'
pytest -m lammps            # real lmp_serial runs, 126 tests -- ask before running on a busy machine
pytest -m integration       # cross-package tier, 150 tests -- NOTE it overrides the
                            # default expression, so the real-LAMMPS smoke runs too
pytest -m 'integration and not lammps'   # the same tier without LAMMPS: 139 tests
pytest tests/aiida -m aiida # the AiiDA tier, 380 tests, temporary SQLite profile
pytest -m scenario          # fake-LAMMPS end-to-end tier, 17 tests, < 60 s (a session check in scenarios/conftest.py)
pytest -m ""                # everything
pytest tests/builders       # one package
pytest --cov=frictionsim2d --cov-report=term-missing tests/core   # coverage for one package
```

The `lammps` marker is derived in `conftest.py` from each test's `skipif(... lmp_serial ...)`
guard, so the guard and the marker cannot drift apart (`repo/test_lammps_marker.py` pins it).

The `scenario` tier is part of the default run. `scenarios/fake_lmp.py` stands in for
`lmp` (`python tests/scenarios/fake_lmp.py -in DECK [-log FILE]`): it writes every
`fix ave/time ... file`, `print ... file|append` and `write_data` target of a deck, for
each value of its `index` variables, plus a finished log; it never interprets physics.
A new row needs its build-time LAMMPS recorded once (`--record-lammps`, real `lmp`, so
keep the config tiny) and its snapshot written (`repo/test_deck_identity.py
--update-deck-snapshot -k NAME`).

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
