# FrictionSim2D Chat Handoff

## Scope Completed

### Phase 1: Data Model Unification and Friction Calculation Cleanup

Implemented a canonical data model and a single source of truth for coefficient of friction calculations.

Main outcomes:

- Added `ResultRecord` and canonical friction helpers in `src/data/models.py`
- Added validation rules in `src/data/validation.py`
- Expanded the PostgreSQL schema in `src/data/database.py`
- Unified COF/statistics computation across CLI, postprocessing, and AiiDA
- Added schema migration `src/data/migrations/v001_to_v002.py`
- Updated `settings.yaml` with database profiles and staging defaults

Canonical friction calculation:

- Uses ratio-of-means: `mean(lateral_force) / mean(normal_force)`
- Uses lateral force magnitude: `sqrt(lfx^2 + lfy^2)`
- Skips the first 20% of the trajectory by default as transient

Important implementation details:

- `compute_friction_stats()` is now the canonical path for summary statistics
- `compute_derived_columns()` is now the canonical path for per-step `lateral_force` and `cof`
- `src/aiida/data/results.py`, `src/postprocessing/read_data.py`, `src/cli.py`, and `src/data/database.py` were aligned to use the shared logic

### Phase 1 Risk Fixes

Addressed the calculation/data-mapping risks identified during review:

- Replaced fragile AFM `_FIELDS` mapping with exact `_AFM_FILE_COLUMNS`
- Fixed sheet column mapping to include the missing `v_comx_top` and `v_comy_top`
- Added file column-count validation in `src/postprocessing/read_data.py`
- Centralized physical conversion constants in `src/data/models.py`
- Updated LAMMPS templates to consume conversion constants through Jinja variables rather than hardcoded numbers
- Updated builders to pass these constants into template context

New shared constants in `src/data/models.py`:

- `EV_A_TO_NN`
- `EV_A3_TO_GPA`
- `NM_TO_EV_A2`

### Phase 2: Database Setup Tooling

Implemented the local/central database profile system and CLI/database plumbing needed to operate a shared PostgreSQL instance.

Main outcomes:

- Added `DatabaseProfileSettings` and `DatabaseSettings` to `src/core/config.py`
- Integrated `database` into `GlobalSettings`
- Added `db_from_profile()` in `src/data/database.py`
- Added a simple migration registry and migration runner
- Updated `_ensure_schema()` to apply pending migrations automatically
- Updated CLI database connection resolution to prefer:
  1. Explicit CLI flags
  2. Environment variables
  3. Active `settings.yaml` profile

New CLI commands added:

- `db init`
- `db migrate`
- `db create-key`
- `db stage`
- `db publish`
- `db reject`

All DB CLI commands now support `--profile local|central`.

## Files Added or Significantly Updated

### New files

- `src/data/models.py`
- `src/data/validation.py`
- `src/data/migrations/__init__.py`
- `src/data/migrations/v001_to_v002.py`

### Updated files

- `src/data/database.py`
- `src/postprocessing/read_data.py`
- `src/aiida/data/results.py`
- `src/cli.py`
- `src/core/config.py`
- `src/builders/afm.py`
- `src/builders/sheetonsheet.py`
- `src/templates/afm/slide.lmp`
- `src/templates/afm/system_init.lmp`
- `src/templates/sheetonsheet/slide.lmp`
- `src/data/__init__.py`
- `settings.yaml`
- `tests/test_database.py`

## Current Database/API Design

### Database model

The database currently supports:

- richer simulation metadata
- staging workflow via `status`
- API key table
- time-series deduplication via `time_series_hash`
- federated data links via `data_url`

Core workflow statuses:

- `local`
- `staged`
- `validated`
- `published`
- `rejected`

### Connection model

Two database profiles are now represented in config:

- `database.local`
- `database.central`

The active profile is selected by `database.active_profile` unless overridden in CLI.

## Verification Status

Verified passing:

- `tests/test_database.py`: 51 passed
- `tests/test_config.py`: 5 passed

Notes on the wider suite:

- There are unrelated pre-existing failures/errors outside the database work, particularly in `tests/test_atomsk.py`, `tests/test_base_builder.py`, and `tests/test_potential_manager.py`
- These were not part of the database/API implementation and were not fixed during this work

## What Still Needs To Be Done

### Next target: Phase 3 - REST API and Dual-Mode Client

This is the next logical implementation step.

Recommended Phase 3 tasks:

1. Create `src/api/__init__.py`
2. Create `src/api/server.py` with FastAPI app
3. Create `src/api/auth.py` for API key validation helpers
4. Add API dependencies to `pyproject.toml`
5. Add endpoint tests using FastAPI `TestClient`
6. Refactor `FrictionDB` to support two modes:
   - direct PostgreSQL mode
   - HTTP API client mode

Recommended first endpoints:

- `POST /results`
- `GET /results`
- `GET /results/{id}`
- `GET /statistics`
- `POST /results/{id}/publish`
- `POST /results/{id}/reject`

### Suggested design for the next chat

Refactor `FrictionDB` so the public methods remain stable while the backend can switch between:

- direct DB access for local development
- API access for remote/shared usage

Target methods to preserve at the public interface:

- `upload_result()`
- `upload_record()`
- `query()`
- `get_statistics()`
- `create_api_key()`
- `publish()`
- `reject()`

## Important Decisions Already Made

- Canonical COF: ratio-of-means with 20% transient skip
- Central DB: PostgreSQL with local/central profile support
- Auth model: per-user API keys hashed in database
- Staging model: upload to staged first, then validate/publish
- Time-series deduplication: `time_series_hash`
- Raw data federation: `data_url` field exists for external archived data links

## Practical Start Point For A New Chat

Use this prompt in the next chat:

"Continue FrictionSim2D from Phase 2 completion. Phase 1 and Phase 2 are already implemented. Read `CHAT_HANDOFF.md`, then implement Phase 3: FastAPI server in `src/api/`, add API tests, and refactor `FrictionDB` for direct/API dual-mode without breaking the current public interface. Keep changes minimal and consistent with the existing codebase."

## Final Snapshot

Status at handoff:

- Phase 1 complete
- Phase 1 risk fixes complete
- Phase 2 complete
- Database/config/CLI/test updates committed in workspace state
- Ready to start Phase 3 implementation