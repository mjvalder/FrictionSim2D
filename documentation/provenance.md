# Provenance

How FrictionSim2D records input provenance for reproducibility, and how that
record feeds the AiiDA store.

## Design goals

- Keep enough metadata to reconstruct exactly how a run was built and what
  potential each material used.
- Record which files (CIF, potential) went into each component, with
  checksums.
- Give AiiDA import and `postprocess`/GUI tooling one shared, versioned schema
  to read.

## The provenance record — schema 2.2.0

Provenance is a Pydantic model, `ProvenanceRecord` (`frictionsim2d/data/provenance.py`),
written by the builder at build time as `provenance.json`. A missing required
field fails the *build*, not later analysis.

Top-level identity:

- `schema_version` — `"2.2.0"` for anything written today.
- `run_uuid` — the run's identity, a uuid4 drawn at build time (2.2.0). Each
  AFM `L{n}` record gets its own, and a rebuild of the same config is a new
  run with a new one. The AiiDA import keys the run and its points on it
  (`point_uuid = uuid5(run_uuid, point label)`), so the same run imported
  under two labels, or from a moved copy, is still one run. A pre-2.2 record
  has none and reads with `run_uuid = None`, and the import refuses it
  (`MissingRunUUID`). Legacy runs are **stamped, not rebuilt**:
  `FrictionSim2D aiida stamp-uuid ROOT [--dry-run]` gives each such record a
  fresh `run_uuid` and records `run_uuid_stamped_at`, leaving every other key
  as it was. Stamp once, before the first import (see
  [aiida.md](aiida.md#42-legacy-runs-stamp-then-import)).
- `run_uuid_stamped_at` — set only on a stamped legacy record (ISO date), so
  a stamped run stays distinguishable from a 2.2 build. The import uses it to
  accept the rounded AFM spring factor such runs were built with
  ([aiida.md](aiida.md#3-parameter-blocks)).
- `family` — what schema this run's result files follow (`afm`, `indent`,
  `sheetonsheet`, `hetero`, `flake`, `pes_tip`, `pes_sheet`).
- `model` — where the run lives in the output tree (the CLI/top-level axis:
  `afm`, `indent`, `pes_tip`, `sheetonsheet`, `pes_sheet`).

### `materials[]` — one entry per material, four roles

Schema 1.x recorded a single `pot_path`/`pot_type`/`pot_file_sha256` triple,
which silently captured only the primary sheet — a hetero stack with two
different potentials had no way to record the second one. Schema 2.0.0
replaces it with a list of `MaterialProvenance` entries, one per material of
the built system:

```python
MATERIAL_ROLES = ('sheet', 'flake', 'tip', 'substrate')
```

- **`sheet`** — a hetero run has two or more of these (one per layer/material);
  a plain sheet-on-sheet run has one or two.
- **`flake`** — the flake family's single mobile body.
- **`tip`** — the AFM tip.
- **`substrate`** — the AFM/indent substrate.

Each entry carries `name`, `role`, `pot_path`, `pot_type`,
`pot_resolution_route`, `pot_file_sha256` — the same quadruple every builder
family used to derive separately, now built once by
`material_provenance(component, role)` and shared by all of them.

Other fields worth knowing: `twist_angle` (the stacking twist a hetero cell
was built with — distinct from `scan_angle`, the slide direction), the
per-source RNG seeds (`seed_thermal_tip`, `seed_thermal_sub`,
`seed_velocity`, `seed_sheet_center`), build identity (`git_commit`,
`git_dirty`, `config_hash`, `lammps_version`), and post-run enrichment fields
(`ramp_converged`, `ramp_r`, `ramp_iterations`, `unloaded_reference_z`,
`wall_time_s`, ...) filled in from the LAMMPS log after the run finishes.
Enrichment (`hpc enrich-provenance`) may happen after the run was imported:
`provenance.json` is part of every deck's fingerprint, so re-running `aiida
import` on the tree supersedes those decks with ones that carry the enriched
record (see [aiida.md](aiida.md#43-re-import-and-supersession)).

### `provenance/settings.yaml`

Alongside `provenance/config.json`, the run's own `settings.yaml` snapshot is
written by `run/simulate.py` next to it. This is what makes `aiida rebuild`
possible without a live `--settings-file`: the exact settings a run was
generated under travel with it.

Both files' credential-named fields (`password`, `api_key`, `token`, ...,
wherever they sit) are blanked before they ever reach the AiiDA store, at
write time and again when the file is staged into a node repository (see
[aiida.md](aiida.md#23-credentials-never-enter-the-store) for the precise
guarantee). Every other field is stored as written. HPC paths, the account
and host names are anonymised only in an exported archive, never in the store
([aiida.md](aiida.md#9-archives)).

## Reading old provenance: `read_provenance`

`read_provenance(path, *, family='afm', model='afm', material_name='')` reads
a `provenance.json` and **upgrades a schema-1.x file in memory only** — the
file on disk is never rewritten. A 1.x file predates `family`/`model`/
`materials`, so on such a file the function:

- fills in `family`/`model` from its keyword arguments (callers such as
  `frictionsim2d.run.tree.discover_runs` pass the run's own tokens),
- turns the old single `pot_path`/`pot_type`/`pot_resolution_route`/
  `pot_file_sha256` quadruple into one `MaterialProvenance` entry with
  `role='sheet'` and `name=material_name`.

A schema-2.x file is returned unchanged; none of the keyword arguments have
any effect on it.

## Typical provenance tree

```text
<run>/
  provenance.json
  provenance/
    config.json
    settings.yaml
    cif/
      <files>.cif
    potentials/
      <potential files>
```

AFM-family runs additionally nest per-layer directories (`L1`, `L2`, ...)
under the model root; each holds its own `provenance.json`.

## Why checksums matter

- Confirm file integrity after transfer (workstation → HPC → back).
- Detect accidental replacement of a source CIF or potential file.
- Let AiiDA import skip re-storing a file it has already seen with the same
  hash, and let a reader confirm two runs used byte-identical assets even if
  the path names differ.

## Consumption by AiiDA and postprocessing

`aiida import` reads each run's `provenance.json`, `provenance/config.json`
and `provenance/settings.yaml` and records the run as one CalcJob per LAMMPS
deck (see [aiida.md](aiida.md)). `postprocess` and the GUI read the same files
directly off disk for a tree that was never imported.

### From provenance to blocks

The import splits the three documents with one table,
`frictionsim2d/aiida/block_schema.py`:

- **Blocks** (`FIELD_MAP`). Every requested leaf of the run's config model
  (`AFMSimulationConfig` or `SheetOnSheetSimulationConfig`) and of
  `GlobalSettings` (as `settings.<section>.<field>`) is assigned to one block
  kind: `material`, `potential`, `interaction`, `thermostat`, `integration`,
  `drive`, `geometry`, `protocol`, `output` or `realization`. One `material`
  and one `potential` block are made per `materials[]` entry; the potential
  block takes the potential type and file sha256 the build resolved, from
  `provenance.json`, not the path the config named.
- **Seeds** (`seed_thermal_tip`, `seed_thermal_sub`, `seed_velocity`,
  `seed_sheet_center`, `substrate_realization_id`, the indent `seed`) form the
  `realization` block.
- **Swept axes** (`force`/`pressure`, `scan_angle`, `scan_speed`) become point
  attributes (`load`, `angle`, `speed`), never block content.
- **`derived`** (`PROVENANCE_MAP` → `'derived'`) collects what the builder
  computed: every `geometry.*` field of the built system (relaxed `lat_c`,
  cell sizes, ...), `twist_angle`, `drive_k`, `drive_rest_length`,
  `damping_ratio`, `damp_ev`, `thermostat_scheme`, contact cutoff, the layer
  type map, tip group sizes and the interlayer details. It is an input of each
  deck, but never hashed, so two runs that asked for the same thing share
  their blocks even when the builder relaxed a lattice differently.
- **Per point and per deck.** Requested and achieved loads go to the points;
  `lammps_version`, wall time, process count and the ramp diagnostics go to
  each deck's `run_info`.
- **Excluded** (`EXCLUDED`): what is not physics — `hpc.*`, `aiida.*`,
  `integrity.*`, how and where the amorphous quench runs, `materials_list`,
  the deck shape (`outer_loop`, `scan_angle_force`), and the record's own
  identity (`schema_version`, `run_uuid`, `run_uuid_stamped_at`, `family`,
  `model`, `git_commit`, `config_hash`, the potential path and resolution
  route).

Within a block, a field equal to its family default is elided, so adding a
field to a model later changes no existing hash.

**Units.** `UNITS` gives every numeric field a unit, checked against the
builder or template that consumes the value: lengths in `angstrom` (LAMMPS
metal units), temperatures in `K`, the requested springs in `N/m`, `drive_k` in
`eV/angstrom^2`, angles in `deg`, step counts in `steps`, and `1` for
dimensionless numbers. A test fails for a numeric field without a unit. The
units, kinds and defaults of a family are stored with each deck as its
`schema` node, so a reader always gets the units the run was stored with.

The import checks one convention: `drive_k` must be the requested
`driving_spring` divided by `NM_TO_EV_A2 = 16.02176634`. Every builder now
uses that exact factor; AFM builds before schema 2.2.0 divided by `16.02`, and
for those records (and stamped ones) the import also accepts the rounded
factor.

## Related docs

- [overview.md](overview.md)
- [aiida.md](aiida.md)
- [python_api.md](python_api.md)
