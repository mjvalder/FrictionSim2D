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

## The provenance record — schema 2.0.0

Provenance is a Pydantic model, `ProvenanceRecord` (`src/data/provenance.py`),
written by the builder at build time as `provenance.json`. A missing required
field fails the *build*, not later analysis.

Top-level identity:

- `schema_version` — `"2.0.0"` for anything written today.
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
Enrichment happens after the AiiDA run node may already exist; re-running
`aiida import` on the tree stores the enriched record (see
[aiida.md](aiida.md), "provenance_updated").

### `provenance/settings.yaml`

Alongside `provenance/config.json`, the run's own `settings.yaml` snapshot is
written by `run/simulate.py` next to it. This is what makes `aiida rebuild`
possible without a live `--settings-file`: the exact settings a run was
generated under travel with it.

Both files' `database.password`/`database.api_key` are stripped before they
ever reach the AiiDA store, at write time and again when the file is staged
into a node repository (see [aiida.md](aiida.md#credentials-never-enter-the-store));
every other field, including `database.host`, travels intact.

## Reading old provenance: `read_provenance`

`read_provenance(path, *, family='afm', model='afm', material_name='')` reads
a `provenance.json` and **upgrades a schema-1.x file in memory only** — the
file on disk is never rewritten. A 1.x file predates `family`/`model`/
`materials`, so on such a file the function:

- fills in `family`/`model` from its keyword arguments (callers such as
  `src.run.tree.discover_runs` pass the run's own tokens),
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

`aiida import` reads `provenance.json` (and the files it points at) to build
the `FrictionSimulationData` node's attributes and repository files — see
[aiida.md](aiida.md). `postprocess` and the GUI read the same file directly
off disk for a tree that was never imported.

## Related docs

- [overview.md](overview.md)
- [aiida.md](aiida.md)
- [python_api.md](python_api.md)
