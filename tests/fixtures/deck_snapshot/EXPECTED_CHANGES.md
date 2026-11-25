# Expected deck changes

Each intentional change to a rendered deck is logged here before the snapshot is updated.

| date | task | files | why |
|---|---|---|---|
| 2026-11-24 | 7 | all 23 `provenance/config.json` (and the gitignored `provenance/settings.yaml`) | `settings.quench.run_local`, `n_procs`, `allow_generate` removed; `melt_steps` 50000 -> 20000, `equilibrate_steps` 20000 -> 10000, the shipped amor_Si block's real conditions (amorphous-quench spec §8, §11.1) |
| 2026-11-24 | 7 | all 27 `provenance.json` | `config_hash` only: it hashes the config, whose `settings.quench` changed as above |
| 2026-11-24 | 7 | new case `quench_si/` (`lammps/quench.in`, `provenance.json`, `build/Si_crystal_block.lmp`, `provenance/{potentials,cif}`) | `write_quench_root` output (`run quench`), root-relative `read_data`/`write_data`; melt_temp 3000 so no shipped block satisfies it |
| 2026-11-25 | 22 | 33 `provenance/config.json`, 36 `provenance.json` (`pot_path`) | `cif_path`/`pot_path` `examples/{materials,potentials}/...` -> the portable library name `materials/...`/`potentials/...`: the library moved into the package, and a library file serialises by its library name (`ComponentConfig.portable_path`, fix round 1), never by the install location |
| 2026-11-25 | 22 | 33 `provenance/manifest.json` | `original_path` `<REPO>/examples/...` -> `<REPO>/frictionsim2d/data/...`, the same move (the manifest records the resolved file read) |
| 2026-11-25 | 22 | 37 `provenance.json` | `config_hash`: it hashes the config, whose path spellings changed as above; it no longer depends on where the package is installed |
| 2026-11-25 | 22 | render set | the shipped-example cases render from `tests/fixtures/examples/` (frozen copies of the pre-rewrite `examples/*.ini`; only `materials_list` repointed at the copies' own list files), so rewriting `examples/` cannot move them |
| 2026-11-23 | f14960a0 | 15 indent `lammps/*.in` | header comment names the import package `frictionsim2d/` instead of `src/` (package rename); no command changed |
| 2026-11-24 | 29ef1b89 (Q7) | 64 files (`lammps/*.in`, `provenance.json`) | one eV/Å -> nN constant, CODATA 2018 `e`: 1.602176565 -> 1.602176634, so `drive_k` 3.12075467163009 -> 3.1207545372303813; `config_hash` follows |
| 2026-11-24 | d645381e (Q1) | twist hetero case: `hetero.data`, `b_twist_supercell.lmp`; `b_twist_rotated.lmp` removed | the no-op layer-B rotation is gone from `build_twisted_supercells`: coordinates move by float noise only, and the intermediate rotated file is no longer written |
| 2026-11-24 | 061111e7 + afe6f8ea (MACE) | 50 `provenance/config.json`, matching `provenance.json` | `precision: "float64"` now in every config (the `-f32` opt-in is the other value), so `config_hash` moves; `mlip_backend` added by 061111e7 and dropped again by afe6f8ea (internal, not serialised) |

Physics changes no snapshot covers (older runs differ from what a rebuild would now produce):

- CU-1: D3 dispersion is now applied in sheet-on-sheet hetero and `flake_slide`. Older runs of those builders recorded `dispersion=d3` but ran without it.
- ML-1 / ML-2 (e5400441): MLIP hetero (AFM and sheet) and `flake_slide` assemblies now `write_data ... nocoeff` under the `pair_style zero` surrogate. Before, those decks aborted ("Must define pair_style before Pair Coeffs"), so no older run of them exists to differ.
