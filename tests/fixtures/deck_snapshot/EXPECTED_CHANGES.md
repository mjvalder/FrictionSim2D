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
