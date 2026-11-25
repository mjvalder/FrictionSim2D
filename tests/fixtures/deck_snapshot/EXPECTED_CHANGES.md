# Expected deck changes

Each intentional change to a rendered deck is logged here before the snapshot is updated.

| date | task | files | why |
|---|---|---|---|
| 2026-11-24 | 7 | all 23 `provenance/config.json` (and the gitignored `provenance/settings.yaml`) | `settings.quench.run_local`, `n_procs`, `allow_generate` removed; `melt_steps` 50000 -> 20000, `equilibrate_steps` 20000 -> 10000, the shipped amor_Si block's real conditions (amorphous-quench spec §8, §11.1) |
| 2026-11-24 | 7 | all 27 `provenance.json` | `config_hash` only: it hashes the config, whose `settings.quench` changed as above |
| 2026-11-24 | 7 | new case `quench_si/` (`lammps/quench.in`, `provenance.json`, `build/Si_crystal_block.lmp`, `provenance/{potentials,cif}`) | `write_quench_root` output (`run quench`), root-relative `read_data`/`write_data`; melt_temp 3000 so no shipped block satisfies it |
| 2026-11-25 | 22 | 33 `provenance/config.json`, 36 `provenance.json` (`pot_path`) | `cif_path`/`pot_path` `examples/{materials,potentials}/...` -> `<REPO>/frictionsim2d/data/{materials,potentials}/...`: the library moved into the package; a name found through the library is stored resolved, as a base_dir hit always was. Frozen example copies keep the legacy spelling (deprecation-warned); `deck_cases/` and `scenarios/configs/` now say `materials/...`/`potentials/...` |
| 2026-11-25 | 22 | 33 `provenance/manifest.json` | `original_path` `<REPO>/examples/...` -> `<REPO>/frictionsim2d/data/...`, the same move |
| 2026-11-25 | 22 | 37 `provenance.json` | `config_hash`: it hashes the config, whose paths changed as above |
| 2026-11-25 | 22 | render set | the shipped-example cases render from `tests/fixtures/examples/` (frozen copies of the pre-rewrite `examples/*.ini`; only `materials_list` repointed at the copies' own list files), so rewriting `examples/` cannot move them |
