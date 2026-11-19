# FrictionSim2D publication dataset

**DOI**: [10.82186/gs4dc-90a11](https://doi.org/10.82186/gs4dc-90a11)

This folder rebuilds every figure of *FrictionSim2D: A High-Throughput Molecular Simulation
Framework to Screen the Friction of Two-Dimensional Materials* from the simulation data. The
data are not stored here. They are in the AiiDA archive `friction2d_archive.aiida`, which you
download from the DOI above.

## The archive

The archive is a ZIP-format AiiDA archive. It holds the complete provenance of 11 simulation
sets: LAMMPS inputs, potentials, structures, results, integrity verdicts and summaries.

| Set | Type | Contents |
|---|---|---|
| `260909-afm` | AFM | 156 materials, 100 × 100 Å, layers 1–2, 2–100 nN |
| `261107-sheet` | Sheet-on-sheet | 156 materials, 100 × 100 Å |
| `260227-sw_lammps` | Sheet-on-sheet | h-MoS₂, 5 layers, SW (Jiang, parameter set 1) |
| `260227-mos2_jing2` | Sheet-on-sheet | h-MoS₂, 5 layers, SW (Jiang, parameter set 2) |
| `260227-sevik_mos2` | Sheet-on-sheet | h-MoS₂, 5 layers, SW (Kandemir/Sevik) |
| `260227-rebomos` | Sheet-on-sheet | h-MoS₂, 5 layers, REBOMoS |
| `260312-force_rebomos` | Sheet-on-sheet | h-MoS₂ force–displacement, REBOMoS |
| `260312-force_sevik` | Sheet-on-sheet | h-MoS₂ force–displacement, SW (Kandemir/Sevik) |
| `260830-deepmd` | Sheet-on-sheet | h-MoS₂, 30 × 30 Å, DeePMD + D3 |
| `260909-mace_d3` | Sheet-on-sheet | h-MoS₂, 30 × 30 Å, MACE + D3 |
| `260909-mace_nod3` | Sheet-on-sheet | h-MoS₂, 30 × 30 Å, MACE without D3 |

Some legacy sets were made before FrictionSim2D recorded a `provenance.json`. For those, the
record was rebuilt from each run's stored config and manifest; such runs carry a
`run_uuid_stamped_at` date. The input deck of `260312-force_rebomos` was lost, so the deck in
the archive is a reconstruction. It starts with a `# RECONSTRUCTED` header that says what it
was rebuilt from.

## Requirements

You need a full installation of FrictionSim2D with the AiiDA extra
(`pip install -e ".[all]"`); see [../installation.md](../installation.md). The archive is a ZIP
file with an SQLite database inside, so it can also be read without AiiDA.

## Rebuild the figures

```bash
FrictionSim2D aiida setup                                 # once: creates the profile
FrictionSim2D aiida import-archive friction2d_archive.aiida
FrictionSim2D aiida list-sets                             # the 11 sets above

# One dataset per set (skip any that `aiida list-sets` already shows):
for s in 260909-afm 261107-sheet 260227-sw_lammps 260227-mos2_jing2 260227-sevik_mos2 \
         260227-rebomos 260312-force_rebomos 260312-force_sevik 260830-deepmd \
         260909-mace_d3 260909-mace_nod3; do
    FrictionSim2D aiida dataset "$s" "$s"
done

cd documentation/publication
FrictionSim2D postprocess plot plots_publication.json --settings plot_settings.json --output-dir plots
python compare_to_reference.py plots
```

`plots_publication.json` reads each set as `aiida://dataset/<set>`. A timeseries or stick-slip
plot reads only the points its filters select, straight from the store; nothing is written to
disk.
`compare_to_reference.py` compares each rebuilt figure with the published one in
`Images/plots/`. It reports each figure as identical byte for byte, identical pixel for pixel
(only the PNG metadata differs), or different.

## Figures

| Figure | Plot type | Sets |
|---|---|---|
| `cof_histogram_mos2.png` | `cof_histogram` | 260227 (four potentials), MLIP sets via `mos2_mlip_lf_nf.json`, literature |
| `lf_vs_nf_100x100_mos2_sx_sy.png` | `summary` | as above |
| `lf_vs_load_100x100_afm.png` | `summary` | `260909-afm` |
| `lf_vs_time_h_MoS2_l2_afm_v3.png`, `lf_vs_time_h_MoS2_l2_afm_2_30_85nN.png` | `timeseries` | `260909-afm` (the two files are the same figure) |
| `lf_vs_time_h_MoS2_l2_v_com_top.png` | `timeseries` | `260312-force_*` |
| `stick_slip_h_MoS2_v_com_top_0.1GPa_{rebomos,sevik}{,_spectrum}.png` | `stick_slip_analysis` | `260312-force_*` |
| `lf_vs_load_100x100_sheetonsheet.png` | `summary` | `261107-sheet` |
| `tribindex_vs_cof_sheetonsheet.png` | `scatter_comparison` | `261107-sheet`, `trib_index.json` |
| `fs2d_afm_stickslip_period.png` | `stickslip_period_histogram` | `260909-afm` |
| `fs2d_contact_area_cof.png`, `fs2d_contact_atoms_cof.png` | `contact_cof_scatter` | `260909-afm` |
| `fs2d_sheet_profile_mos2.png` | `sheet_profile` | `260909-afm` (h-MoS₂ snapshots) |
| `integrity_panel_h_{MoS2,NiSe2}_l2.png` | `integrity_panel` | `260909-afm` (snapshots and traces) |

**Not yet rebuildable from the archive alone:**
- The last five rows read tables that are analyses of the AFM campaign: contact metrics,
  stick-slip periods, the COF ranking and integrity traces. FrictionSim2D will compute these
  itself, and until it does, those plots expect the tables in `tables/`.
- The sheet profile and the integrity panels also read the run's snapshots from `trees/`,
  which `FrictionSim2D aiida export-results 260909-afm trees/260909-afm` writes.
- The two sheet-on-sheet figures were drawn from runs cut to whole lattice periods. That cut
  is one of the same analyses.

## Files

| File | Contents |
|---|---|
| `plots_publication.json` | The plot document: one entry per figure |
| `plot_settings.json` | Figure style (size, fonts, palette) used for the article |
| `trib_index.json` | Lubricating figure of merit from high-throughput DFT [1], restricted to the materials whose structure matches ours |
| `mos2_comparison_data.json` | Literature MoS₂ COF against load from MD [2] |
| `mos2_reaxff_comparison.json` | Literature MoS₂ ReaxFF and experimental nanoscale friction [3] |
| `mos2_mlip_lf_nf.json` | Mean NF and mean LF (`|v_sx, v_sy|`) per load of the three MLIP sets |
| `compare_to_reference.py` | Rebuilt figures against `Images/plots/` |
| `Images/` | The article's figures (`plots/`) and artwork |

## Re-running the simulations

`FrictionSim2D aiida rebuild <set> --output-dir rebuilt/ --hpc-scripts` writes a set's complete
input tree back out from its provenance, ready to run again.

## References

\[1\] Tang, K., Qi, W., Ru, G., & Liu, W. (2024). Screening of Low-Friction Two-Dimensional
Materials from High-Throughput Calculations Using Lubricating Figure of Merit. *Friction*,
12(8), 1897–1908. https://doi.org/10.1007/s40544-024-0901-8

\[2\] Hu, C., Yi, C., Bai, M., Lv, J., & Tang, D. (2020). Molecular dynamics study of the
frictional properties of multilayer MoS₂. *RSC Advances*, 10(30), 17418–17426.
https://doi.org/10.1039/D0RA00995D

\[3\] Serpini, E., Rota, A., Valeri, S., Ukraintsev, E., Rezek, B., Polcar, T., & Nicolini, P.
(2019). Nanoscale frictional properties of ordered and disordered MoS₂. *Tribology
International*, 136, 67–74. https://doi.org/10.1016/j.triboint.2019.03.004
