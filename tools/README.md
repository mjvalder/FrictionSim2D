# tools/

Repository tooling. Not part of the installed package (the wheel carries
`frictionsim2d/` only); nothing under `frictionsim2d/` imports from here.

- `materials_pack/`: staging and manifest builder for the materials pack that
  `FrictionSim2D fetch materials` installs (see its own README).
- `agent_data/`: a local, untracked, gitignored folder of assistant training runs
  (about 62 GB). It belongs to the training repo and is pending removal from here.

One-off work (legacy publication scripts, the integrity reference check, the AiiDA
service hooks, the Milestone-0 MLIP decks) was moved out of this repository on
2026-11-25.
