# tools/

Repository tooling. Not part of the installed package (the wheel carries
`frictionsim2d/` only); nothing under `frictionsim2d/` imports from here.

- `aiida_services/`: shell scripts and conda activate/deactivate hooks that start and stop the
  RabbitMQ/PostgreSQL services for the conda `-aiida` package variant.
- `integrity/`: `compare_reference.py`, a developer check of a run tree against a reference.
- `materials_pack/`: staging and manifest builder for the materials pack that
  `FrictionSim2D fetch materials` installs (see its own README).
- `publication/`: scripts that compare and re-stamp the legacy publication runs
  (`compare_legacy.py`, `legacy_provenance.py`).
- `agent_data/`: a local, untracked, gitignored training-data folder (about 62 GB) for the
  assistant's fine-tuning. It belongs to the training repo and is pending removal from here.
