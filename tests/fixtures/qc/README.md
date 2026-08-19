# QC classifier fixtures

All five are AFM-shaped (`time/nf/lfx/lfy/comx/comy/comz/tipx/tipy/tipz`),
cut from the first 200 rows of the real 2026 campaign file in
`../legacy_2026/` (h-MoS2, 30 nN, L2). Synthesis (seeded, rng 42):

- `converged.csv` — the real run, untouched (slides ~4.5 A over these rows).
- `seized.csv`    — tipx/tipy flattened to their first value. The natural
  seized cases are sheet runs, which the AFM-shaped classifier cannot consume.
- `detached.csv`  — nf replaced by 2 nN + N(0, 1.5) noise (the 2 nN
  poor-contact pathology: CV >> 0.30).
- `punched.csv`   — tipz set 5 A below comz.
- `drifted.csv`   — a `pe` column appended ramping at 2 eV/ps (legacy files
  carry no pe; represents new-schema output).

Regenerate with the script in the Task 10 commit message if the source
fixture changes.
