# The Assistant pane

The Assistant is a chat pane docked on the right of the GUI window, beside the
Builder, Database and Plots tabs. It is not a tab: one pane, and one
conversation, stays in view whichever tab is open. Show or hide it with the
**Assistant** button at the right end of the tab bar or **Ctrl+Shift+A**;
drag the divider to resize it. Each workspace remembers whether the pane was
open and how wide, and a new workspace starts with it collapsed. The pane's
header shows which tab you are on (display only, not sent to the model).
Applying a change never switches tabs: you stay where you are, and the pane
says what happened.

It edits the Builder's config and settings, finds structures and potentials,
reads your result sets and proposes plots for the Plots tab, and adds folders,
copied files or fetched structure files to your workspace. Config and settings
edits land in the Builder straight away (marked), and plots in the Plots tab,
each with an Undo (below); **file operations (folders, copies, fetched
files) always wait for you to click Apply.**

## Enabling it

    pip install -e ".[agent]"      # Python 3.11+

## Providers

- **Gemini (default, cloud).** Needs a free API key from
  https://aistudio.google.com/apikey. Set `GEMINI_API_KEY`, or paste the key
  when asked: it is stored owner-only in `~/.config/FrictionSim2D/agent.secrets`
  (honours `XDG_CONFIG_HOME`), never in your workspace or the repo. First use
  asks for consent: messages and tool results (config contents, file names)
  are sent to Google, and free-tier data may be used by Google. The free tier
  limits requests per minute; a reply costs 2-7 requests, and the pane shows a
  Retry button when the limit is hit.
- **Claude (cloud).** Anthropic API, a paid key (`ANTHROPIC_API_KEY`, or pasted
  when asked, as for Gemini). The model box offers `claude-opus-5-5` (default),
  `claude-sonnet-5-5` and `claude-haiku-5-5`; `anthropic.effort` sets how hard
  it thinks.
- **OpenAI (cloud).** OpenAI API, a paid key (`OPENAI_API_KEY`, or pasted when
  asked). The model box offers `gpt-5.5` (default), `gpt-5.4-mini` and
  `gpt-5.4-nano`; any other name can be typed. Consent is asked once, as for
  the other cloud providers.
- **Ollama (local).** Fully offline. The default model is the fine-tune
  `fs2d-assistant:qwen3.5-4b` (Qwen3.5-4B trained on this app's tools; a few
  seconds per short reply on a Radeon 780M). It is not in the Ollama library:
  the status line offers **Install** (below). When it can be neither found
  nor installed, the Assistant uses the stock `qwen3.5:4b` instead (fetch it
  with `ollama pull qwen3.5:4b`). The Assistant turns thinking off and sets
  temperature 0 itself. An `agent.yaml` from before the fine-tune that names
  `fs2d-qwen3-4b` or `qwen3:4b` is moved to the new default once, with a note
  in the chat; choose either again afterwards and it stays.

  Needs Ollama 0.40.7 or newer (older ones do not know the `qwen3.5`
  renderer, and Install fails with `unknown renderer`). The Assistant uses the
  `ollama` on `PATH`, else `~/.local/ollama/bin/ollama`. To keep it inside a
  conda env, `conda remove ollama` (conda-forge stops at 0.34) and unpack the
  release tarball into the env: `tar -I zstd -xf ollama-linux-amd64.tar.zst
  -C "$CONDA_PREFIX"`, which gives `bin/ollama` and `lib/ollama/` (CPU, CUDA
  and Vulkan runners).

### Fine-tuned models

A fine-tune ships as a bundle in `assistant_models/<tag>/`:
`model.gguf` (the weights, gitignored), a `Modelfile` whose `FROM
./model.gguf` is relative, and `manifest.json` (base model, the tool-schema
hash it was trained for, the weights' sha256). Three ship: `qwen3.5-2b`, `qwen3.5-4b`
(the default) and `qwen3.5-9b`, Qwen3.5 fine-tunes given the same prompt as the
hosted models, reference text included. Every bundle there is offered by name. When the selected
one is missing, the status line reads `fs2d-assistant:<tag> is not installed`
with an **Install** button: it checks the weights against the manifest (a
bundle still syncing is refused) and runs `ollama create` with the status
line showing progress. By hand, the same is

    ollama create fs2d-assistant:qwen3.5-4b -f assistant_models/qwen3.5-4b/Modelfile

If its tool-schema hash is not this app's (the tools changed since
it was trained), the chat says so once: results may degrade, and retraining
is the fix. Install is offered only for a local server.

Training data, training and export live in a separate repository
(`FrictionSim2D-agent-training`); its `export.py` writes the bundles here.
The knowledge corpus is built there too (`corpus.py`) and installed to
`~/.local/share/FrictionSim2D/knowledge`.

Syncthing: `assistant_models/` stays synced, so a new bundle arrives on its
own; a `.stignore` at the repo root (never synced, gitignored) can exclude
`tests/agent/evals/.results`.

Not done yet, for later: publishing a bundle so a machine without the repo can
`ollama pull` it. That is `ollama push <namespace>/fs2d-assistant:<tag>` from
an ollama.com account, then an entry in `PUBLISHED` in `frictionsim2d/agent/bundles.py`
(the GUI then pulls it), plus a Hugging Face or Zenodo mirror of the GGUF and
manifest for a citable DOI. A plain `ollama pull hf.co/...` is not enough:
Ollama would build the chat template from the GGUF, not the bundle's
Modelfile, which is a known training/serving mismatch.

### Ollama starts and stops itself

The Assistant starts `ollama serve` when you select Ollama, start a New chat
with it, or send a message while it is down (the message goes as soon as it
is up; if the start fails, a red line says why and your text goes back into
the input). The status line reads `Starting Ollama…` meanwhile; nothing in
the window blocks. It is started throttled for a laptop: `nice -n 10`,
`OLLAMA_IGPU_ENABLE=1 OLLAMA_VULKAN=1 OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=2m` and `OLLAMA_CONTEXT_LENGTH` from
`ollama.context_tokens`, with its output in
`~/.cache/FrictionSim2D/ollama.log` (honours `XDG_CACHE_HOME`).

A server the Assistant started is stopped after `ollama.idle_minutes`
(default 10) without a reply, and when the window closes; the status then
reads `Ollama stopped after N min idle (starts again on your next message)`.
It is not stopped while a model is still loaded (another window or client
used it in the last couple of minutes). It also ends if the GUI crashes or is
killed; if it survives anyway, its pid is in `~/.cache/FrictionSim2D/ollama.pid`
and the next session takes it over. **A server you started yourself is never
stopped.** Only a server on this machine (`localhost`, `127.0.0.1`, `::1` in
`ollama.base_url`) is ever started; a remote one is reported unreachable. If
the port answers but is not Ollama, the status says so and nothing is started.

### Status line

One line under the provider and model pickers: `Ollama ready (fs2d-assistant:qwen3.5-4b)`,
`Starting Ollama…`, `Ollama not running — starts on your next message`, or
why it could not start. **Check again** re-asks Ollama (it never starts it).
**Details** unfolds the start command, a **Copy start command** button, the
log path and, when the model is missing, the `ollama pull` line. Gemini shows
one line too (`Gemini: key set`).

The model box is editable: for Ollama it lists the installed models (from
`/api/tags`); for Gemini it shows the configured model. Type any other name
to use it.

### agent.yaml

Settings live in `~/.config/FrictionSim2D/agent.yaml`:

| key | default | meaning |
|---|---|---|
| `provider` | `gemini` | `gemini`, `anthropic`, `openai` or `ollama` |
| `ask_before_apply` | `false` | changes wait as cards (below) |
| `max_requests_per_turn` | `50` | model requests one reply may make |
| `ollama.model` | `fs2d-assistant:qwen3.5-4b` | local model |
| `ollama.idle_minutes` | `10` | stop a server the Assistant started after this long idle |
| `ollama.context_tokens` | `16384` | `OLLAMA_CONTEXT_LENGTH` of a started server |
| `ollama.keep_alive` | `2m` | `OLLAMA_KEEP_ALIVE` of a started server |
| `gemini.model` | `gemini-3.5-flash-lite` | cloud model |
| `anthropic.model` | `claude-opus-5-5` | cloud model |
| `openai.model` | `gpt-5.5` | cloud model |
| `materials_project.enabled` | `true` | allow Materials Project / 2DMatPedia lookups |

### Step cap

Each tool call the model makes is one request ("step"). A reply stops after
`max_requests_per_turn` of them with `Stopped after N steps to save quota —
type 'continue' to carry on.` The work so far is kept, so typing `continue`
picks up where it stopped. Raise the value in `agent.yaml` for long setups;
lower it to protect a free-tier quota.

## What it can do

- Set up AFM, indent, PES tip, sheet-on-sheet and PES sheet simulations from a
  sentence, including material sweeps.
- Find any config or settings parameter and set it.
- Search structures and potentials: your workspace first, then the bundled
  examples (labelled; verify them before production runs).
- Describe workspace folders, create folders, copy found files in.
- Look a formula up on Materials Project (stability, band gap, moduli) and
  2DMatPedia, and propose saving a 2DMatPedia monolayer as
  `materials/<formula>_<dm_id>.cif` (see below).
- Turn "low / mid / high load" into numbers from reference load bands (see
  below).
- Find datasets and simulation roots in the workspace, summarise them, and
  propose plots of them (see below).

It cannot run or submit jobs, delete, move or rename anything.

### Materials Project and 2DMatPedia

Needs `pip install mp-api` (pulls in pymatgen) and a free key from
https://next-gen.materialsproject.org/api, looked up in this order:
`$MP_API_KEY`, `~/.mp_api_key`, `PMG_MAPI_KEY` in `~/.config/.pmgrc.yaml`.
Without one the tools reply `No Materials Project key: set MP_API_KEY or put
it in ~/.mp_api_key`. **Only formulas and ids are sent to Materials Project**
(the key goes in a request header and is never logged or cached).
2DMatPedia is read from the MPContribs project `2dmatpedia`; its full listing
is cached for 7 days in `~/.cache/FrictionSim2D/2dmatpedia.json` (honours
`XDG_CACHE_HOME`). A fetched structure is a new file proposal: it waits for
Apply, lands only inside the workspace and never overwrites. Its card names
what will be written -- reduced formula, number of sites and lattice lengths
a/b/c, read from the CIF with pymatgen -- never the CIF text itself. The file
is written to a temporary file beside the target and then published, so a
failed write leaves nothing behind; it gets the permissions of any new file
(`0666` minus your umask). Without pymatgen a 2DMatPedia lookup says so before
anything is downloaded. Stop is honoured between listing pages and before each
request; a request already sent (a single Materials Project summary search
included) runs to its end. Turn lookups off with
`materials_project.enabled: false`.

### Load bands

Asked for a "low", "mid" or "high" load (also small/light, moderate/medium/
intermediate, large/heavy), the Assistant takes the values from reference
bands in `frictionsim2d/core/scaling.py`, spread linearly over the band:

| band | nominal pressure F/R² | at R = 25 Å | where the numbers come from |
|---|---|---|---|
| low | 1.6-3.2 GPa | 10-20 nN | the baseline loads (10, 20, 30 nN), chosen for good contact; 2-5 nN give poor contact |
| mid | 3.2-8 GPa | 20-50 nN | the mid-load core window the reference trends are fitted on |
| high | 8-16 GPa | 50-100 nN | from the top of the mid window to the example campaign's maximum (100 nN) |

For the AFM family (AFM, indent, PES tip) the band is a **nominal pressure,
F/R²**, calibrated on the reference R = 25 Å tip, so the load scales with the
tip radius squared: a 50 Å tip gets 40-80 nN for "low". With no tip radius
set, 25 Å is assumed (and said); a swept radius uses the first value and lists
the others. This is simple tip-size scaling, **not a contact model** and not
measured regimes: stiffness, sheet size and contact area are ignored. A
temperature or driving spring that differs from the reference (300 K, 8 or
50 N/m) keeps the numbers and adds a warning. The Assistant says where the
numbers come from and asks you to confirm. Sheet-on-sheet pressure, speed and
temperature have no band yet: it offers the example config's values and asks
for a range instead.

### Result sets and analysis

All read-only; each result is at most 1,500 characters and shows paths as
`workspace:/...`.

- **find_result_sets** walks the workspace up to 4 folder levels deep (no
  dot-folders except `.preview/_datasets`, where exported sets go; no
  symlinks that leave the workspace) and lists up to 15 hits: *datasets*
  (a folder with `outputs/output_full_*.json`, as `postprocess read` writes)
  with their sizes and run counts, and *simulation roots* with
  `provenance.json` runs per model (counted up to 1000). Run counts parse
  files of at most 20 MB, 50 MB in one call; past that a dataset shows its
  size in MB instead.
- **summarise_results** on a dataset (or one `output_full_*.json`): per file,
  status counts, materials, temperatures and loads, then a run table (complete
  runs first, 15 rows by default), filterable by material and size. A file
  over 200 MB is reported as too large; at most 400 MB are parsed per call,
  and files past that are listed as `not read (budget)`. On a simulation root
  that is not post-processed yet it lists runs from their folder names, with
  the points expected and the results files present, and suggests running
  `FrictionSim2D postprocess read` on it.
- **plot_types** lists every plot type with one line each; **plot_schema**
  lists the fields one type accepts, from the plot schema.

### Plot proposals

A proposed plot is checked against the plot schema, and each dataset it reads
must be a dataset folder inside the workspace. It lands in the **Plots tab**
(never a file): the datasets it needs are added under a free label, the plot
is appended, selected and rendered, and the tab you are on does not change --
the chat line gets a **Show** link to the Plots tab when you are elsewhere.
If you are editing a plot (cursor in the Plots form, a refused value there,
or a dialog open), it waits as a card instead (**Add plot**, or **Add
anyway** when it has issues). With *Ask before applying* every plot is a
card. **Undo** puts the Plots tab back as it was before the plot; if you
edited the Plots tab since, it asks first.

## How changes land

- **Direct edits (default).** A config or settings change is applied to the
  open config as soon as the assistant makes it. If you edited the Builder
  while it was working, the change is re-checked against your current config
  first. A change that cannot be applied leaves the Builder untouched and shows
  a red line in the chat saying why. If you are in the middle of editing (the
  cursor is in a Builder or Settings field, a dialog is open, or a field was
  refused), the change is not applied but kept as a card in the Pending strip.
  If a file error stops a change part-way, the line says so and Undo is offered.
- **Yellow highlights.** Every field the assistant set turns pale yellow; its
  tooltip reads `Set by assistant: <before> → <after>`. Validation errors still
  show their red border on top. Sections holding a highlighted field unfold.
  A highlight goes when you edit that field, when you click **Accept** (clears
  them all; the values stay), or when you save the config with the Builder's
  Save. A switch of simulation model has no field; it is named in the chat line.
  A field changed twice keeps its original value in the tooltip (`a → c`), and
  undoing the second change puts the first change's mark back.
- **Chat line and Undo.** Each applied change adds one line, e.g. `Set 3
  fields: [general] force, [2D] x, [2D] y (+ model → sheetonsheet)`, with an
  **Undo** link. Undo puts the Builder back exactly as it was just before that
  change -- config, settings, and a material-list file it wrote (restored, or
  deleted if the change created it). If you edited the Builder after the
  change, Undo first asks, naming the fields it would also revert (default No).
  Only the most recent change can be undone; older lines say "undo newer
  changes first" until the newer ones are undone. Saving the config (or opening
  another one) and writing the settings to the workspace -- Save to workspace,
  or the automatic write before Render/Generate -- end Undo for every config
  and settings line so far; plot lines keep their Undo, since it restores only
  the Plots tab. New chat drops them all.
- **Ask before applying.** Tick this in the pane's header (remembered in
  `agent.yaml` as `ask_before_apply`) to review changes first: each one waits
  as a card showing every field it changes, files it would write, validation
  issues and the run count (configs / builds / jobs). **Apply** changes the
  open config, **Apply as new config** starts from a blank one (asking first if
  the open config has unsaved edits), **Discard** drops it. If you edited the
  Builder after the proposal, Apply re-checks it and shows the re-checked card
  to apply again. An applied card is highlighted and undoable as above.
- **File operations still ask.** Creating folders, copying files in and
  writing a fetched structure always arrive as cards, whatever the switch
  says. None of them overwrites a file; a copy or write that fails part-way
  leaves no partial file, so it can simply be retried.

Cards sit in a compact **Pending** strip above the input, shown only while
something is waiting; a card leaves it as soon as it is applied or discarded,
and the result is written to the chat.

## Testing the assistant

    pytest tests/agent -q                                  # no network
    pytest tests/agent/evals -m agent_eval                 # live model

The eval tier judges one provider/model on ~60 cases in groups (setup,
params, sweeps, materials, workspace, analysis, materials_db, must_ask,
declines): setup per family, parameters, sweeps and reference load bands,
materials, workspace, analysis and plots, Materials Project against a fake
client, prompts that must be asked back, and requests that must be declined.
Switches: `AGENT_EVAL_PROVIDER` (`gemini` default, `anthropic`, `openai`, `ollama`), `AGENT_EVAL_MODEL`,
`AGENT_EVAL_REPEATS` (default 3), `AGENT_EVAL_PAUSE` (seconds between turns).
Each run writes `tests/agent/evals/.results/<provider>-<model>-<timestamp>.json`
(gitignored; failed runs keep what the model did) and prints pass rates per
case, per group and overall, warm p50/p95 latency (the first, cold turn left
out; Ollama also gets one unrecorded warm-up turn), and mean requests and
input/output tokens per turn. A rate-limited turn is skipped, not failed, and
the table then says the pass rate is not comparable.

On the Gemini free tier a full run (cases x 3 repeats) exhausts the daily
quota. Use one repeat, one group at a time, and a pause between turns:

    AGENT_EVAL_REPEATS=1 AGENT_EVAL_PAUSE=20 pytest tests/agent/evals -m agent_eval -k "afm_ or sheet_"
    AGENT_EVAL_REPEATS=1 AGENT_EVAL_PAUSE=20 pytest tests/agent/evals -m agent_eval -k "plot_ or summarise"

(`-k` matches case names; the group of each case is in `tests/agent/evals/cases.py`.)
