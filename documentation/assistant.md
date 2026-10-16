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
and adds folders or files to your workspace. Config and settings edits land in
the Builder straight away, marked and undoable (below); **file operations always
wait for you to click Apply.**

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
- **Ollama (local, experimental).** Fully offline but slow on a laptop iGPU
  (about a minute per reply). The default model is `qwen3:4b`, used as Ollama
  ships it: the Assistant turns thinking off (`/no_think`) and sets
  temperature 0 itself, so no Modelfile or `ollama create` is needed. Fetch it
  once with `ollama pull qwen3:4b` (the user-local install lives at
  `~/.local/ollama/bin/ollama`). An `agent.yaml` that still names the retired
  `fs2d-qwen3-4b` falls back to `qwen3:4b` when that model is not installed,
  with a note in the chat.

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

One line under the provider and model pickers: `Ollama ready (qwen3:4b)`,
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
| `provider` | `gemini` | `gemini` or `ollama` |
| `ask_before_apply` | `false` | changes wait as cards (below) |
| `max_requests_per_turn` | `50` | model requests one reply may make |
| `ollama.model` | `qwen3:4b` | local model |
| `ollama.idle_minutes` | `10` | stop a server the Assistant started after this long idle |
| `ollama.context_tokens` | `8192` | `OLLAMA_CONTEXT_LENGTH` of a started server |
| `ollama.keep_alive` | `2m` | `OLLAMA_KEEP_ALIVE` of a started server |
| `gemini.model` | `gemini-3.5-flash-lite` | cloud model |
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
Apply, lands only inside the workspace and never overwrites. Turn lookups off
with `materials_project.enabled: false`.

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
  or the automatic write before Render/Generate -- end Undo for every line so
  far; New chat drops them.
- **Ask before applying.** Tick this in the pane's header (remembered in
  `agent.yaml` as `ask_before_apply`) to review changes first: each one waits
  as a card showing every field it changes, files it would write, validation
  issues and the run count (configs / builds / jobs). **Apply** changes the
  open config, **Apply as new config** starts from a blank one (asking first if
  the open config has unsaved edits), **Discard** drops it. If you edited the
  Builder after the proposal, Apply re-checks it and shows the re-checked card
  to apply again. An applied card is highlighted and undoable as above.
- **File operations still ask.** Creating folders and copying files in always
  arrive as cards, whatever the switch says.

Cards sit in a compact **Pending** strip above the input, shown only while
something is waiting; a card leaves it as soon as it is applied or discarded,
and the result is written to the chat.

## Testing the assistant

    pytest tests/agent -q                                  # no network
    pytest tests/agent/evals -m agent_eval                 # live model
