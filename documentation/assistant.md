# The Assistant tab

The Assistant is a chat pane in the GUI. It proposes changes to the Builder's
config and settings, finds structures and potentials, and adds folders or files
to your workspace. **Nothing changes until you click Apply on a card.**

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
  (about a minute per reply). The pane shows the setup commands and a
  Modelfile when you select it.

Settings live in `~/.config/FrictionSim2D/agent.yaml`.

## What it can do

- Set up AFM, indent, PES tip, sheet-on-sheet and PES sheet simulations from a
  sentence, including material sweeps.
- Find any config or settings parameter and propose a value.
- Search structures and potentials: your workspace first, then the bundled
  examples (labelled; verify them before production runs).
- Describe workspace folders, create folders, copy found files in.

It cannot run or submit jobs, delete, move or rename anything.

## Cards

Each proposal shows every field it changes, files it would write, validation
issues and the run count (configs / builds / jobs). **Apply** changes the open
config; **Apply as new config** starts from a blank one. If you edited the
Builder after the proposal, Apply re-checks it against your current config
first. Save the config with the Builder's Save as usual.

## Testing the assistant

    pytest tests/agent -q                                  # no network
    pytest tests/agent/evals -m agent_eval                 # live model
