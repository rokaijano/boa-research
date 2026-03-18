# BOA Research

`boaresearch` packages BOA (Bayesian Optimized Agents) as a reusable controller for third-party Git repositories.

## Install

```bash
pip install -e .
```

For DeepAgents support:

```bash
pip install -e ".[deepagents]"
```

## Onboarding

BOA is initialized through an interactive `InquirerPy` wizard:

```bash
boa init
boa init /path/to/repo
```

`boa init` resolves the Git root, shows the banner, lets the user choose a coding agent and runner mode, runs preflight checks, analyzes the repository, walks through a review step, and writes:

- `boa.config`
- `boa.md`
- `.boa/`

## Run

```bash
boa run
boa run /path/to/target/repo
```

`boa run` loads the current BOA config, injects `boa.md` into the patch-authoring prompt, creates managed BOA branches, evaluates candidates with the configured local or SSH runner, and persists runtime state inside `.boa/`.

BOA remains the execution authority and experiment memory. The coding agent chooses hypotheses and may steer lineage using BOA-provided search tools, but BOA still validates diffs, runs stages, decides acceptance, and promotes branches.

## Tools

BOA exposes its Bayesian oracle to agents and to the shell through `boa tools`:

```bash
boa tools recent-trials
boa tools list-lineage-options
boa tools suggest-parents
boa tools score-candidate-descriptor
boa tools rank-patch-families
boa tools propose-numeric-knob-regions
```

These commands accept JSON on stdin and return JSON on stdout with a `call_id` so agent plans and candidates can cite which BO calls informed the patch.

## Examples

The repository includes three progressively harder deep learning target repos under `examples/`. These are ordinary example projects intended to be onboarded with `boa init`; they are not part of BOA's runtime or schema.

Progression:

- `examples/mnist_mlp_or_cnn`: simple CPU-friendly MNIST classification with a small CNN.
- `examples/cifar10_resnet`: harder image classification on CIFAR-10 with stronger architecture and regularization surfaces.
- `examples/tiny_transformer_text`: the most complex example, a small transformer-based text classifier that is practical on CPU for smoke checks and better suited to GPU for serious BOA runs.

Typical flow:

```bash
cd examples/mnist_mlp_or_cnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

boa init .
boa run .
```

Each example README documents:

- the expected hardware profile
- the default `train.py` / `eval.py` contract
- the metric artifact path under `reports/metrics.json`
- likely BOA optimization surfaces

## Config Shape

BOA uses a versioned v3 config only.

```toml
schema_version = 3

[run]
tag = "default"
max_trials = 3

[agent]
preset = "codex"
runtime = "cli"
command = "codex"

[guardrails]
allowed_paths = ["src", "tests"]
protected_paths = [".boa", "vendor"]

[runner]
mode = "local"

[runner.local]
activation_command = "source .venv/bin/activate"

[runner.ssh]
host_alias = "train-box"
repo_path = "~/target-repo"
git_remote = "origin"
runtime_root = ".boa/remote"

[runner.scout]
enabled = true
commands = ["python train.py", "python eval.py"]
timeout_seconds = 1800

[[metrics]]
name = "accuracy"
source = "json_file"
path = "reports/metrics.json"
json_key = "accuracy"

[objective]
primary_metric = "accuracy"
direction = "maximize"

[search]
oracle = "bayesian_optimization"
max_history = 50
parent_suggestion_count = 5
family_suggestion_count = 5
knob_region_count = 5
risk_penalty = 0.25
family_bonus = 0.1
lineage_bonus = 0.05
exploration_weight = 0.35
observation_noise = 0.15
```

## Notes

- `boa init` is the supported way to create BOA files. Users should not hand-author `boa.config` or `boa.md` for first-time setup.
- `boa run` expects the current BOA schema and does not load older config shapes.
- BOA keeps branch management, evaluation, and acceptance decisions inside BOA rather than in endpoint adapters.
- Agents can choose a parent lineage only from BOA-provided lineage options and must cite the BO tool calls that informed the plan and final candidate.
