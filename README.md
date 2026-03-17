# BOA Researcher

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

## Config Shape

BOA uses a versioned v2 config only.

```toml
schema_version = 2

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
policy = "local_ranking"
```

## Notes

- `boa init` is the supported way to create BOA files. Users should not hand-author `boa.config` or `boa.md` for first-time setup.
- `boa run` expects the current BOA schema and does not load older config shapes.
- BOA keeps branch management, evaluation, and acceptance decisions inside BOA rather than in endpoint adapters.
