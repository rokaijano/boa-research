# AGENTS.md

This repository packages BOA (Bayesian Optimized Agents) as a reusable Python tool that runs against third-party Git repositories.

## Core Model

- BOA is the controller, onboarding flow, search layer, acceptance engine, experiment store, and trial runner.
- Patch-authoring endpoints such as Codex, Claude Code, Copilot, or DeepAgents do not decide acceptance, promotion, or final lineage.
- The BO layer is decision support. It chooses parent lineage and search hints, but it does not author diffs directly.
- Target repositories are the runtime boundary. `boa init` converts an existing Git repository into a BOA-ready repository.
- BOA persists runtime state inside the target repo under `.boa/`.
- BOA is the source of truth for:
  - initialization and contract-file generation
  - branch creation
  - trial identifiers
  - diff computation
  - metric extraction
  - stage outcomes
  - acceptance / rejection / promotion
  - experiment persistence

## Public Contract

- `boa init` is the supported onboarding flow:
  - `boa init`
  - `boa init /path/to/repo`
- `boa init` launches an interactive `InquirerPy` setup wizard by default.
- `boa.md` is required in the target repo root and is injected verbatim into the patch-authoring prompt for `boa run`.
- `boa.config` is required in the target repo root, parsed as TOML, and must use `schema_version = 2`.
- The execution CLI is:
  - `boa run`
  - `boa run /path/to/target/repo`
- Managed branches are:
  - accepted branch: `boa/<run_tag>/accepted`
  - trial branch: `boa/<run_tag>/trial/<trial_id>`
- BOA promotes only the managed accepted branch. It must not write directly to the user’s main branch.
- BOA-owned runtime state lives under `.boa/`. Agents and adapters must not create parallel hidden state elsewhere in the target repo.
- Only the current BOA schema is supported. Do not add compatibility code for superseded config shapes.

## Runtime Layout Under `.boa/`

The current code uses these BOA-owned paths:

- `.boa/artifacts/trials/<trial_id>/`
- `.boa/prompts/<trial_id>/`
- `.boa/store/experiments.sqlite`
- `.boa/worktrees/<run_tag>/`
- `.boa/STOP`

If this layout changes materially, update this file in the same change.

## Package Layout

- `pyproject.toml`: package metadata and the `boa` console script
- `boaresearch/cli.py`: CLI entrypoint for `boa init` and `boa run`
- `boaresearch/init_app.py`: InquirerPy onboarding wizard
- `boaresearch/init_services.py`: repo detection, preflight, analysis, file writing, and validation for init
- `boaresearch/init_banner.py`: terminal banner rendering for `boa init`
- `boaresearch/schema.py`: current config schema, runtime dataclasses, and agent context models
- `boaresearch/loader.py`: `boa.config` loading and validation
- `boaresearch/controller.py`: top-level orchestration loop
- `boaresearch/runner.py`: local and SSH stage runners
- `boaresearch/store.py`: SQLite experiment memory
- `boaresearch/acceptance.py`: staged scout/confirm/promoted acceptance logic
- `boaresearch/search.py`: pluggable search policies
- `boaresearch/descriptors.py`: diff-to-`PatchDescriptor` extraction
- `boaresearch/metrics.py`: JSON/regex/metric-file extraction
- `boaresearch/agents/`: endpoint adapters for CLI tools and DeepAgents
- `tests/`: unit coverage for config loading, runner modes, init flows, banner rendering, acceptance, search, descriptors, metrics, and persistence

## Init Contract

- The interactive wizard owns onboarding state through typed models:
  - `InitSetupSelection`
  - `RepoAnalysisProposal`
  - `ReviewedInitPlan`
- The wizard must not mutate final config state ad hoc from widget values.
- Repo analysis is separate from patch authoring. It returns a structured proposal that BOA merges with wizard selections before writing files.
- File generation for `boa.config` and `boa.md` must be deterministic. Agent analysis may suggest content, but BOA writers own the final rendering.
- Existing BOA files are handled as:
  - valid current setup: `review`, `overwrite`, or `update`
  - invalid or foreign setup: raw preview plus `overwrite`

## Endpoint Contract

- Endpoints run locally inside the BOA-managed trial worktree.
- Endpoints may inspect the worktree and edit only allowed files.
- Endpoints must not make acceptance, promotion, persistence, or branch-policy decisions.
- Endpoints must emit exactly one candidate metadata JSON object at the BOA-designated path for the current trial.
- BOA computes the actual diff, touched files, descriptor, branch lineage, and evaluation outcomes.

Candidate metadata fields are:

- `hypothesis`
- `rationale_summary`
- `patch_category`
- `operation_type`
- `estimated_risk`
- optional `target_symbols`
- optional `numeric_knobs`
- optional `notes`

If candidate metadata is missing, malformed, or inconsistent with the produced diff, BOA treats the trial as failed.

## Trial Semantics

A trial is valid only if all of the following hold:

- the endpoint completed without fatal adapter error
- the worktree contains a non-empty diff
- the diff touches only allowed files
- candidate metadata was emitted and parsed successfully
- the configured stage command executed
- metric extraction succeeded for the stage being evaluated

The following trial outcomes should remain distinct in persistence and reporting:

- `agent_failed`
- `no_patch`
- `invalid_patch`
- `policy_rejected`
- `stage_failed`
- `metric_missing`
- `timed_out`
- `completed_rejected`
- `completed_accepted`

## Trial Execution

- Trial execution can run locally or over SSH, depending on `runner.mode`.
- BOA remains the source of truth for trial state and acceptance.
- Stages are `scout`, optional `confirm`, and optional `promoted`.
- Each stage runs configured commands, captures stdout and stderr, enforces timeouts, and records simple resource metadata.
- Metrics can come from:
  - JSON files
  - regexes over logs
  - explicit metric files

## Acceptance And Search

- Acceptance compares stage results using the configured primary metric direction (`maximize` or `minimize`).
- Acceptance may also apply:
  - a raw threshold
  - a cost penalty
  - a minimum improvement delta
- Lower stages qualify advancement. The highest enabled successfully completed stage is the canonical evaluation stage for that trial.
- Endpoint self-reports never determine acceptance.
- Search policies currently include:
  - `random`
  - `greedy_best_first`
  - `local_ranking`

`local_ranking` is a BOA policy over BOA-maintained descriptors and outcomes. It is not delegated to endpoint adapters.

## Config Discipline

- `boa.config` is parsed as typed TOML, not ad hoc dictionaries.
- `schema_version = 2` is required.
- Use the current `agent` and `runner` sections. Do not reintroduce deprecated `endpoint` or `remote` top-level config shapes.
- Prefer extending typed config and schema surfaces rather than adding hidden conventions.

## Working Rules For Agents

- Preserve the target-repo contract around `boa init`, `boa run`, `boa.md`, `boa.config`, `.boa/`, and managed BOA branches.
- Keep BOA’s init, acceptance, and search decisions inside BOA, not inside endpoint-specific adapters.
- Do not let endpoint adapters silently define runtime paths, acceptance behavior, or branch policy.
- If a change materially affects architecture, workflow, public contracts, runtime paths, branch policy, config schema, persistence schema, or testing expectations, update this `AGENTS.md` in the same change.
- Treat `AGENTS.md` as a living contract. When major changes are made, upgrade this file so future agents inherit an accurate description of the repo.

## Verification

Before wrapping up a substantive change, prefer running:

- `python3 -m unittest discover -s tests -v`
- `python3 -m compileall boaresearch tests`

Minimum useful smoke path when relevant:

- run `boa init` services against a Git repo
- write `boa.config` and `boa.md`
- load `boa.config`
- create a trial
- emit candidate metadata
- compute a diff
- run one stage
- parse one metric
- persist one trial result

If relevant integration surfaces were not exercised, say so explicitly.
