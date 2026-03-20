# AGENTS.md

This repository packages BOA (Bayesian Optimized Agents) as a reusable Python tool that runs against third-party Git repositories.

## Core Model

- BOA is the controller, onboarding flow, BO oracle layer, acceptance engine, experiment store, and trial runner.
- Patch-authoring endpoints such as Codex, Claude Code, Copilot, or DeepAgents do not decide acceptance or promotion.
- The coding agent is the search strategist. It may choose hypotheses and select a parent lineage from BOA-provided lineage options.
- The BO layer is decision support. It provides acquisition-style guidance, lineage ranking, family ranking, and numeric knob suggestions, but it does not author diffs directly.
- Target repositories are the runtime boundary. `boa init` converts an existing Git repository into a BOA-ready repository.
- BOA persists core runtime state inside the target repo under `.boa/` and uses a BOA-managed trial worktree path for editable trial checkouts.
- BOA is the source of truth for:
  - initialization and contract-file generation
  - branch creation
  - trial identifiers
  - diff computation
  - metric extraction
  - stage outcomes
  - post-stage reflection and lesson memory
  - acceptance / rejection / promotion
  - experiment persistence

## Public Contract

- `boa init` is the supported onboarding flow:
  - `boa init`
  - `boa init /path/to/repo`
- `boa init` launches an interactive `InquirerPy` setup wizard by default.
- `boa.md` is required in the target repo root and is injected verbatim into the patch-authoring prompt for `boa run`.
- `boa init` also captures the desired coding aggressiveness for `boa.md`: `light`, `normal`, or `aggressive`.
- `boa.config` is required in the target repo root, parsed as TOML, and must use `schema_version = 3`.
- The execution CLI is:
  - `boa run`
  - `boa run /path/to/target/repo`
- On interactive terminals, `boa run` should surface a live BOA-managed run monitor with trial/phase/stage progress, recent log events, and streamed agent or runner output when available. Non-interactive terminals should still receive readable progress logs.
- The BO oracle CLI is:
  - `boa tools recent-trials`
  - `boa tools list-lineage-options`
  - `boa tools suggest-parents`
  - `boa tools score-candidate-descriptor`
  - `boa tools rank-patch-families`
  - `boa tools propose-numeric-knob-regions`
- Planning prompts also include BOA-authored static context:
  - compact lesson memory derived from prior trial reflections and follow-up outcomes
  - an advisory BO suggestion report for likely next variations
  - a trial dataset with prior descriptors and recorded metric outcomes
- Managed branches are:
  - accepted branch: `boa/<run_tag>/accepted`
  - trial branch: `boa/<run_tag>/trial/<trial_id>`
- BOA promotes only the managed accepted branch. It must not write directly to the user’s main branch.
- BOA-owned runtime state lives under `.boa/`, with editable trial checkouts under `.boa/worktree/` and BOA-protected state under `.boa/protected/`. Agents and adapters must not create parallel hidden state elsewhere in the target repo.
- Only the current BOA schema is supported. Do not add compatibility code for superseded config shapes.

## Runtime Layout

The current code uses these BOA-owned paths:

- `.boa/worktree/<run_tag>/`
- `.boa/protected/artifacts/trials/<trial_id>/`
- `.boa/protected/agent_outputs/<trial_id>/`
- `.boa/protected/agent_traces/<trial_id>/search_calls.jsonl`
- `.boa/protected/prompts/<trial_id>/`
- `.boa/protected/store/experiments.sqlite`
- `.boa/protected/STOP`

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
- `boaresearch/search.py`: Bayesian oracle service, lineage ranking, and search tool tracing
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
- The selected coding aggressiveness is wizard-owned state and must be rendered into `boa.md` as BOA-authored guidance for patch size and rewrite scope.
- Existing BOA files are handled as:
  - valid current setup: `review`, `overwrite`, or `update`
  - invalid or foreign setup: raw preview plus `overwrite`

## Endpoint Contract

- Endpoints run in two phases:
  - planning on the accepted-branch workspace
  - execution inside the BOA-managed trial worktree
- After evaluation, BOA may invoke the configured agent in a reflection phase that inspects stage output and emits one compact trial reflection JSON object. Reflection is analysis-only and does not edit files, choose branches, or affect acceptance directly.
- Endpoints may inspect the workspace and edit only allowed files during execution.
- Endpoints must not make acceptance, promotion, persistence, or arbitrary branch-policy decisions.
- Endpoints may choose a parent lineage only from BOA-provided lineage options.
- Endpoints must emit exactly one candidate plan JSON object and exactly one candidate metadata JSON object at the BOA-designated paths for the current trial.
- Endpoints may call BOA tools multiple times during planning and execution.
- During planning, BOA injects compact lesson memory, an advisory static BO suggestion report, and a static trial dataset into the prompt. These are decision-support context only; the endpoint may ignore them.
- During execution, the endpoint is expected to leave at least one surviving tracked edit under allowed paths before emitting candidate metadata. If BOA validates a clean worktree, it may issue one explicit retry request before marking the trial as `no_patch`.
- BOA computes the actual diff, touched files, descriptor, validated branch lineage, and evaluation outcomes.

Candidate plan fields are:

- `hypothesis`
- `rationale_summary`
- `selected_parent_branch`
- optional `selected_parent_trial_id`
- `patch_category`
- `operation_type`
- `estimated_risk`
- optional `target_symbols`
- optional `numeric_knobs`
- optional `notes`
- `informed_by_call_ids`
- `addressed_lesson_ids`

Candidate metadata fields are:

- `hypothesis`
- `rationale_summary`
- `patch_category`
- `operation_type`
- `estimated_risk`
- optional `target_symbols`
- optional `numeric_knobs`
- optional `notes`
- `informed_by_call_ids`
- `addressed_lesson_ids`

If the candidate plan or candidate metadata is missing, malformed, cites unknown BO tool call ids, cites unknown lesson ids, selects an invalid parent branch, or is inconsistent with the produced diff, BOA treats the trial as failed or policy-rejected.

## Trial Semantics

A trial is valid only if all of the following hold:

- the planning phase completed without fatal adapter error
- the candidate plan was emitted and parsed successfully
- the selected parent branch is a BOA-known lineage option
- the endpoint completed without fatal adapter error
- the worktree contains a non-empty diff
- the diff touches only allowed files
- candidate metadata was emitted and parsed successfully
- the cited BO tool call ids resolve to the current trial trace
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
- After the highest completed stage or first failing stage is known, BOA may run a reflection pass over a capped excerpt of stage output and persist the resulting compact reflection alongside the trial.
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
- Search is agent-led and BO-assisted.
- The only supported BO oracle in the current schema is `bayesian_optimization`.
- BOA exposes search primitives rather than a top-level policy gate:
  - recent trial summaries
  - lineage option listing
  - parent suggestions
  - candidate descriptor scoring
  - patch family ranking
  - numeric knob region proposals
- BOA also materializes planning-only static search context from experiment memory:
  - compact lesson memory from prior trial reflections plus later addressed outcomes
  - an advisory next-variation report
  - a metric-labelled trial dataset of prior modifications
- BOA validates any lineage chosen by the agent before the trial branch is created.

## Config Discipline

- `boa.config` is parsed as typed TOML, not ad hoc dictionaries.
- `schema_version = 3` is required.
- Use the current `agent` and `runner` sections. Do not reintroduce deprecated `endpoint` or `remote` top-level config shapes.
- Use the current `search.oracle` contract. Do not reintroduce `search.policy`.
- Prefer extending typed config and schema surfaces rather than adding hidden conventions.

## Working Rules For Agents

- Preserve the target-repo contract around `boa init`, `boa run`, `boa.md`, `boa.config`, `.boa/`, and managed BOA branches.
- Keep BOA's init, acceptance, search, and reflection-memory decisions inside BOA, not inside endpoint-specific adapters.
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
- emit candidate plan and candidate metadata
- call at least one BO tool
- compute a diff
- run one stage
- parse one metric
- persist one trial result

If relevant integration surfaces were not exercised, say so explicitly.
