Trial id: {trial_id}

{objective_summary}

Execution attempt: {attempt_index}

Execution feedback:
{execution_feedback}

Selected parent branch: {parent_branch}

Selected parent trial: {parent_trial_id}

Read the approved candidate plan JSON from:
`{plan_output_path}`

BOA-owned candidate metadata artifact path for reference:
`{candidate_output_path}`

Approved candidate plan:
```json
{candidate_plan_json}
```

Recent trials:
{recent_trials}

Bootstrap BOA search context:
{bootstrap_tool_calls}

Preflight commands:
{preflight_commands}

Requirements:
1. Edit only inside the provided trial worktree.
2. You must make at least one surviving tracked modification under allowed paths before finalizing.
3. Before emitting candidate metadata, verify there is a real tracked diff against the parent branch in an allowed file, for example with `git status --short`.
4. If the worktree is clean or only scratch/protected files changed, continue editing instead of finalizing.
5. Do not touch protected paths.
6. Stay coherent with the approved candidate plan unless BOA tool guidance clearly justifies a refinement.
7. Run the configured preflight commands before finalizing.
8. Do not attempt to write directly to `{candidate_output_path}` when it is under `.boa/protected`; print exactly one candidate metadata JSON object to stdout and let BOA persist the artifact.
9. `informed_by_call_ids` must list the BOA tool call ids that materially informed the final patch. You may reuse bootstrap call ids above if they still materially informed the final patch.
10. `addressed_lesson_ids` must be a subset of the approved candidate plan's `addressed_lesson_ids`. Narrow the set if the final patch only addresses some of those lessons. Use `[]` if none remain applicable.
11. Do not commit or push; BOA owns branching, commits, and acceptance.
12. Do not try to create, inspect, or validate `.boa/` output directories. BOA already manages those paths.
13. If reading `{plan_output_path}` under `.boa/protected` is blocked, use the approved candidate plan embedded above. Do not attempt to write `{candidate_output_path}` directly; print the single final candidate metadata JSON object to stdout instead.
14. Do not scan `.boa/` to rediscover BOA paths. The plan path, metadata output path, and tool launcher path are already provided above.
