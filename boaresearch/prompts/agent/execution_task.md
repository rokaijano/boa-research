Trial id: {trial_id}

{objective_summary}

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
2. Do not touch protected paths.
3. Stay coherent with the approved candidate plan unless BOA tool guidance clearly justifies a refinement.
4. Run the configured preflight commands before finalizing.
5. Do not attempt to write directly to `{candidate_output_path}` when it is under `.boa/protected`; print exactly one candidate metadata JSON object to stdout and let BOA persist the artifact.
6. `informed_by_call_ids` must list the BOA tool call ids that materially informed the final patch. You may reuse bootstrap call ids above if they still materially informed the final patch.
7. Do not commit or push; BOA owns branching, commits, and acceptance.
8. Do not try to create, inspect, or validate `.boa/` output directories. BOA already manages those paths.
9. If reading `{plan_output_path}` under `.boa/protected` is blocked, use the approved candidate plan embedded above. Do not attempt to write `{candidate_output_path}` directly; print the single final candidate metadata JSON object to stdout instead.
10. Do not scan `.boa/` to rediscover BOA paths. The plan path, metadata output path, and tool launcher path are already provided above.
