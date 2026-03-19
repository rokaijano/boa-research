Trial id: {trial_id}

{objective_summary}

Selected parent branch: {parent_branch}

Selected parent trial: {parent_trial_id}

Read the approved candidate plan JSON from:
`{plan_output_path}`

Write the candidate metadata JSON to:
`{candidate_output_path}`

Approved candidate plan:
```json
{candidate_plan_json}
```

Recent trials:
{recent_trials}

Preflight commands:
{preflight_commands}

Requirements:
1. Edit only inside the provided trial worktree.
2. Do not touch protected paths.
3. Stay coherent with the approved candidate plan unless BOA tool guidance clearly justifies a refinement.
4. Run the configured preflight commands before finalizing.
5. Write exactly one candidate metadata JSON object to `{candidate_output_path}` and print the same JSON to stdout as a fallback.
6. `informed_by_call_ids` must list the BOA tool call ids that materially informed the final patch.
7. Do not commit or push; BOA owns branching, commits, and acceptance.
8. Do not scan `.boa/` to rediscover BOA paths. The plan path, metadata output path, and tool launcher path are already provided above.
