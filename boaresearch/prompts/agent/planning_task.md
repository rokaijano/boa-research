Trial id: {trial_id}

{objective_summary}

Write the candidate plan JSON to:
`{plan_output_path}`

Recent trials:
{recent_trials}

Requirements:
1. Use BOA search tools before choosing a parent branch.
2. Choose `selected_parent_branch` from `boa tools list-lineage-options` only.
3. Do not commit, push, or finalize a patch in this phase.
4. Prefer the accepted branch when BOA memory is sparse or lineage evidence is weak.
5. Write exactly one candidate plan JSON object to `{plan_output_path}` and print the same JSON to stdout as a fallback.
6. `informed_by_call_ids` must list the BOA tool call ids that materially informed the plan.
7. Do not scan `.boa/` to rediscover BOA paths. The output path and tool launcher path are already provided above.
