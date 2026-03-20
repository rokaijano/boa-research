Trial id: {trial_id}

{objective_summary}

BOA-owned candidate plan artifact path for reference:
`{plan_output_path}`

Recent trials:
{recent_trials}

Bootstrap BOA search context:
{bootstrap_tool_calls}

BOA lesson memory:
```json
{lesson_memory}
```

Static BOA suggestion report:
```json
{bo_suggestion_report}
```

Static BOA trial dataset:
```json
{trial_dataset}
```

Requirements:
1. Use BOA search information before choosing a parent branch. The static BOA suggestion report is advisory only; you may follow it, refine it, or ignore it if the repository evidence points elsewhere. If shell or CLI tool execution is blocked, use the provided bootstrap BOA search context.
2. Choose `selected_parent_branch` from the provided lineage options or from `boa tools list-lineage-options` only.
3. Do not commit, push, or finalize a patch in this phase.
4. Prefer the accepted branch when BOA memory is sparse or lineage evidence is weak.
5. Do not attempt to write directly to `{plan_output_path}` when it is under `.boa/protected`; print exactly one candidate plan JSON object to stdout and let BOA persist the artifact.
6. `informed_by_call_ids` must list the BOA tool call ids that materially informed the plan. You may cite the bootstrap call ids above if they materially informed your decision.
7. If you are directly testing a lesson from BOA lesson memory, cite its ids in `addressed_lesson_ids`. Use `[]` if no listed lesson directly applies.
8. Do not try to create, inspect, or validate `.boa/` output directories. BOA already manages those paths.
9. If a shell or tool suggests creating files under `.boa/protected`, stop and print the single final JSON object to stdout instead.
10. Do not scan `.boa/` to rediscover BOA paths. The output path and tool launcher path are already provided above.
