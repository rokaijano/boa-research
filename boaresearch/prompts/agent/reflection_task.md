BOA-owned trial reflection artifact path for reference:
`{reflection_output_path}`

Reflection input:
```json
{reflection_payload}
```

Requirements:
1. Analyze only the provided evidence.
2. Keep `behavior_summary`, `primary_problem`, and `outcome` concise and concrete.
3. Keep each `suggested_fixes` item action-oriented and short.
4. Keep each `evidence` item short enough for prompt reuse.
5. If the patch failed or was rejected, explain what still looks under-optimized or insufficient.
6. Print exactly one final JSON object to stdout. Do not wrap it in commentary.
