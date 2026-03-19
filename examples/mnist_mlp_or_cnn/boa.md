# BOA Guidance

Optimize for validation `accuracy` written to `reports/metrics.json`.

Use the standard flow:
- `python train.py --device cpu`
- `python eval.py --device cpu`

Work inside `src/mnist_demo/` unless a change outside that package is clearly necessary. Keep the CLI contract, checkpoint path behavior, and metric file schema stable.

Do not switch scoring runs to `--use-fake-data`; fake data is only for smoke tests. Preserve the metric keys `accuracy`, `loss`, and `runtime_seconds`, and continue writing them to `reports/metrics.json`.

Favor direct improvements to model architecture, optimizer settings, regularization, preprocessing, and training logic over broad project rewrites. Keep runs practical for a laptop CPU.

## Agent Edit Strategy

- Aggressiveness: `normal`
- Guidance: Prefer standard-sized changes. Improve the target area directly without forcing either ultra-minimal edits or large rewrites.

### Mode Rules

- Aim for direct, standard-sized improvements in the most relevant code paths.
- Refactor only when it materially helps the target change land cleanly.
- Keep diffs readable and bounded even when multiple nearby edits are needed.

## Caveats

- `eval.py` requires a checkpoint at `artifacts/best_model.pt`, so training must run successfully before evaluation.
- Both `train.py` and `eval.py` overwrite `reports/metrics.json`; BOA should read the post-eval file.
- The default device mode is `auto`; pinning `--device cpu` makes local runs more comparable.
- The default config trains on a subset of MNIST (`train_size=2048`, `val_size=512`), so accuracy is measured on that configured subset unless editable defaults are changed.

## Agent Edit Strategy

- Aggressiveness: `normal`
- Guidance: Prefer standard-sized changes. Improve the target area directly without forcing either ultra-minimal edits or large rewrites.

### Mode Rules

- Aim for direct, standard-sized improvements in the most relevant code paths.
- Refactor only when it materially helps the target change land cleanly.
- Keep diffs readable and bounded even when multiple nearby edits are needed.

## BOA Setup Summary

- Train command: `python train.py --device cpu`
- Eval command: `python eval.py --device cpu`
- Primary metric: `accuracy` (maximize)
- Editable paths: src/mnist_demo/cli.py, src/mnist_demo/config.py, src/mnist_demo/data.py, src/mnist_demo/model.py, src/mnist_demo/runtime.py, src/mnist_demo/training.py
- Protected paths: train.py, .boa, eval.py, .git, requirements.txt, reports/metrics.json, README.md, tests/test_smoke.py
- Likely optimization surfaces: learning rate, optimizer choice, batch size, dropout, channel width, weight decay, epoch count, CNN architecture, training loop logic, data preprocessing / transforms

## Caveats

- `eval.py` depends on a checkpoint at `artifacts/best_model.pt`, so training must succeed before evaluation.
- Both `train.py` and `eval.py` overwrite `reports/metrics.json`; BOA should score the post-eval file.
- Real scoring runs should not use `--use-fake-data`; fake data is only appropriate for smoke tests.
- Running with real MNIST may download data into `data/` on the first run.
- The CLI defaults to `device=auto`, but local comparisons are more stable when pinned to `--device cpu`.
- By default the repo trains on a 2048-example subset and evaluates on a 512-example subset, so the score is not from full-dataset MNIST unless those defaults are changed in editable code.