# BOA Guidance

Optimize for validation `accuracy` written to `reports/metrics.json`.

Use the standard flow:
- `python train.py --device cpu`
- `python eval.py --device cpu`

Work inside `src/mnist_demo/` unless a change outside that package is clearly necessary. Keep the CLI contract, checkpoint path behavior, and metric file schema stable.

Do not switch scoring runs to `--use-fake-data`; fake data is only for smoke tests. Preserve the metric keys `accuracy`, `loss`, and `runtime_seconds`, and continue writing them to `reports/metrics.json`.

Broader rewrites and structural refactors are acceptable when they materially improve accuracy or training efficiency. Prioritize improvements to model architecture, optimizer settings, regularization, preprocessing, and training logic. Coherence and testability must be maintained across any larger diffs. Keep runs practical for a laptop CPU.

## BOA Setup Summary

- Train command: `python train.py --device cpu`
- Eval command: `python eval.py --device cpu`
- Primary metric: `accuracy` (maximize)
- Editable paths: `src/mnist_demo/cli.py`, `src/mnist_demo/config.py`, `src/mnist_demo/data.py`, `src/mnist_demo/model.py`, `src/mnist_demo/runtime.py`, `src/mnist_demo/training.py`
- Protected paths: `train.py`, `eval.py`, `requirements.txt`, `reports/metrics.json`, `README.md`, `tests/test_smoke.py`, `.git`, `.boa/protected`
- Likely optimization surfaces: learning rate, optimizer choice, batch size, dropout, channel width, weight decay, epoch count, train_size, val_size, CNN architecture, training loop logic, data preprocessing / transforms

## Caveats

- `eval.py` requires `artifacts/best_model.pt`, so training must complete successfully before evaluation.
- `train.py` and `eval.py` both overwrite `reports/metrics.json`; BOA should read the post-eval file.
- Use `--use-fake-data` only for smoke tests, not scoring runs.
- The default dataset sizes are `train_size=2048` and `val_size=512`, so reported accuracy is on that configured subset unless defaults change.
- Real MNIST runs may download data into `data/` on first run.
- The default device is `auto`; pinning `--device cpu` makes local runs more comparable.

## Agent Edit Strategy

- Aggressiveness: `aggressive`
- Guidance: Allow broader rewrites when they improve the objective. Structural refactors are acceptable if the resulting diff remains coherent and testable.

### Mode Rules

- Pursue the highest-impact changes to model architecture, hyperparameters, and training logic.
- Structural refactors (e.g., reorganizing the model class, rewriting the training loop) are acceptable when they enable better optimization.
- Ensure all changes remain testable: smoke tests must continue to pass with `--use-fake-data`.
- Keep diffs coherent — large changes should be logically self-contained and easy to review.
- Do not rewrite the CLI contract, checkpoint paths, or metric file schema.

## Agent Edit Strategy

- Aggressiveness: `aggressive`
- Guidance: Allow broader rewrites when they improve the objective. Structural refactors are acceptable if the resulting diff remains coherent and testable.

### Mode Rules

- Broader rewrites are allowed when they clearly support the objective.
- You may restructure functions or modules if the change stays within BOA guardrails.
- Accept larger diffs when they simplify the design or unlock measurable gains.

## BOA Setup Summary

- Train command: `python train.py --device cpu`
- Eval command: `python eval.py --device cpu`
- Primary metric: `accuracy` (maximize)
- Editable paths: src/mnist_demo/cli.py, src/mnist_demo/config.py, src/mnist_demo/data.py, src/mnist_demo/model.py, src/mnist_demo/runtime.py, src/mnist_demo/training.py
- Protected paths: reports/metrics.json, tests/test_smoke.py, .git, train.py, eval.py, .boa, README.md, .boa/protected, requirements.txt
- Likely optimization surfaces: learning rate, optimizer choice, batch size, dropout, channel width, weight decay, epoch count, train_size, val_size, CNN architecture, training loop logic, data preprocessing / transforms

## Caveats

- eval.py requires artifacts/best_model.pt, so training must complete successfully before evaluation.
- train.py and eval.py both overwrite reports/metrics.json; BOA should read the post-eval file.
- Use --use-fake-data only for smoke tests, not scoring runs.
- The default dataset sizes are train_size=2048 and val_size=512, so reported accuracy is on that configured subset unless defaults change.
- Real MNIST runs may download data into data/ on first run.
- The default device is auto; pinning --device cpu makes local runs more comparable.