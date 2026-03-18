# MNIST CNN BOA Demo

This example is a compact image-classification repository designed to be initialized with `boa init` and iterated on with `boa run`.

## Why This Example

- Fast CPU-oriented training loop.
- Clear deep learning optimization knobs.
- Small enough that BOA edits are easy to reason about.

## Hardware Expectation

- Intended for CPU.
- Typical default run: a few minutes on a laptop CPU, faster on GPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Training:

```bash
python train.py
```

Evaluation:

```bash
python eval.py
```

Smoke path without downloading MNIST:

```bash
python train.py --use-fake-data --train-size 256 --val-size 128 --epochs 1
python eval.py --use-fake-data --val-size 128
```

## BOA Flow

```bash
boa init .
boa run .
```

BOA should detect:

- `train.py`
- `eval.py`
- primary metric in `reports/metrics.json`

## Metric Contract

- Output file: `reports/metrics.json`
- Primary metric: `accuracy`
- Additional metrics: `loss`, `runtime_seconds`

## Likely Optimization Surfaces

- learning rate
- batch size
- optimizer choice
- dropout
- channel width
- epoch count

