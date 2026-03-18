# CIFAR-10 ResNet BOA Demo

This example is a more realistic vision training repository for BOA. It expands the search space beyond obvious scalar tweaks and gives BOA room to improve augmentation, optimization, and model capacity.

## Hardware Expectation

- Runs on CPU for smoke-scale experiments.
- Better suited to GPU for repeated BOA trials.
- Typical default run: several minutes on GPU, substantially slower on CPU.

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

Smoke path:

```bash
python train.py --use-fake-data --train-size 128 --val-size 64 --epochs 1
python eval.py --use-fake-data --val-size 64
```

## BOA Flow

```bash
boa init .
boa run .
```

## Metric Contract

- Output file: `reports/metrics.json`
- Primary metric: `accuracy`
- Additional metrics: `loss`, `runtime_seconds`

## Likely Optimization Surfaces

- augmentation strength
- optimizer and scheduler
- weight decay
- label smoothing
- depth and width
- normalization and dropout

