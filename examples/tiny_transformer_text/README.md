# Tiny Transformer Text BOA Demo

This example is the hardest BOA target in the set. It is a compact transformer-based text classification repository with a richer architecture and training search space than the vision demos.

## Hardware Expectation

- Smoke-scale runs work on CPU.
- Recommended for GPU when running many BOA trials.
- Typical default run: acceptable on GPU, noticeably slower on CPU.

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

- sequence length
- embedding dimension
- number of layers and heads
- dropout
- learning rate and warmup
- batch size
- vocabulary size and tokenization thresholds

