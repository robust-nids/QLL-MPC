# ML training on QLL properties under MPC

The purpose of this repository is to collect all the tools, modifications, benchmarks and reports used in our evaluation of training neural networks on QLL properties under MPC.

## Overview

The aim of the experiment is to explore neuro-symbolic training with QLL properties in an MPC setting. The approach can be described as follows:

Given a property `ϕ(θ)` that concerns an ML model fθ with trainable parameters `θ`, the goal is to define a loss function `L(ϕ, θ)` that can be optimised to find a `θ∗` satisfying `ϕ`. In principle, `ϕ` can be an arbitrary propositional formula built from logical connectives such as `and` (∧), `or` (∨) and `negation` (¬). QLL interprets such connectives as operations over suitable subsets of the reals, for example:
```
    ⟦ϕ ∨ ψ⟧_p = ⟦ϕ⟧_p ∪^p ⟦ψ⟧_p = −1/p log(e^(−pa) + e^(-pb))
    where ⟦_⟧_p is the denotation function under softness p
```

## Goal

The goal is to extend the [EzPC](./EzPC) compiler with the types and logical connectives, along with their semantics, and to find the circuits that correspond to them.
The compiler supports multiple backends (e.g. [ABY](./ABY)), so it will be
interesting to see which ones are the most suitable for our experiment.

### Challenges

*Encoding the extended reals:*
QLL is interpreted over the extended reals `ℝ̄ = ℝ ∪ {−∞, +∞}`, and the connectives produce the infinities as ordinary truth values. IEEE 754 represents them natively (`±Inf`, plus `NaN` for the indeterminate forms), but MPC backends share values over ring or fields and work in fixed-point, so floating point has to be emulated in circuits at a high cost. Finding an encoding of `ℝ̄` that is faithful enough to keep the loss adequate yet cheap enough to evaluate under MPC is an open point of this work.

## Running the experiment

### Requirements

- Docker. The image is `linux/amd64`: it runs natively on x86_64 and under emulation on Apple Silicon.
- Python 3 with PyTorch, only for the cleartext reference.

### Setup

Run from the repository root:

```bash
# 1. Fetch the SCI build files and pinned dependencies
bash docker/fetch_upstream.sh

# 2. Build the image
docker build --platform linux/amd64 -t beacon-sci docker/

# 3. Build the EzPC compiler and SCI inside the container
docker run --rm --platform linux/amd64 -v "$PWD":/work beacon-sci bash docker/build.sh
```
Re-run step 3 after changing files in `EzPC/SCI/src`.

### Training

```
docker run --rm --platform linux/amd64 -v "$PWD":/work -e NETWORK=ToyNetwork -e LOSS=TorchQLL -e QLL_P=2.0 -e BATCH=128 beacon-sci bash docker/train.sh
```
This trains ToyNetwork with the `TorchQLL` loss and p = 2. Both parties (server and client) run in the container. The script prints the revealed loss for each iteration, followed by the time, communication (MiB) and rounds.

| Variable | Values | Default |
|---|---|---|
| `NETWORK` | `ToyNetwork`, `Logistic`, `FFNN` (MNIST) | Logistic |
| `LOSS` | `QLL` (hand-written C++ loss), `TorchQLL` (loss traced from PyTorch), `CE`, `MSE` | CE |
| `QLL_P` | QLL softness p | 2.0 |
| `BATCH` / `ITERS` / `LR` | batch size, iterations, learning rate | 128 / 5 / 0.01|

With `TorchQLL`, the property is the `loss()` function in [`beacon_frontend.py`](./EzPC/Beacon/beacon_frontend.py). It must match the chosen network's inputs and outputs.

### Cleartext reference for ToyNetwork

The same loss can be trained in plain PyTorch, on the same initial weights and data, to compare against the secure run. Run it from `EzPC/Beacon` after a training run, with the same loss, batch, iterations, lr and p.

```bash
python3 qll_reference.py ToyNetwork 128 5 0.01 2.0   # network, batch, iterations, lr, qll_p
```

## References

[[1]](https://arxiv.org/pdf/2605.13845) Flinkow, T., Komendantskaya, E., Capucci, M., & Monahan, R. (2026). Quantitative Linear Logic for Neuro-Symbolic Learning and Verification. ArXiv, abs/2605.13845.

[[2]](https://arxiv.org/pdf/2605.13348) Capucci, M., et al. (2026) Adequate Losses via Quantitative Linear Logic.

[[3]](https://www.informatik.tu-darmstadt.de/media/encrypto/encrypto_code/abydevguide.pdf) ABY Developer Guide

[[4]](https://ieeexplore.ieee.org/document/8806756) N. Chandran, D. Gupta, A. Rastogi, R. Sharma and S. Tripathi, "EzPC: Programmable and Efficient Secure Two-Party Computation for Machine Learning," 2019 (EuroS&P)

