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

### Running Experiments
> Note: To run the ezpc training, please refer to this [readme](./ezpc/README.md).

#### Requirements

- macOS on Apple silicon, with `clang++` (C++20).
- OpenSSL 3 from Homebrew: `brew install openssl@3` (set `OPENSSL=<path>` if it is not under `/opt/homebrew/opt/openssl@3`).
- Python 3 with PyTorch: `pip install torch`.

#### Training

```bash
cd hpmpc
scripts/acasXu.sh mac
```
Running with env variables:

```bash
OPTIMIZER=adam ITERS=1170 REPORT_EVERY=39 scripts/acasXu.sh mac
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPTIMIZER` | `sgd` | `sgd` or `adam` |
| `ITERS` | `100` | training steps |
| `BATCH_SIZE` | `256` | data points per step (plus as many property points) |
| `LEARNING_RATE` | `1e-4` | step size |
| `REPORT_EVERY` | `10` | print the loss every this many steps |
| `FRACTIONAL` | `20` | fixed-point fractional bits |
| `BETA1`, `BETA2`, `EPS`, `GRAD_SCALE` | `0.9`, `0.999`, `1e-4`, `5` | Adam only |
| `ACASXU_DIR` | `examples/acasXu_run` | run directory |

P1 writes the fine-tuned weights to `finetuned.inp` in the run directory; per-party logs are in `executables-mac/`.

Convert the fine-tuned weights back to ONNX, reusing the original network's graph:

```bash
python3 programs/torch/onnx_converter.py to_onnx examples/acasXu_run/finetuned.inp examples/acasXu_run/acasXu_2_9.onnx examples/acasXu_run/acasXu_2_9_1_mpc.onnx --meta lr=1e-4
```

## References

[[1]](https://arxiv.org/pdf/2605.13845) Flinkow, T., Komendantskaya, E., Capucci, M., & Monahan, R. (2026). Quantitative Linear Logic for Neuro-Symbolic Learning and Verification. ArXiv, abs/2605.13845.

[[2]](https://arxiv.org/pdf/2605.13348) Capucci, M., et al. (2026) Adequate Losses via Quantitative Linear Logic.

[[3]](https://www.informatik.tu-darmstadt.de/media/encrypto/encrypto_code/abydevguide.pdf) ABY Developer Guide

[[4]](https://ieeexplore.ieee.org/document/8806756) N. Chandran, D. Gupta, A. Rastogi, R. Sharma and S. Tripathi, "EzPC: Programmable and Efficient Secure Two-Party Computation for Machine Learning," 2019 (EuroS&P)
