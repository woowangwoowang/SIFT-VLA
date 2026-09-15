# SIFT-VLA

**Task-Aware Visual Token Pruning without Attention Maps**

## Overview

SIFT-VLA is a training-free visual-token pruning method for vision-language-action
models. It combines instruction relevance and subset diversity using hidden
representations to reduce the visual tokens processed by subsequent decoder layers.

- **Training-free:** Applies to pretrained VLA models without additional training.
- **Attention-map-free:** Selects tokens without extracting attention maps,
  preserving compatibility with FlashAttention.
- **Relevance and diversity:** Iteratively balances instruction relevance with
  diversity among the retained tokens.

This repository provides implementations for **OpenVLA** and **$\pi_{0.5}$
** on
LIBERO.

## Core Implementation

The token selection algorithm is implemented in:

- **OpenVLA:** [`sift_vla_prune`](OpenVLA/rd_token/methods/sift_vla.py)
- **$\pi_{0.5}$
:** [`sift_vla_prune_prefix`](openpi/src/openpi/models_pytorch/sift_vla.py)

Both files contain relevance scoring, diversity scoring, and iterative token
selection, with PyTorch and Triton backends.

## Installation

Use Linux and a CUDA GPU. Set up OpenVLA and OpenPI in **separate environments**
using the modified source included in this repository.

| Setup and usage | Guide |
| --- | --- |
| OpenVLA | [OpenVLA/README.md](OpenVLA/README.md#installation) |
| $\pi_{0.5}$ | [openpi/README.md](openpi/README.md#SIFT-VLA-setup) |

The OpenVLA guide covers its Python 3.10 environment and LIBERO initialization.
The OpenPI guide covers the Python 3.11 policy environment, a separate LIBERO
simulator environment, and PyTorch checkpoint preparation.

## Quick Start

After completing installation and checkpoint preparation, run the following
commands from the **repository root**. Choose one LIBERO suite to evaluate;
`libero_10` is the Long-Horizon suite.

### OpenVLA

Activate the OpenVLA environment, then run:

```bash
source OpenVLA/.venv/bin/activate
bash OpenVLA/run_eval/eval_libero_spatial.sh
bash OpenVLA/run_eval/eval_libero_object.sh
bash OpenVLA/run_eval/eval_libero_goal.sh
bash OpenVLA/run_eval/eval_libero_10.sh
```

Each script selects its matching `openvla/openvla-7b-finetuned-libero-{suite}`
checkpoint. Set `CKPT=/path/to/checkpoint` to use a local checkpoint.

### $\pi_{0.5}$

Select the policy and simulator environments and the prepared checkpoint:

```bash
export SERVER_PYTHON="$PWD/openpi/.venv/bin/python"
export CLIENT_PYTHON="$PWD/openpi/examples/libero/.venv/bin/python"
export POLICY_DIR="$PWD/openpi/checkpoints/pi05_libero_pytorch"

bash openpi/examples/libero/eval_libero_spatial.sh
bash openpi/examples/libero/eval_libero_object.sh
bash openpi/examples/libero/eval_libero_goal.sh
bash openpi/examples/libero/eval_libero_10.sh
```

Each script starts the policy server, evaluates the selected suite, and stops
the server when finished.

### Customize the token budget

Set `SIFT_VLA_TOKEN_BUDGET` to choose how many visual tokens to retain:

```bash
SIFT_VLA_TOKEN_BUDGET=128 bash OpenVLA/run_eval/eval_libero_goal.sh
SIFT_VLA_TOKEN_BUDGET=64 bash openpi/examples/libero/eval_libero_goal.sh
```

Environment variables use the `SIFT_VLA_` prefix, and Python settings use
`sift_vla_`. CLI options use `--sift_vla_*` for OpenVLA and `--sift-vla-*`
for OpenPI. The OpenPI evaluation config is `pi05_libero_sift_vla`.
