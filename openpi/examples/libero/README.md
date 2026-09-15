# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="policy:checkpoint --policy.config pi05_libero --policy.dir ./checkpoints/pi05_libero_pytorch" CLIENT_ARGS="--args.task-suite-name libero_10" docker compose -f examples/libero/compose.yml up --build

SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## SIFT-VLA evaluation

Run each suite from the repository root:

```bash
bash examples/libero/eval_libero_spatial.sh
bash examples/libero/eval_libero_object.sh
bash examples/libero/eval_libero_goal.sh
bash examples/libero/eval_libero_10.sh
```

Each script starts the policy server, waits for `/healthz`, runs the selected
suite, and stops its server, including after an evaluation failure. The default
PyTorch checkpoint directory is `checkpoints/pi05_libero_pytorch`; it must contain
`model.safetensors` and the checkpoint's normalization assets.

The defaults retain **128 visual tokens in total across both cameras** (128/512,
25%), prune before zero-based decoder index **3**, use KV caching, and run
**50 trials per task with seed 7**. Server noise sampling and the LIBERO client
both use that seed. Pruning always uses **Joint selection** over both 256-token
cameras, with no fixed per-camera quota or scope option.

The policy server and simulator can use separate Python environments. Local
`.venv/bin/python` and `examples/libero/.venv/bin/python` are used when present;
otherwise the active `python` is used. Set the executables explicitly when needed:

```bash
SERVER_PYTHON=/path/to/policy-env/bin/python \
CLIENT_PYTHON=/path/to/libero-env/bin/python \
POLICY_DIR=/path/to/pi05_libero_pytorch \
  bash examples/libero/eval_libero_spatial.sh

# Change the total visual-token budget:
SIFT_VLA_TOKEN_BUDGET=64 bash examples/libero/eval_libero_goal.sh
```

Each suite script has an editable `token_budget=128` near its top. Change that
value to set the suite's default; `SIFT_VLA_TOKEN_BUDGET`, when set, overrides it.

The server environment needs this repository's patched Transformers 4.53.2,
as described in the root [PyTorch setup](../../README.md#setup). Its PyTorch CUDA
build must support the GPU being used. See the checked environment and validation
coverage in [SIFT-VLA validation](../../docs/SIFT_VLA_VALIDATION.md).

```bash
# Reuse a policy server that is already running:
START_SERVER=False HOST=127.0.0.1 PORT=8000 \
  bash examples/libero/eval_libero_10.sh
```

When reusing a server, its existing token budget and cache settings apply. Output
filenames use `external_server` to avoid claiming that unapplied launcher settings
were used. For automatically started servers, `HOST` must be local. Concurrent
suite runs need different `PORT` values. Results and videos are saved under
`data/libero/results` and `data/libero/videos`, with separate run names.

Use `bash examples/libero/_eval_libero.sh --help` for all supported settings.
The implementation and manuscript differences are described in
[SIFT-VLA notes](../../docs/SIFT_VLA.md).

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
