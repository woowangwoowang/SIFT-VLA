#!/usr/bin/env bash
# Shared SIFT-VLA launcher; normally use one of the four eval_libero_*.sh files.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash run_eval/_eval_libero.sh {spatial|object|goal|10} [evaluation arguments...]

The suite scripts select their own checkpoint and run 50 trials per task with
seed 7. SIFT-VLA retains K=64 of 256 visual tokens (25%) before zero-based
decoder layer 3, after layers 0, 1, 2 have run.
Edit token_budget at the top of each suite script to change its retained count.
Environment overrides:

  CKPT                     Hugging Face model ID or local checkpoint path
  PYTHON_BIN               Python executable (default: python in current environment)
  CONDA_ENV                Optional conda environment; uses conda run when set
  CONDA_EXE                Conda executable (default: conda)
  SIFT_VLA_TOKEN_BUDGET     Retained visual-token count (paper settings: 32, 64, 128)
  SIFT_VLA_LAYER            Zero-based decoder layer (default: 3)
  SIFT_VLA_SELECTOR_BACKEND auto, eager, or triton (default: auto)
  NUM_TRIALS, SEED          Trials per task and random seed (default: 50, 7)
  LOCAL_LOG_DIR            Evaluation log directory

Additional arguments are passed to run_libero_eval.py.
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
suite="$1"
shift
case "$suite" in
    spatial|object|goal|10) ;;
    *) fail "Unsupported suite '$suite'; choose spatial, object, goal, or 10." ;;
esac

extra_args=()
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        *) extra_args+=("$arg") ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

token_budget="${SIFT_VLA_TOKEN_BUDGET:-64}"
layer="${SIFT_VLA_LAYER:-3}"
num_trials="${NUM_TRIALS:-50}"
seed="${SEED:-7}"
selector_backend="${SIFT_VLA_SELECTOR_BACKEND:-auto}"
# Validate an explicit CLI override before Python loads the model, too.
for ((index=0; index<${#extra_args[@]}; index++)); do
    case "${extra_args[index]}" in
        --sift_vla_selector_backend)
            (( index + 1 < ${#extra_args[@]} )) || fail '--sift_vla_selector_backend requires a value.'
            selector_backend="${extra_args[index+1]}"
            ;;
        --sift_vla_selector_backend=*) selector_backend="${extra_args[index]#*=}" ;;
    esac
done

require_integer() {
    local name="$1" value="$2" minimum="$3" maximum="$4"
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 10 ]] || fail "$name must be an integer."
    (( value >= minimum && value <= maximum )) || fail "$name must be between $minimum and $maximum."
}
require_integer SIFT_VLA_TOKEN_BUDGET "$token_budget" 1 256
require_integer SIFT_VLA_LAYER "$layer" 0 31
require_integer NUM_TRIALS "$num_trials" 1 50
require_integer SEED "$seed" 0 4294967295
case "$selector_backend" in auto|eager|triton) ;; *) fail 'SIFT_VLA_SELECTOR_BACKEND must be auto, eager, or triton.' ;; esac

bool_value() {
    local name="$1" value="$2"
    case "${value,,}" in
        true|1|yes|on) printf True ;;
        false|0|no|off) printf False ;;
        *) fail "$name must be True or False." ;;
    esac
}

# Separate assignments preserve validation failures under set -e.
for spec in CENTER_CROP:True USE_WANDB:False MODEL_USE_CACHE:True; do
    name="${spec%%:*}"
    default="${spec#*:}"
    normalized="$(bool_value "$name" "${!name:-$default}")"
    printf -v "$name" '%s' "$normalized"
done

python_bin="${PYTHON_BIN:-python}"
command=("$python_bin" "${repo_root}/experiments/robot/libero/run_libero_eval.py"
    --model_family openvla
    --pretrained_checkpoint "${CKPT:-openvla/openvla-7b-finetuned-libero-${suite}}"
    --task_suite_name "libero_${suite}"
    --center_crop "${CENTER_CROP}"
    --num_trials_per_task "$num_trials"
    --seed "$seed"
    --local_log_dir "${LOCAL_LOG_DIR:-./experiments/logs}"
    --use_wandb "${USE_WANDB}"
    --sift_vla_enable True
    --sift_vla_layer "$layer"
    --sift_vla_token_budget "$token_budget"
    --sift_vla_selector_backend "$selector_backend"
    --model_use_cache "${MODEL_USE_CACHE}"
    "${extra_args[@]}")

# Opt in to conda explicitly; otherwise use the caller's active environment.
if [[ -n ${CONDA_ENV:-} ]]; then
    command=("${CONDA_EXE:-conda}" run --no-capture-output -n "$CONDA_ENV" "${command[@]}")
fi

environment=(
    "PYTHONPATH=${repo_root}/transformers/src:${repo_root}:${repo_root}/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
    "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}"
    "MUJOCO_GL=${MUJOCO_GL:-egl}"
    "WANDB_MODE=${WANDB_MODE:-offline}")

cd -- "$repo_root"
exec env "${environment[@]}" "${command[@]}"
