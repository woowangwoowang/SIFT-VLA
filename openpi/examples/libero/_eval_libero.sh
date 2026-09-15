#!/usr/bin/env bash
# Shared SIFT-VLA launcher for the PyTorch pi0.5 LIBERO checkpoint.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash examples/libero/_eval_libero.sh {spatial|object|goal|10}

Starts a policy server, evaluates the selected suite, then stops that server.
Defaults: 128 visual tokens TOTAL from 512 camera tokens (25% retained, Joint),
pruning before decoder index 3, 50 trials, seed 7.

Environment overrides:
  SERVER_PYTHON, CLIENT_PYTHON  Python executables for policy and LIBERO environments
  POLICY_DIR                  PyTorch checkpoint (default: checkpoints/pi05_libero_pytorch)
  POLICY_CONFIG               Training config (default: pi05_libero_sift_vla)
  SIFT_VLA_TOKEN_BUDGET        Total retained visual tokens (default: 128)
  SIFT_VLA_LAYER               Zero-based pruning layer (default: 3)
  SIFT_VLA_SELECTOR_BACKEND    auto, eager, or triton (default: auto)
  SIFT_VLA_ENABLE             Enable pruning (default: True)
  MODEL_USE_CACHE             Cache prefix K/V between denoising steps (default: True)
  OPENPI_ATTENTION_IMPLEMENTATION  auto, eager, sdpa, flash_attention_2 (default: auto)
  NUM_TRIALS_PER_TASK, SEED    Trial count and random seed (default: 50, 7)
  REPLAN_STEPS, RESIZE_SIZE   Actions per inference and image size (default: 5, 224)
  HOST, PORT                 Policy endpoint (default: 127.0.0.1, 8000)
  START_SERVER               False to use an existing server (default: True)
  SERVER_WAIT_SECONDS        Server readiness timeout (default: 900)
  LOG_DIR, RESULTS_OUT_DIR, VIDEO_OUT_PATH  Output directories

With START_SERVER=False, the running server determines pruning/cache settings.
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in
    spatial|object|goal|10) suite="libero_$1" ;;
    *) fail "Unknown suite '$1'." ;;
esac
shift
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown argument '$arg'; use --help for environment overrides." ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "$repo_root"

server_python="${SERVER_PYTHON:-${repo_root}/.venv/bin/python}"
client_python="${CLIENT_PYTHON:-${repo_root}/examples/libero/.venv/bin/python}"
[[ -n ${SERVER_PYTHON:-} || -x "$server_python" ]] || server_python=python
[[ -n ${CLIENT_PYTHON:-} || -x "$client_python" ]] || client_python=python

token_budget="${SIFT_VLA_TOKEN_BUDGET:-128}"
layer="${SIFT_VLA_LAYER:-3}"
backend="${SIFT_VLA_SELECTOR_BACKEND:-auto}"
attention="${OPENPI_ATTENTION_IMPLEMENTATION:-auto}"
num_trials="${NUM_TRIALS_PER_TASK:-50}"
seed="${SEED:-7}"
replan_steps="${REPLAN_STEPS:-5}"
resize_size="${RESIZE_SIZE:-224}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
wait_seconds="${SERVER_WAIT_SECONDS:-900}"

require_integer() {
    local name="$1" value="$2" minimum="$3" maximum="$4"
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 10 ]] || fail "$name must be an integer."
    (( value >= minimum && value <= maximum )) || fail "$name must be between $minimum and $maximum."
}
require_integer SIFT_VLA_TOKEN_BUDGET "$token_budget" 1 512
require_integer SIFT_VLA_LAYER "$layer" 0 17
require_integer NUM_TRIALS_PER_TASK "$num_trials" 1 50
require_integer SEED "$seed" 0 4294967295
require_integer PORT "$port" 1 65535
require_integer SERVER_WAIT_SECONDS "$wait_seconds" 1 86400
require_integer REPLAN_STEPS "$replan_steps" 1 50
require_integer RESIZE_SIZE "$resize_size" 1 4096
case "$backend" in auto|eager|triton) ;; *) fail 'SIFT_VLA_SELECTOR_BACKEND must be auto, eager, or triton.' ;; esac
case "$attention" in auto|eager|sdpa|flash_attention_2) ;; *) fail 'Invalid OPENPI_ATTENTION_IMPLEMENTATION.' ;; esac

bool_value() {
    local name="$1" value="$2"
    case "${value,,}" in
        true|1|yes|on) printf true ;;
        false|0|no|off) printf false ;;
        *) fail "$name must be True or False." ;;
    esac
}
for spec in START_SERVER:True SIFT_VLA_ENABLE:True MODEL_USE_CACHE:True; do
    name="${spec%%:*}"
    normalized="$(bool_value "$name" "${!name:-${spec#*:}}")"
    printf -v "$name" '%s' "$normalized"
done

if "$START_SERVER"; then
    case "$host" in
        127.0.0.1|localhost) ;;
        *) fail 'Use HOST=127.0.0.1 for an automatically started server; use START_SERVER=False for a remote server.' ;;
    esac
fi

server_cmd=("$server_python" scripts/serve_policy.py --env LIBERO --port "$port" --seed "$seed")
if "$MODEL_USE_CACHE"; then server_cmd+=(--model-use-cache); else server_cmd+=(--no-model-use-cache); fi
if "$SIFT_VLA_ENABLE"; then
    server_cmd+=(--sift-vla-enable --sift-vla-layer "$layer" --sift-vla-token-budget "$token_budget"
        --sift-vla-selector-backend "$backend")
else
    server_cmd+=(--no-sift-vla-enable)
fi
server_cmd+=(policy:checkpoint --policy.config "${POLICY_CONFIG:-pi05_libero_sift_vla}"
    --policy.dir "${POLICY_DIR:-${repo_root}/checkpoints/pi05_libero_pytorch}")

environment=(
    "PYTHONPATH=${repo_root}/src:${repo_root}/packages/openpi-client/src:${repo_root}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
    "OPENPI_ATTENTION_IMPLEMENTATION=$attention"
    "MUJOCO_GL=${MUJOCO_GL:-egl}"
    "MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/openpi_matplotlib}")
run_id="$(date +%Y-%m-%d_%H-%M-%S)_$$"
log_dir="${LOG_DIR:-data/libero/logs}"
results_dir="${RESULTS_OUT_DIR:-data/libero/results}"
video_dir="${VIDEO_OUT_PATH:-data/libero/videos}"
run_tag="sift_vla_L${layer}_K${token_budget}_joint_cache-${MODEL_USE_CACHE}"
"$SIFT_VLA_ENABLE" || run_tag="dense_cache-${MODEL_USE_CACHE}"
# Do not label results with settings that were not applied to an external server.
"$START_SERVER" || run_tag=external_server
server_log="${log_dir}/server_${run_tag}_${run_id}.log"

client_cmd=("$client_python" examples/libero/main.py --args.host "$host" --args.port "$port"
    --args.task-suite-name "$suite" --args.num-trials-per-task "$num_trials" --args.seed "$seed"
    --args.replan-steps "$replan_steps" --args.resize-size "$resize_size"
    --args.results-out-dir "$results_dir"
    --args.results-out-path "${results_dir}/${suite}_${run_tag}_${run_id}.txt"
    --args.video-out-path "${video_dir}/${suite}_${run_tag}_${run_id}")

server_pid=""
cleanup_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if "$START_SERVER"; then
    # Catch a busy port before launching a model, so another server cannot pass readiness.
    "$client_python" - "$port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    try:
        sock.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError as exc:
        sys.exit(f"Policy port {sys.argv[1]} is unavailable: {exc}")
PY
    mkdir -p -- "$log_dir"
    printf 'Starting policy server. Log: %s\n' "$server_log"
    env "${environment[@]}" "${server_cmd[@]}" >"$server_log" 2>&1 &
    server_pid="$!"
fi

if ! "$client_python" - "$host" "$port" "$wait_seconds" "$server_pid" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

host, port, timeout, pid = sys.argv[1:]
deadline = time.monotonic() + int(timeout)
while time.monotonic() < deadline:
    if pid:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            sys.exit("Policy server exited before becoming ready.")
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=1) as response:
            if response.status == 200:
                sys.exit(0)
    except (OSError, urllib.error.URLError):
        time.sleep(1)
sys.exit(f"Timed out waiting for policy server at {host}:{port}")
PY
then
    if [[ -n "$server_pid" ]]; then tail -n 40 "$server_log" >&2; fi
    exit 1
fi

printf 'Evaluating %s (%s).\n' "$suite" "$run_tag"
env "${environment[@]}" "${client_cmd[@]}"
