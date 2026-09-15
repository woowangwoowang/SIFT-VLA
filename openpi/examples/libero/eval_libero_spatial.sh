#!/usr/bin/env bash
set -euo pipefail

# Total retained visual tokens across both cameras.
token_budget=128

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIFT_VLA_TOKEN_BUDGET="${SIFT_VLA_TOKEN_BUDGET:-$token_budget}" \
    exec bash "${script_dir}/_eval_libero.sh" spatial "$@"
