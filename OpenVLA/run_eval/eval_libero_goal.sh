#!/usr/bin/env bash
# Evaluate SIFT-VLA on LIBERO-Goal. See --help for environment overrides.
set -euo pipefail

token_budget=64  # Retained visual tokens: 64/256 = 25%.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIFT_VLA_TOKEN_BUDGET="${SIFT_VLA_TOKEN_BUDGET:-$token_budget}" exec bash "${script_dir}/_eval_libero.sh" goal "$@"
