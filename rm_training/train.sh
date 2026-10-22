#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RELEASE_ROOT}"
: "${MODEL:?Set MODEL to a local base-model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to your training preference JSONL}"
: "${VALIDATION_FILE:?Set VALIDATION_FILE to your validation preference JSONL}"
TRAINER="rm_training/train_pairwise.py"
EXTRA_ARGS=()
if [[ -n "${TEST_FILE:-}" ]]; then
    EXTRA_ARGS+=(--test_set_path "${TEST_FILE}")
fi
if [[ "${USE_DEEPSPEED:-0}" == 1 ]]; then
    TRAINER="rm_training/train_pairwise_zero3.py"
    EXTRA_ARGS+=(--deepspeed rm_training/configs/deepspeed_zero3.json)
fi
accelerate launch --num_processes "${NUM_PROCESSES:-1}" "${TRAINER}" \
    --model_name "${MODEL}" \
    --train_set_path "${TRAIN_FILE}" \
    --eval_set_path "${VALIDATION_FILE}" \
    --output_path "${OUTPUT_DIR:-outputs/reward_model}" \
    --report_to none \
    "${EXTRA_ARGS[@]}" "$@"
