#!/usr/bin/env bash
# Full pipeline (data -> capture -> masks -> train -> eval) for all 4 tracks, unattended.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# qwen3-1.7b: vanilla + spectral (no external baseline, needs a local vanilla run)
"${BASE_PATH}/scripts/qwen3/data/data_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/data/capture_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/data/masks_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh" \
  "${BASE_PATH}/checkpoints/vanilla-qwen3-1.7b" vanilla-qwen3-1.7b

# qwen3-4b-instruct: vanilla + spectral (no external baseline, needs a local vanilla run)
"${BASE_PATH}/scripts/qwen3/data/data_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/data/capture_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/data/masks_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh" \
  "${BASE_PATH}/checkpoints/vanilla-qwen3-4b-instruct" vanilla-qwen3-4b-instruct

# qwen3-8b: spectral only (compared against P-ALIGN's published vanilla-SFT numbers)
"${BASE_PATH}/scripts/qwen3/data/data_qwen3-8b.sh"
"${BASE_PATH}/scripts/qwen3/data/capture_qwen3-8b.sh"
"${BASE_PATH}/scripts/qwen3/data/masks_qwen3-8b.sh"
"${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-8b.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-8b.sh"

# qwen25-7b: spectral only (compared against P-ALIGN's published vanilla-SFT numbers)
"${BASE_PATH}/scripts/qwen25/data/data_qwen25-7b.sh"
"${BASE_PATH}/scripts/qwen25/data/capture_qwen25-7b.sh"
"${BASE_PATH}/scripts/qwen25/data/masks_qwen25-7b.sh"
"${BASE_PATH}/scripts/qwen25/spectral/spectral_qwen25-7b.sh"
"${BASE_PATH}/scripts/qwen25/eval/eval_qwen25-7b.sh"

python "${BASE_PATH}/src/compare_results.py"
