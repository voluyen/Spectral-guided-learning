#!/usr/bin/env bash
# Full pipeline (data -> capture -> masks -> train -> eval) for all 4 tracks, unattended.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# qwen3-1.7b: vanilla + spectral (no external baseline, needs a local vanilla run) + prucot
"${BASE_PATH}/scripts/qwen3/data/data_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/data/capture_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/data/masks_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/prucot/weight_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/prucot/prune_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/prucot/prucot_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh" \
  "${BASE_PATH}/checkpoints/vanilla-qwen3-1.7b" vanilla-qwen3-1.7b
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh" \
  "${BASE_PATH}/checkpoints/prucot-qwen3-1.7b" prucot-qwen3-1.7b

# qwen3-4b-instruct: vanilla + spectral (no external baseline, needs a local vanilla run) + prucot
"${BASE_PATH}/scripts/qwen3/data/data_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/data/capture_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/data/masks_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/prucot/weight_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/prucot/prune_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/prucot/prucot_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh"
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh" \
  "${BASE_PATH}/checkpoints/vanilla-qwen3-4b-instruct" vanilla-qwen3-4b-instruct
"${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh" \
  "${BASE_PATH}/checkpoints/prucot-qwen3-4b-instruct" prucot-qwen3-4b-instruct

# qwen3-8b and qwen25-7b: out of scope for now (only 1.7b + 4b-instruct are being run) --
# uncomment when ready to resume these tracks.
# "${BASE_PATH}/scripts/qwen3/data/data_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/data/capture_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/data/masks_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/prucot/weight_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/prucot/prune_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/prucot/prucot_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-8b.sh"
# "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-8b.sh" \
#   "${BASE_PATH}/checkpoints/prucot-qwen3-8b" prucot-qwen3-8b
#
# "${BASE_PATH}/scripts/qwen25/data/data_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/data/capture_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/data/masks_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/prucot/weight_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/prucot/prune_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/spectral/spectral_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/prucot/prucot_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/eval/eval_qwen25-7b.sh"
# "${BASE_PATH}/scripts/qwen25/eval/eval_qwen25-7b.sh" \
#   "${BASE_PATH}/checkpoints/prucot-qwen25-7b" prucot-qwen25-7b

python "${BASE_PATH}/src/compare_results.py"
