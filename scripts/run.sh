#!/usr/bin/env bash
# Full pipeline: setup -> train (vanilla + spectral) -> eval both -> compare -> push.
# Run steps individually instead if you want to stop and inspect between them.
#   ./scripts/run.sh
set -euo pipefail
HERE="$(dirname "$0")"
BASE_PATH="$(cd "${HERE}/.." && pwd)"
export PYTHONPATH="${BASE_PATH}/src"

bash "${HERE}/setup.sh"
bash "${HERE}/train-vanilla.sh"
bash "${HERE}/train-spectral.sh"
bash "${HERE}/eval.sh" "${BASE_PATH}/checkpoints/vanilla"  vanilla
bash "${HERE}/eval.sh" "${BASE_PATH}/checkpoints/spectral" spectral
python "${BASE_PATH}/src/compare_results.py"
bash "${HERE}/push.sh"
