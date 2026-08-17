#!/usr/bin/env bash
# Phase 3: gradient capture (--verify) for the Qwen3-4B-Instruct-2507 track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] || ./scripts/setup.sh
  source .venv/bin/activate
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH="data/qwen3-4b-instruct/train-s1k-segmented.jsonl"
OUTPUT_DIR="data/qwen3-4b-instruct/spectral"
STRENGTHS_PATH="data/qwen3-4b-instruct/spectral-strengths.parquet"
ENERGY_CUTOFF=0.95
CHUNK_SIZE=1024

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --strengths-path ${STRENGTHS_PATH}"
OPTS+=" --energy-cutoff ${ENERGY_CUTOFF}"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
OPTS+=" --verify"

CMD="python ${BASE_PATH}/src/gradient_capture.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-4b-instruct-capture.log

echo ">>> STOP AND READ: check 'k*/T mean ratio' and 'strength spread' above (docs/server-runbook.md §5)."
