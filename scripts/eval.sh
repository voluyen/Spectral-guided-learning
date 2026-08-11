#!/usr/bin/env bash
# Phase 6: eval one checkpoint on the benchmarks with vLLM (vLLM manages the GPU, no torchrun).
# Self-contained: every setting is inline below.
#   ./scripts/eval.sh                              # eval MODEL set below
#   ./scripts/eval.sh checkpoints/spectral spec    # optional: MODEL path + tag as args

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

# what to eval — checkpoint dir + a short tag for output/log filenames (args override)
MODEL="${BASE_PATH}/checkpoints/spectral"
TAG="spectral"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"

# benchmarks
BENCHMARKS="all"          # or e.g. "math500,aime24"
# generation (paper Section 4.1)
TEMPERATURE=0.6
TOP_P=0.95
N_SAMPLES=4               # responses per problem, accuracy averaged over them
MAX_TOKENS=32768          # generation length cap
MAX_MODEL_LEN=34816       # prompt + generation headroom
GPU_MEM_UTIL=0.9
SEED=42
# output dirs
RAW_DIR="${BASE_PATH}/results/raw"
RESULTS_DIR="${BASE_PATH}/results"

OPTS=""
OPTS+=" --model ${MODEL}"
OPTS+=" --tag ${TAG}"
OPTS+=" --benchmarks ${BENCHMARKS}"
# generation
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
OPTS+=" --seed ${SEED}"
# output
OPTS+=" --raw-dir ${RAW_DIR}"
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
