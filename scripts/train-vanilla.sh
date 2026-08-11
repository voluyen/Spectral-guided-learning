#!/usr/bin/env bash
# Phase 5: masked SFT — VANILLA baseline (supervises every response token).
# Self-contained: every setting is inline below. Run: ./scripts/train-vanilla.sh

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

# data + output (this is what makes it the vanilla run)
DATA_PATH="${BASE_PATH}/data/train-vanilla.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/vanilla"
# model
MODEL_NAME="Qwen/Qwen3-1.7B-Base"
# hp (paper Table 3)
EPOCHS=6
LR=5.0e-5
MIN_LR=1.0e-5
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=32
ATTN=sdpa                 # sdpa | flash_attention_2
LOG_INTERVAL=5
SEED=42
# checkpointing: SAVE_STRATEGY = epoch | steps | no. With "steps", SAVE_STEPS sets the interval.
# SAVE_TOTAL_LIMIT caps how many checkpoints are kept (each ~17GB with optimizer state).
SAVE_STRATEGY=epoch
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=6

OPTS=""
# model + data
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
# hp
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --min-learning-rate ${MIN_LR}"
OPTS+=" --warmup-ratio ${WARMUP_RATIO}"
OPTS+=" --per-device-batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
# runtime
OPTS+=" --attn-implementation ${ATTN}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
# checkpointing
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-steps ${SAVE_STEPS}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
# seed
OPTS+=" --seed ${SEED}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/train-vanilla.log"
