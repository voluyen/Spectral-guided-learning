#!/usr/bin/env bash
# Full reproduction pipeline. Run on the GPU server, one stage at a time.
#
#   ./scripts/run-pipeline.sh data      # phase 2: ~30 min, CPU + network
#   ./scripts/run-pipeline.sh capture   # phase 3: ~2-4 h GPU
#   ./scripts/run-pipeline.sh masks     # phase 4: seconds, CPU (p=0.95 fixed)
#   ./scripts/run-pipeline.sh train     # phase 5: ~10-20 h GPU per run, x2
#   ./scripts/run-pipeline.sh eval      # phase 6: ~4-6 h GPU per model
#
# Stages are resumable: each writes to disk and later stages read those files.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
mkdir -p logs
# Activate the project venv unless one is already active, so `python` has the ML stack.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

case "${1:-}" in
  data)
    python src/data_prep.py --config configs/data-config.yaml
    ;;

  capture)
    # --verify asserts the analytic gradient matches autograd before spending GPU hours
    python src/gradient_capture.py --config configs/capture-config.yaml --verify
    ;;

  masks)
    python src/build_masks.py --config configs/data-config.yaml --vanilla
    echo
    echo "p=0.95 is the paper's value (not tuned). Check the drop table above: if both the"
    echo "step-drop and token-drop ratios are ~0%, the spectral run is identical to vanilla."
    ;;

  smoke)
    # cheap end-to-end check on the real model before committing to full runs
    python src/train_sft.py --config configs/train-vanilla.yaml --smoke
    ;;

  train)
    python src/train_sft.py --config configs/train-vanilla.yaml  2>&1 | tee logs/train-vanilla.log
    python src/train_sft.py --config configs/train-spectral.yaml 2>&1 | tee logs/train-spectral.log
    ;;

  eval)
    python src/evaluate.py --model checkpoints/vanilla  --tag vanilla  --benchmarks all
    python src/evaluate.py --model checkpoints/spectral --tag spectral --benchmarks all
    python src/compare_results.py
    ;;

  *)
    sed -n '2,10p' "$0"   # print only the usage comment block
    exit 1
    ;;
esac
