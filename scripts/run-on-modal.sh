#!/usr/bin/env bash
# Self-contained runner for the Spectral-guided Learning reproduction on a fresh GPU box
# (Modal container / any cloud GPU). Unlike run-pipeline.sh, this one also installs the
# stack and checks the GPU, so it works on a clean image with nothing preinstalled.
#
# Usage (run ONE stage at a time; check the gates between them):
#   bash scripts/run-on-modal.sh setup          # install deps + sanity-check GPU/HF (run once)
#   bash scripts/run-on-modal.sh data           # phase 2  (CPU, ~15 min)  -> data/train-2k-segmented.jsonl
#   bash scripts/run-on-modal.sh capture-test 20 # phase 3 test (GPU): SVD timing on N samples
#   bash scripts/run-on-modal.sh capture        # phase 3  (GPU, ~2-4 h)   -> data/spectral-strengths.parquet
#   bash scripts/run-on-modal.sh masks-sweep    # phase 4 gate: pick p (CPU, seconds)
#   bash scripts/run-on-modal.sh masks          # phase 4  (CPU) -> train-{vanilla,spectral}.jsonl
#   bash scripts/run-on-modal.sh train          # phase 5  (GPU, ~10-20 h per run, x2)
#   bash scripts/run-on-modal.sh eval           # phase 6  (GPU, ~4-6 h per model)
#   bash scripts/run-on-modal.sh report         # phase 7  (CPU)
#
# MODAL NOTES (read before launching):
#   * Persistence: mount a Modal Volume at the repo's data/ and checkpoints/ (or at the
#     whole repo). Container filesystems are wiped on exit — without a Volume every stage
#     loses its output. Both gradient_capture and data_prep resume from what is on disk,
#     so a Volume also makes an interrupted stage restartable.
#   * GPU: request the GPU in your Modal entrypoint (e.g. @app.function(gpu="A100-40GB"))
#     and call this script from inside it; a bare .sh cannot request a GPU by itself.
#   * Timeouts: set the Modal function timeout per stage. `train` needs the longest
#     (configure up to Modal's 24 h max, and run vanilla/spectral as two separate calls
#     if one run risks exceeding it).
#   * Secrets: pass HF_TOKEN as a Modal Secret; it is required (GPQA-Diamond is gated).

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs data checkpoints results

require_gpu() {
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: no CUDA GPU visible to torch — this stage needs a GPU." >&2
    echo "On Modal, request one in the entrypoint (e.g. gpu=\"A100-40GB\") before calling this." >&2
    exit 1
  fi
  python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
}

require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set (GPQA-Diamond is gated and the loader fails without it)." >&2
    echo "Pass it as a Modal Secret, or: export HF_TOKEN=hf_xxx" >&2
    exit 1
  fi
}

case "${1:-}" in
  setup)
    echo ">> installing the stack (torch/transformers/datasets/...); vllm is only needed for eval"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install "vllm>=0.8"      # eval generation; comment out if you only run data->train
    echo ">> versions:"
    python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"
    require_gpu || echo "(GPU not required for setup itself, but capture/train/eval will need one)"
    [[ -n "${HF_TOKEN:-}" ]] && python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])" && echo "HF login OK" || echo "WARN: HF_TOKEN not set yet (needed before eval)"
    echo ">> setup done."
    ;;

  data)
    # phase 2: CPU + network. Streams AceReason, tokenizes, segments into step spans.
    python src/data_prep.py --config configs/data-config.yaml 2>&1 | tee logs/data-prep.log
    ;;

  capture-test)
    # quick sanity + SVD-timing run on a few samples BEFORE the full corpus. Prints per-sample
    # gradient vs SVD wall time so you can size phase 3 (the paper never publishes SVD cost).
    # Optional 2nd arg = number of samples (default 20):  bash run-on-modal.sh capture-test 30
    require_gpu
    python src/gradient_capture.py --config configs/capture-config.yaml --verify --limit "${2:-20}" 2>&1 | tee logs/capture-test.log
    ;;

  capture)
    # phase 3: GPU. --verify asserts analytic gradient == autograd before spending GPU hours.
    require_gpu
    python src/gradient_capture.py --config configs/capture-config.yaml --verify 2>&1 | tee logs/capture.log
    echo
    echo "================= GATE k* ================="
    echo "Check the 'k*/T mean ratio' printed above. The paper's low-rank premise holds only"
    echo "if k* << min(T, d). If k*/T is NOT clearly small on the real 1.7B model, reconsider"
    echo "before spending GPU-days on training. See plan gate: phase 3."
    echo "==========================================="
    ;;

  masks-sweep)
    # phase 4 gate: the paper does not publish p (Eq. 8). Inspect drop ratios across candidates.
    python src/build_masks.py --config configs/data-config.yaml --sweep 0.7,0.8,0.9,0.95 2>&1 | tee logs/masks-sweep.log
    echo
    echo "Pick the smallest p with a clearly non-zero token-drop, then set energy_threshold_p"
    echo "in configs/data-config.yaml. A ~0% drop makes Spectral == Vanilla by construction."
    ;;

  masks)
    # phase 4: emit both datasets from the same code path (only the loss mask differs).
    python src/build_masks.py --config configs/data-config.yaml --vanilla 2>&1 | tee logs/masks.log
    ;;

  train)
    # phase 5: GPU x2. Vanilla and spectral share every hyperparameter; only the mask differs.
    require_gpu
    python src/train_sft.py --config configs/train-vanilla.yaml  2>&1 | tee logs/train-vanilla.log
    python src/train_sft.py --config configs/train-spectral.yaml 2>&1 | tee logs/train-spectral.log
    ;;

  smoke)
    # cheap end-to-end check on the real model before committing to the full train runs
    require_gpu
    python src/train_sft.py --config configs/train-vanilla.yaml --smoke 2>&1 | tee logs/smoke.log
    ;;

  eval)
    # phase 6: GPU. vLLM generation + scoring on the five benchmarks.
    require_gpu
    require_hf_token
    python src/evaluate.py --model checkpoints/vanilla  --tag vanilla  --benchmarks all 2>&1 | tee logs/eval-vanilla.log
    python src/evaluate.py --model checkpoints/spectral --tag spectral --benchmarks all 2>&1 | tee logs/eval-spectral.log
    ;;

  report)
    # phase 7: compare Spectral vs Vanilla (trend + token-drop).
    python src/compare_results.py 2>&1 | tee logs/report.log
    ;;

  *)
    sed -n '3,25p' "$0"   # print the usage/notes block
    exit 1
    ;;
esac
