#!/usr/bin/env bash
# Push each trained checkpoint to HuggingFace Hub (skips intermediate checkpoint-*/ epoch saves).
# Needs HF_TOKEN exported and HF_NAMESPACE set below.
# Self-contained: every setting is inline below. `./scripts/push.sh` (or `./scripts/push.sh spectral`).
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"
export PYTHONPATH="${BASE_PATH}/src"

HF_NAMESPACE="${HF_NAMESPACE:-}"          # your HF user/org, e.g. "voluyen" — required
HF_REPO_PREFIX="spectral-guided-learning"  # -> HF_NAMESPACE/HF_REPO_PREFIX-<variant>
HF_PRIVATE=true

VARIANTS=(vanilla spectral)
[[ $# -gt 0 ]] && VARIANTS=("$@")
declare -A OUTPUT=(
  [vanilla]="${BASE_PATH}/checkpoints/vanilla"
  [spectral]="${BASE_PATH}/checkpoints/spectral"
)

[[ -z "${HF_NAMESPACE}" ]] && { echo "ERROR: set HF_NAMESPACE in scripts/push.sh" >&2; exit 1; }
[[ -z "${HF_TOKEN:-}" ]]   && { echo "ERROR: export HF_TOKEN=hf_..." >&2; exit 1; }

for variant in "${VARIANTS[@]}"; do
  MODEL_DIR="${OUTPUT[$variant]}"
  REPO_ID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${variant}"
  [[ -d "${MODEL_DIR}" ]] || { echo "ERROR: ${MODEL_DIR} missing — run train.sh first" >&2; exit 1; }

  echo "push ${MODEL_DIR} -> https://huggingface.co/${REPO_ID}"
  REPO_ID="${REPO_ID}" MODEL_DIR="${MODEL_DIR}" HF_PRIVATE="${HF_PRIVATE}" VARIANT="${variant}" \
  python - <<'PY'
import glob, os
from huggingface_hub import HfApi

api = HfApi()
repo = os.environ["REPO_ID"]
api.create_repo(repo, private=os.environ["HF_PRIVATE"] == "true", exist_ok=True)
# checkpoint-*/ are per-epoch saves with optimizer state (~17GB each), kept only for local resume.
api.upload_folder(repo_id=repo, folder_path=os.environ["MODEL_DIR"], ignore_patterns=["checkpoint-*"])
logs = sorted(glob.glob(f"logs/train-{os.environ['VARIANT']}.log"))
if logs:
    api.upload_file(repo_id=repo, path_or_fileobj=logs[-1], path_in_repo=f"logs/{os.path.basename(logs[-1])}")
print(f"pushed: https://huggingface.co/{repo}")
PY
done
