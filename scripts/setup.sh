#!/usr/bin/env bash
# Install the full stack (training + eval/vLLM) into a single venv on a fresh GPU box.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

# cu130 = earliest PyTorch stable build with Blackwell/B200 kernels; also covers Ampere/Hopper.
CUDA_TAG="${CUDA_TAG:-cu130}"
VENV_DIR=.venv
INSTALL_FLASH_ATTN=false
VLLM_VERSION=0.27.1  # torch/torchvision/torchaudio below must match this vLLM's own pins

command -v uv >/dev/null || { echo "ERROR: uv not found (needed for a Python 3.12 venv here)" >&2; exit 1; }
[[ -d "${VENV_DIR}" ]] || uv venv --python 3.12 "${VENV_DIR}"
VENV_PY="${BASE_PATH}/${VENV_DIR}/bin/python"

# --reinstall + shared index-url: keeps torch/torchvision/torchaudio on the same CUDA build.
uv pip install --python "${VENV_PY}" --reinstall \
  "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install --python "${VENV_PY}" "vllm==${VLLM_VERSION}"
uv pip install --python "${VENV_PY}" -r requirements.txt
uv pip install --python "${VENV_PY}" "numpy<2.5"  # repin: datasets/pandas pull >=2.5, vllm needs <2.5
[[ "${INSTALL_FLASH_ATTN}" == true ]] && uv pip install --python "${VENV_PY}" flash-attn --no-build-isolation

"${VENV_PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
from vllm import LLM
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda} | vllm import OK")
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  "${VENV_PY}" -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
  echo "HF login OK"
else
  echo "WARN: HF_TOKEN not set (needed for gated datasets, e.g. GPQA in evaluate.py)"
fi
echo "setup done"
