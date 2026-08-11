#!/usr/bin/env bash
# Install the stack on a fresh GPU box: venv + CUDA 12 torch + requirements.txt.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

CUDA_TAG=cu121            # torch wheel tag; match `nvidia-smi` CUDA version
VENV_DIR=.venv
INSTALL_FLASH_ATTN=false

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -r requirements.txt
[[ "${INSTALL_FLASH_ATTN}" == true ]] && python -m pip install flash-attn --no-build-isolation

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
  echo "HF login OK"
else
  echo "WARN: HF_TOKEN not set (needed for push.sh)"
fi
echo "setup done"
