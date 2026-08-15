#!/usr/bin/env bash
# Install the stack on a fresh GPU box: venv + torch (matching CUDA build) + requirements.txt.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

# Check `nvidia-smi`'s "CUDA Version" (top-right) first -- that's the driver's ceiling, not a
# free choice. cu118 default is for this box's A100s; H100/H200 boxes usually have a newer
# driver and should override, e.g. CUDA_TAG=cu124 ./scripts/setup.sh
CUDA_TAG="${CUDA_TAG:-cu118}"
VENV_DIR=.venv
INSTALL_FLASH_ATTN=false

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
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
  echo "WARN: HF_TOKEN not set (needed for gated datasets, e.g. GPQA in evaluate.py)"
fi
echo "setup done"
