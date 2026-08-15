#!/usr/bin/env bash
# Install the eval/vLLM stack into a separate venv (.venv-eval), isolated from the training
# venv (.venv). vLLM pins its own torch/transformers versions that don't match what training
# needs — installing it into .venv would silently break train_sft.py.
# Requires `uv` (this box's Python 3.12 install is uv-managed; plain `python3.12 -m venv` fails
# with "externally-managed-environment").
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

# Must match setup.sh's CUDA_TAG -- capped by `nvidia-smi`'s driver, not a free choice. cu124
# default is for the H200 target box; override for other hardware, e.g.
# CUDA_TAG=cu118 ./scripts/setup-eval.sh (torch==2.6.0 below has cu118/cu124/cu126 wheels --
# pick whichever matches the driver).
CUDA_TAG="${CUDA_TAG:-cu124}"
VENV_DIR=.venv-eval
VLLM_VERSION=0.8.3       # newest vllm whose pinned torch (2.6.0) still has wheels for the CUDA
                         # tags above, and whose transformers floor isn't broken by transformers v5

command -v uv >/dev/null || { echo "ERROR: uv not found (needed for a Python 3.12 venv here)" >&2; exit 1; }
[[ -d "${VENV_DIR}" ]] || uv venv --python 3.12 "${VENV_DIR}"
VENV_PY="${BASE_PATH}/${VENV_DIR}/bin/python"

# torch/torchvision/torchaudio version strings carry no CUDA marker, so installing vllm alone
# can silently pull a different CUDA build for torchvision/torchaudio than the pinned torch —
# force all three from the same CUDA_TAG index together, --reinstall so a mismatched prior
# install (same version string, wrong CUDA build) actually gets replaced instead of "already
# satisfied".
uv pip install --python "${VENV_PY}" --reinstall \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install --python "${VENV_PY}" "vllm==${VLLM_VERSION}"
# vllm 0.8.3's LoRA cache (vllm/utils.py LRUCache.touch) calls self._LRUCache__update(key),
# relying on a private method name-mangled from cachetools.LRUCache's own internals. cachetools
# 6.0 removed it (renamed/inlined) -- pulling in latest cachetools breaks LoRA eval at request
# time with "AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'"
# (confirmed empirically). Pin to a version that still has it.
uv pip install --python "${VENV_PY}" "cachetools==5.5.2"
# vllm's own pin (transformers>=4.51.0, no upper bound) is too permissive: transformers v5
# dropped APIs vllm 0.8.3 imports at startup (e.g. ProcessorMixin), breaking `from vllm import LLM`.
uv pip install --python "${VENV_PY}" "transformers>=4.51.0,<5"
# evaluate.py / benchmarks.py / answer_scoring.py deps — not part of vllm's own dependency tree
uv pip install --python "${VENV_PY}" datasets math-verify
# datasets/pandas pull the latest numpy (2.4+), but vllm's numba dep (speculative decoding,
# imported at engine startup even when unused) hard-requires numpy<2.2 — repin after the fact.
uv pip install --python "${VENV_PY}" "numpy<2.2"

"${VENV_PY}" -c "
import torch
assert torch.cuda.is_available(), 'no CUDA GPU visible to torch'
from vllm import LLM
print(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | vllm import OK')
"
echo "eval setup done"
