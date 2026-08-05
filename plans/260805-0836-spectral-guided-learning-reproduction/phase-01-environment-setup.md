---
phase: 1
title: "Environment Setup"
status: in-progress
priority: P1
effort: "2-4h"
dependencies: []
---

# Phase 1: Environment Setup

## Overview
Prepare GPU server env + code sync workflow. Local machine (no GPU) is where code is written; all compute runs on the SSH server.

## Requirements
- Functional: Python ≥3.10 env on server with full ML stack; repeatable sync of `src/` + `configs/` to server
- Non-functional: pinned versions (reproducibility); verify GPU is 40-48GB free

## Architecture
Local (edit) → git push / rsync → Server (run). Data + checkpoints live on server only; small artifacts (stats, selection masks summary, eval results JSON) synced back.

## Related Code Files
- Create: `requirements.txt`, `scripts/sync-to-server.sh`, `configs/paths.yaml` (server-side data/ckpt roots)
- Create: `.gitignore` (exclude data/, checkpoints/, *.pt)

## Implementation Steps
1. Ask user for server SSH alias + preferred sync (git remote vs rsync); write `scripts/sync-to-server.sh` accordingly. **[USER INPUT NEEDED]**
2. `git init` repo locally (project is not a git repo yet).
3. `requirements.txt`: torch (cu12x), transformers ≥4.51 (Qwen3 support), datasets, accelerate, flash-attn, vllm, math-verify, pyyaml, numpy. Pin versions after first successful install.
4. On server: create venv/conda env, install; `flash-attn` needs matching torch/cuda — fallback to SDPA attention if build fails (config flag, not blocker).
5. Verify: `nvidia-smi` (≥40GB), `python -c "import torch; print(torch.cuda.get_device_name())"`, load Qwen3-1.7B-Base tokenizer from HF hub (checks connectivity/HF cache dir).
6. Set `HF_HOME` on server to a disk with ≥100GB free (model + datasets + checkpoints of 2 runs ≈ 60-80GB).

## Success Criteria
- [ ] `sync-to-server.sh` round-trips a test file
- [ ] torch sees GPU with ≥40GB
- [ ] Qwen3-1.7B-Base loads in bf16 on server (smoke test forward pass)
- [ ] flash-attn works OR SDPA fallback flag documented in config

## Risk Assessment
- flash-attn compile failures common → SDPA fallback acceptable (slower, still fits 16k seq with grad ckpt)
- Server disk/quota unknown → check before downloading datasets
