---
phase: 6
title: "Evaluation on 5 Benchmarks"
status: in-progress
priority: P1
effort: "1d GPU"
dependencies: [5]
---

# Phase 6: Evaluation on 5 Benchmarks

## Overview
Evaluate both checkpoints with vLLM on AIME24, AIME25, MATH500, OlympiadBench (math), GPQA-Diamond. Paper setup: temp 0.6, top-p 0.95, n=4 responses/problem, max 32,768 gen tokens; report avg accuracy.

## Requirements
- Functional: `evaluate.py --model {ckpt} --bench {name}` → per-problem JSONL + summary JSON; final comparison table
- Non-functional: resumable per benchmark; identical prompts/parsing across both models

## Architecture
- vLLM offline batch inference (`LLM.generate`, `SamplingParams(n=4, temperature=0.6, top_p=0.95, max_tokens=32768)`).
- Benchmarks (HF sources, pin at implementation): AIME24 (`math-ai/aime24` or `Maxwell-Jia/AIME_2024`, 30), AIME25 (`math-ai/aime25`, 30), MATH500 (`HuggingFaceH4/MATH-500`, 500), OlympiadBench math-text-EN subset (`Hothan/OlympiadBench` OE_TO_maths_en_COMP), GPQA-Diamond (`Idavidrein/gpqa`, 198 — gated, needs HF token).
- Scoring: math → `math-verify` (extract \boxed{} / final answer, symbolic equivalence); GPQA → multiple-choice letter extraction (fixed template, shuffle-free). accuracy = mean over n=4 × problems.
- Prompting matches training format (Base model, plain problem prompt as in phase 2) + instruction to box final answer.

## Related Code Files
- Create: `src/evaluate.py`, `src/answer-scoring.py`, `configs/eval-config.yaml`
- Create: `tests/test-answer-scoring.py` (known answer strings → correct/incorrect)

## Implementation Steps
1. Implement benchmark loaders + unified record format {id, prompt, gold}.
2. Implement generation loop (per-benchmark shard, save raw generations — enables re-scoring without re-generating).
3. Implement scoring; unit tests with tricky cases (fractions, sqrt forms, MCQ letters).
4. Run: 2 models × 5 benchmarks. GPU-hours dominated by max_tokens=32k on OlympiadBench/MATH500 — batch aggressively, vLLM handles.
5. Produce `results/comparison-table.md`: per-benchmark + avg, Vanilla vs Spectral, Δ column.

## Success Criteria
- [ ] All 10 (model × benchmark) runs complete; raw generations persisted
- [ ] Scoring unit tests pass
- [ ] Comparison table with per-benchmark accuracy + average
- [ ] Same prompt/scoring code path for both models (verified in code review)

## Risk Assessment
- GPQA gated dataset → user needs HF access approval; fallback: drop GPQA, note in report
- 1.7B model may generate degenerate/endless CoT → max_tokens cap handles; report truncation rate
- AIME n=30 noise: report per-benchmark but base conclusions on MATH500+OlympiadBench (pre-registered in brainstorm)
