"""Phase 3 causal probe: does a step-selection metric keep the steps the student actually needs?

Forced-answer sufficiency curve. For each sample we take the teacher CoT, keep only the top-p%
steps ranked by a metric, append an answer prompt, and let the student (Qwen3-1.7B-Base) fill in
`\\boxed{...}`. Accuracy vs token budget tells us which metric preserves solvability with the fewest
steps. A metric that ranks the causally-necessary steps highest keeps accuracy high at low budget.

Two questions this answers (see plan 260808-1425 phase 3):
  1. Importance axis: does `strength` beat `random` (selection works at all)?
  2. Novelty axis:    does `qd` (strength x IPR novelty) beat `strength` alone at the same budget
                      (diversity improves coverage)?

Controls: full CoT (upper bound), empty CoT (lower bound / headroom check), random (isolates
"fewer tokens" from "the right tokens"). Only samples that are solvable with the full CoT AND
unsolved with no CoT are kept, so removing steps can actually change the outcome.

Baselines (entropy / PPL / local-logprob) come from src/baseline_scores.py written to its own
parquet; when present they join in as extra metrics, else they are skipped with a warning.

    python src/baseline_scores.py --config configs/capture-diversity.yaml   # optional, first
    python src/causal_probe.py --config configs/causal-probe.yaml
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from answer_scoring import extract_boxed, score_generation

ANSWER_PRIME = "\n\nThe final answer is \\boxed{"


def load_samples(data_path: str, scores_path: str, tokenizer, baseline_path: str | None = None) -> list[dict]:
    """Join segmented records with their per-step spectral (and optional baseline) scores.

    Only samples usable for the probe are kept: scores aligned to steps, >=2 steps, gold present.
    Baseline scores (entropy/perplexity/logprob) are attached when the parquet exists.
    """
    scores = pd.read_parquet(scores_path).set_index("id")
    baselines = None
    if baseline_path and Path(baseline_path).exists():
        baselines = pd.read_parquet(baseline_path).set_index("id")
    samples = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rid = record["id"]
            if rid not in scores.index:
                continue
            row = scores.loc[rid]
            steps = record["steps"]
            strength = list(row.step_strengths)
            if len(steps) != len(strength) or len(steps) < 2:
                continue  # alignment guard / nothing to select between
            gold = extract_boxed(record["response"])
            if not gold:
                continue
            ids = record["input_ids"]
            step_texts = [tokenizer.decode(ids[s["token_start"] : s["token_end"]]) for s in steps]
            sample = {
                "id": rid,
                "prompt": record["prompt"],
                "gold": gold,
                "step_texts": step_texts,
                "strength": strength,
                "diversity": list(row.step_diversity),
                "novelty": list(row.step_novelty),
                "n_tokens": [s["token_end"] - s["token_start"] for s in steps],
            }
            if baselines is not None and rid in baselines.index:
                brow = baselines.loc[rid]
                if len(brow.step_entropy) == len(steps):
                    sample["entropy"] = list(brow.step_entropy)
                    sample["ppl"] = list(brow.step_perplexity)
                    sample["logprob"] = list(brow.step_logprob)
            samples.append(sample)
    return samples


def metric_scores(sample: dict, metric: str, rng: random.Random) -> list[float]:
    if metric == "qd":  # quality-diversity: spectral strength weighted by IPR novelty
        return [s * n for s, n in zip(sample["strength"], sample["novelty"])]
    if metric == "random":
        return [rng.random() for _ in sample["strength"]]
    return sample[metric]  # strength | diversity | novelty


def keep_by_token_budget(scores: list[float], n_tokens: list[int], budget: float) -> list[bool]:
    """Keep highest-scoring steps until ~budget of the CoT's TOKENS are kept (not step count).

    Budgeting by tokens equalizes how much text every metric's selection feeds the student, so the
    accuracy differences reflect *which* steps were chosen, not how long they were. Ranking by step
    count instead lets a metric win just by preferring longer steps.
    """
    total = sum(n_tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = [False] * len(scores)
    kept = 0
    for i in order:
        if kept >= budget * total:
            break
        keep[i] = True
        kept += n_tokens[i]
    if not any(keep):  # always keep at least the top-ranked step
        keep[order[0]] = True
    return keep


def build_prompt(sample: dict, keep: list[bool]) -> str:
    cot = "".join(text for text, keep_it in zip(sample["step_texts"], keep) if keep_it)
    return sample["prompt"] + cot + ANSWER_PRIME


def kept_token_fraction(sample: dict, keep: list[bool]) -> float:
    total = sum(sample["n_tokens"])
    return sum(n for n, k in zip(sample["n_tokens"], keep) if k) / max(total, 1)


def forced_answer(llm, prompts: list[str], config: dict) -> list[list[str]]:
    """Generate the boxed answer for each prompt; returns per-prompt lists of completions."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        n=config["n_samples"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_answer_tokens"],
        stop=["}"],  # prompt already ends with "\boxed{"; stop at the closing brace
        seed=config["seed"],
    )
    outputs = llm.generate(prompts, sampling)
    return [[completion.text for completion in output.outputs] for output in outputs]


def accuracy(completions: list[str], gold: str) -> float:
    """Fraction of completions whose reconstructed \\boxed{...} matches gold."""
    correct = sum(score_generation("\\boxed{" + text + "}", gold, "math") for text in completions)
    return correct / max(len(completions), 1)


def run(config: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    samples = load_samples(
        config["data_path"], config["scores_path"], tokenizer, config.get("baseline_scores_path")
    )[: config["limit"]]
    print(f"loaded {len(samples)} candidate samples")

    # Baseline metrics are usable only if every sample carries them (all-or-none per run).
    spectral = {"strength", "diversity", "novelty", "qd", "random"}
    metrics = [
        m for m in config["metrics"] if m in spectral or (samples and all(m in s for s in samples))
    ]
    dropped = [m for m in config["metrics"] if m not in metrics]
    if dropped:
        print(f"skipping metrics with no baseline scores: {dropped} (run baseline-scores first)")

    from vllm import LLM

    llm = LLM(
        model=config["model_name"],
        max_model_len=config["max_model_len"],
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        dtype="bfloat16",
    )

    # Headroom filter: full-CoT must solve, empty-CoT must fail, else removing steps proves nothing.
    all_keep = [[True] * len(s["step_texts"]) for s in samples]
    none_keep = [[False] * len(s["step_texts"]) for s in samples]
    full_prompts = [build_prompt(s, k) for s, k in zip(samples, all_keep)]
    empty_prompts = [build_prompt(s, k) for s, k in zip(samples, none_keep)]
    full_gen = forced_answer(llm, full_prompts, config)
    empty_gen = forced_answer(llm, empty_prompts, config)

    kept = []
    for sample, full_c, empty_c in zip(samples, full_gen, empty_gen):
        full_acc = accuracy(full_c, sample["gold"])
        empty_acc = accuracy(empty_c, sample["gold"])
        if full_acc >= config["full_threshold"] and empty_acc <= config["empty_threshold"]:
            sample["_full_acc"], sample["_empty_acc"] = full_acc, empty_acc
            kept.append(sample)
    print(
        f"headroom set: {len(kept)}/{len(samples)} samples "
        f"(full>={config['full_threshold']}, empty<={config['empty_threshold']})"
    )
    if not kept:
        print("no samples with headroom -- probe cannot run; consider a stronger student (4B)")
        return {"headroom": 0}

    # Build every (metric, budget) intervened prompt for the kept set, generate in one batch.
    rng = random.Random(config["seed"])
    jobs, prompts = [], []
    for metric in metrics:
        for budget in config["budgets"]:
            for sample in kept:
                keep = keep_by_token_budget(metric_scores(sample, metric, rng), sample["n_tokens"], budget)
                jobs.append((metric, budget, sample, kept_token_fraction(sample, keep)))
                prompts.append(build_prompt(sample, keep))
    generations = forced_answer(llm, prompts, config)

    # Aggregate the sufficiency curve: mean accuracy + mean kept-token fraction per (metric, budget).
    curve: dict = {}
    for (metric, budget, sample, tok_frac), gen in zip(jobs, generations):
        cell = curve.setdefault((metric, budget), {"acc": [], "tok": []})
        cell["acc"].append(accuracy(gen, sample["gold"]))
        cell["tok"].append(tok_frac)

    summary = {
        "headroom": len(kept),
        "full_acc": sum(s["_full_acc"] for s in kept) / len(kept),
        "empty_acc": sum(s["_empty_acc"] for s in kept) / len(kept),
        "curve": {},
    }
    for metric in metrics:
        for budget in config["budgets"]:
            cell = curve[(metric, budget)]
            summary["curve"][f"{metric}@{budget}"] = {
                "acc": sum(cell["acc"]) / len(kept),
                "tok_frac": sum(cell["tok"]) / len(kept),
            }

    print(f"\nfull CoT acc={summary['full_acc']:.3f}  empty CoT acc={summary['empty_acc']:.3f}")
    header = f"{'metric':<10} " + "  ".join(f"p={b:<5}" for b in config["budgets"])
    print("\nACCURACY (token-budgeted; higher at low p = ranks needed steps first)")
    print(header)
    for metric in metrics:
        cells = [f"{summary['curve'][f'{metric}@{b}']['acc']:.3f}" for b in config["budgets"]]
        print(f"{metric:<10} " + "  ".join(f"{c:<7}" for c in cells))
    print("\nTOKENS KEPT (fraction; should be ~equal across metrics per column = confound removed)")
    print(header)
    for metric in metrics:
        cells = [f"{summary['curve'][f'{metric}@{b}']['tok_frac']:.3f}" for b in config["budgets"]]
        print(f"{metric:<10} " + "  ".join(f"{c:<7}" for c in cells))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal-probe.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary = run(config)
    out_path = Path(config["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
