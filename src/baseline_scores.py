"""Per-step forward-pass baseline scorers (entropy / perplexity / local log-prob) for the probe.

These are the cheap competitors spectral strength / diversity are benchmarked against in the causal
probe: one student forward per sample gives the predictive distribution at each response position,
aggregated to steps. Written to its own parquet keyed by id, so the probe can join it next to the
spectral scores.

    python src/baseline_scores.py --config configs/capture-diversity.yaml [--limit N]

Reuses the same model/data/chunking as gradient capture; no gradients are needed here.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM

from gradient_utils import shift_for_causal_lm, token_distribution_stats
from segmentation import record_step_spans


def aggregate_steps(values: torch.Tensor, step_rows: list[tuple[int, int]], perplexity: bool) -> list[float]:
    """Mean over each step's tokens; for perplexity, exp of the mean NLL."""
    out = []
    for start, end in step_rows:
        if end <= start:
            out.append(0.0)
        elif perplexity:
            out.append(float(torch.exp(values[start:end].mean()).item()))
        else:
            out.append(float(values[start:end].mean().item()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/capture-diversity.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/baseline-step-scores.parquet")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    device = config["device"]
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attn_implementation"],
        )
        .to(device)
        .eval()
    )
    unembedding = model.get_output_embeddings().weight.float()

    with open(config["data_path"]) as handle:
        records = [json.loads(line) for line in handle]
    if args.limit:
        records = records[: args.limit]

    rows, started = [], time.time()
    for index, record in enumerate(records):
        response_start, response_end = record["response_token_span"]
        step_rows = [(a - response_start, b - response_start) for a, b in record_step_spans(record)]
        input_ids = torch.tensor([record["input_ids"]], device=device)
        # no_grad is essential: a plain forward over a ~16k-token sequence would retain the whole
        # autograd graph (tens of GB) and OOM. We only need the logits, never a backward.
        with torch.no_grad():
            hidden = model.model(input_ids=input_ids).last_hidden_state[0]
            hidden_rows, targets = shift_for_causal_lm(hidden, input_ids[0], (response_start, response_end))
            entropy, nll, target_logprob = token_distribution_stats(
                hidden_rows, targets, unembedding, chunk_size=config["chunk_size"]
            )
        rows.append(
            {
                "id": record["id"],
                "step_entropy": aggregate_steps(entropy.cpu(), step_rows, perplexity=False),
                "step_perplexity": aggregate_steps(nll.cpu(), step_rows, perplexity=True),
                "step_logprob": aggregate_steps(target_logprob.cpu(), step_rows, perplexity=False),
            }
        )
        del hidden, hidden_rows, entropy, nll, target_logprob, input_ids
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(records)} ({(time.time() - started) / (index + 1):.2f}s/sample)")

    frame = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} rows -> {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
