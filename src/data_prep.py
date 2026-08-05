"""Phase 2: sample long-CoT trajectories, tokenize, and segment them into steps.

Output: JSONL with one record per sample containing input_ids, the supervised response
span, and absolute token spans for every reasoning step. Step text is not stored — it is
recovered exactly by decoding input_ids[token_start:token_end] (round-trip asserted below).

    python src/data_prep.py --config configs/data-config.yaml
"""

import argparse
import json
import statistics
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from segmentation import segment_response_token_spans

PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
SHUFFLE_BUFFER = 10_000


def build_record(tokenizer, problem: str, response: str) -> dict:
    """Tokenize one problem/response pair and map its steps to absolute token spans."""
    prompt = PROMPT_TEMPLATE.format(problem=problem)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids, step_spans = segment_response_token_spans(tokenizer, prompt_ids, response)
    input_ids = prompt_ids + response_ids
    return {
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "response_token_span": [len(prompt_ids), len(input_ids)],
        "steps": [{"token_start": start, "token_end": end} for start, end in step_spans],
        "n_tokens": len(input_ids),
    }


def verify_span_alignment(tokenizer, record: dict) -> bool:
    """Decoded step spans must reconstruct the original response text, step by step."""
    decoded = "".join(
        tokenizer.decode(record["input_ids"][step["token_start"] : step["token_end"]])
        for step in record["steps"]
    )
    return decoded.strip() == record["response"].strip()


def iter_samples(config: dict):
    """Stream the source dataset in shuffled order, yielding (problem, response) pairs.

    Shuffling with the configured seed makes the 2k selection a random draw from the
    corpus rather than its first rows, while staying deterministic and streaming.
    """
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"], streaming=True)
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=SHUFFLE_BUFFER)
    wanted = config.get("category_filter")
    for row in dataset:
        category = str(row.get("category", row.get("source", ""))).lower()
        if wanted and wanted not in category:
            continue
        problem = row.get("problem") or row.get("question") or row.get("input")
        response = row.get("solution") or row.get("response") or row.get("output")
        if problem and response:
            yield problem, response


def percentiles(values: list[int]) -> str:
    """Compact distribution summary: p10 / median / p90 / max."""
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return f"p10={pick(0.1)} median={pick(0.5)} p90={pick(0.9)} max={ordered[-1]}"


def log_stats(records: list[dict]) -> None:
    """Token-length, steps/sample and tokens/step distributions (phase 2 step 3)."""
    tokens = [record["n_tokens"] for record in records]
    steps = [len(record["steps"]) for record in records]
    step_lengths = [
        step["token_end"] - step["token_start"] for record in records for step in record["steps"]
    ]
    print(f"tokens/sample : mean={statistics.mean(tokens):.0f} {percentiles(tokens)}")
    print(f"steps/sample  : mean={statistics.mean(steps):.1f} {percentiles(steps)}")
    print(f"tokens/step   : mean={statistics.mean(step_lengths):.1f} {percentiles(step_lengths)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-config.yaml")
    parser.add_argument("--limit-scan", type=int, default=50000, help="max source rows to scan")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

    records, scanned, skipped_long, skipped_few_steps = [], 0, 0, 0
    for problem, response in iter_samples(config):
        scanned += 1
        if scanned > args.limit_scan or len(records) >= config["n_samples"]:
            break

        record = build_record(tokenizer, problem, response)
        if record["n_tokens"] > config["max_tokens"]:
            skipped_long += 1
            continue
        if len(record["steps"]) < 2:  # nothing to select between
            skipped_few_steps += 1
            continue

        record["id"] = len(records)
        records.append(record)
        if len(records) <= 20 and not verify_span_alignment(tokenizer, record):
            raise RuntimeError(f"step span round-trip failed on record {record['id']}")

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} samples -> {output_path}")
    print(f"scanned={scanned} skipped_long={skipped_long} skipped_few_steps={skipped_few_steps}")
    log_stats(records)


if __name__ == "__main__":
    main()
