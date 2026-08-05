"""Phase 6: generate with vLLM and score both checkpoints on the paper's benchmarks.

    python src/evaluate.py --model checkpoints/vanilla --benchmarks all
    python src/evaluate.py --model checkpoints/spectral --benchmarks math500,aime24
    python src/evaluate.py --rescore results/raw/spectral-math500.jsonl   # no GPU needed

Raw generations are persisted so scoring can be revised without regenerating.
"""

import argparse
import json
from pathlib import Path

import yaml

from answer_scoring import score_generation
from benchmarks import BENCHMARKS


def generate(model_path: str, records: list[dict], config: dict) -> list[list[dict]]:
    """Generate n samples per problem. Returns per-problem lists of {text, finish_reason}."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        max_model_len=config["max_model_len"],
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        n=config["n_samples"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config.get("seed", 42),
    )
    outputs = llm.generate([record["prompt"] for record in records], sampling)
    return [
        [
            {"text": completion.text, "finish_reason": completion.finish_reason}
            for completion in output.outputs
        ]
        for output in outputs
    ]


def score_file(path: Path) -> dict:
    """Score a raw generations file; returns accuracy summary."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]

    correct = total = truncated = unboxed = 0
    for row in rows:
        for generation in row["generations"]:
            text = generation["text"]
            correct += int(score_generation(text, row["gold"], row["task_type"]))
            total += 1
            # hit the max_tokens cap = the model never finished its CoT (paper caps at 32k)
            truncated += int(generation.get("finish_reason") == "length")
            unboxed += int(len(text) > 0 and "\\boxed" not in text)

    return {
        "benchmark": path.stem,
        "accuracy": correct / total if total else 0.0,
        "n_problems": len(rows),
        "n_generations": total,
        "truncation_rate": truncated / total if total else 0.0,
        "no_boxed_answer_rate": unboxed / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval-config.yaml")
    parser.add_argument("--model", help="checkpoint path")
    parser.add_argument("--tag", help="short name used in output filenames (default: dir name)")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--rescore", help="score an existing raw generations file and exit")
    args = parser.parse_args()

    if args.rescore:
        print(json.dumps(score_file(Path(args.rescore)), indent=2))
        return

    config = yaml.safe_load(Path(args.config).read_text())
    tag = args.tag or Path(args.model).name
    names = list(BENCHMARKS) if args.benchmarks == "all" else args.benchmarks.split(",")

    raw_dir = Path(config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name in names:
        raw_path = raw_dir / f"{tag}-{name}.jsonl"
        if raw_path.exists():
            print(f"[{name}] reusing existing generations")
        else:
            records = BENCHMARKS[name]()
            print(f"[{name}] generating for {len(records)} problems x {config['n_samples']}")
            generations = generate(args.model, records, config)
            with raw_path.open("w") as handle:
                for record, completions in zip(records, generations):
                    handle.write(
                        json.dumps(
                            {
                                "id": record["id"],
                                "gold": record["gold"],
                                "task_type": record["task_type"],
                                "generations": completions,
                            }
                        )
                        + "\n"
                    )

        summary = score_file(raw_path)
        summary["model"] = tag
        summaries.append(summary)
        print(
            f"[{name}] accuracy = {summary['accuracy']:.1%}  "
            f"truncated = {summary['truncation_rate']:.1%}"
        )

    results_path = Path(config["results_dir"]) / f"{tag}-summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summaries, indent=2))
    average = sum(s["accuracy"] for s in summaries) / len(summaries)
    print(f"\n{tag}: average accuracy = {average:.1%} -> {results_path}")


if __name__ == "__main__":
    main()
