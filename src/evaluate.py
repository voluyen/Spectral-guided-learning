"""Phase 6: generate with vLLM and score both checkpoints on the paper's benchmarks.

    python src/evaluate.py --model checkpoints/vanilla --benchmarks all
    python src/evaluate.py --model checkpoints/spectral --benchmarks math500,aime24
    python src/evaluate.py --rescore results/spectral/raw/math500.jsonl   # no GPU needed

Each run gets its own subtree under --results-dir, keyed by --tag, so different
tracks/checkpoints never mix files in one flat directory:

    results/<tag>/summary.json
    results/<tag>/raw/<benchmark>.jsonl

Raw generations are persisted so scoring can be revised without regenerating.
"""

import argparse
import json
from pathlib import Path

import yaml

from answer_scoring import score_generation
from benchmarks import BENCHMARKS


def build_prompts(model_path: str, records: list[dict], config: dict) -> list[str]:
    """Wrap each benchmark's plain instruction into a chat-templated prompt when requested.

    Base models (chat_template unset, e.g. Qwen3-0.6B-Base) get the raw instruction text
    unchanged. Instruct/hybrid-thinking models (Qwen3-8B, Qwen2.5-7B-Instruct) need their own
    chat template so generation actually starts from "<|im_start|>assistant\n" the way they
    were trained/aligned, instead of continuing raw text as a base model would.
    """
    if not config.get("chat_template"):
        return [record["prompt"] for record in records]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model") or model_path)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.get("enable_thinking", True),
        )
        for record in records
    ]


def generate(model_path: str, records: list[dict], config: dict) -> list[list[dict]]:
    """Generate n samples per problem. Returns per-problem lists of {text, finish_reason}."""
    from vllm import LLM, SamplingParams

    prompts = build_prompts(model_path, records, config)

    # lora_adapter: model_path is an adapter-only checkpoint (--no-lora-merge in train_sft.py)
    # saved on top of config["base_model"] — load the base weights once via vLLM's native LoRA
    # support instead of a second merged ~16GB copy per model (see plans/reports handoff).
    if config.get("lora_adapter"):
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=config["base_model"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=config.get("lora_r", 16),
        )
        lora_request = LoRARequest("adapter", 1, model_path)
    else:
        llm = LLM(
            model=model_path,
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            dtype="bfloat16",
        )
        lora_request = None

    sampling = SamplingParams(
        n=config["n_samples"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config.get("seed", 42),
    )
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
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
    # --config is optional: the shell launchers (scripts/<family>/eval/eval_<model>.sh) pass every setting as a CLI
    # flag instead of a yaml. When both are given, CLI flags override the yaml.
    parser.add_argument("--config")
    parser.add_argument("--model", help="checkpoint path")
    parser.add_argument("--tag", help="short name used in output filenames (default: dir name)")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--rescore", help="score an existing raw generations file and exit")
    # generation/sampling settings (override the yaml, or stand in for it entirely)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", help="root dir; each run writes under <results-dir>/<tag>/")
    # chat template (instruct/hybrid-thinking models) + LoRA adapter (--no-lora-merge checkpoints)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--base-model", help="base model for chat template / LoRA adapter loading")
    parser.add_argument("--lora-adapter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora-r", type=int, help="adapter rank, for vLLM's max_lora_rank")
    args = parser.parse_args()

    if args.rescore:
        print(json.dumps(score_file(Path(args.rescore)), indent=2))
        return

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "results_dir": args.results_dir,
        "chat_template": args.chat_template,
        "enable_thinking": args.enable_thinking,
        "base_model": args.base_model,
        "lora_adapter": args.lora_adapter,
        "lora_r": args.lora_r,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    tag = args.tag or Path(args.model).name
    names = list(BENCHMARKS) if args.benchmarks == "all" else args.benchmarks.split(",")

    # everything for this run lives under one subtree, instead of a flat results/ dir
    # mixing raw generations and summaries from every tag/track together.
    run_dir = Path(config["results_dir"]) / tag
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name in names:
        raw_path = raw_dir / f"{name}.jsonl"
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

    results_path = run_dir / "summary.json"
    results_path.write_text(json.dumps(summaries, indent=2))
    average = sum(s["accuracy"] for s in summaries) / len(summaries)
    print(f"\n{tag}: average accuracy = {average:.1%} -> {results_path}")


if __name__ == "__main__":
    main()
