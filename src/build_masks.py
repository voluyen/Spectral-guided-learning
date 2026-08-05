"""Phase 4: apply the paper's energy threshold p, report drop stats, emit training masks.

    python src/build_masks.py --config configs/data-config.yaml            # spectral masks
    python src/build_masks.py --config configs/data-config.yaml --vanilla  # + all-ones baseline

p is the paper's published value (configs/data-config.yaml: energy_threshold_p), not a
tuned hyperparameter — there is no sweep. The vanilla baseline is emitted by the same code
path so the two training runs differ only in the loss mask.
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import yaml

from segmentation import record_step_spans
from step_selection import build_loss_mask, select_steps_by_energy, selection_stats


def emit_masked_dataset(records: list[dict], strengths: dict[int, list[float]], threshold: float | None, output_path: Path) -> list[dict]:
    """Write a JSONL with a loss_mask per record. threshold=None -> supervise all response tokens."""
    all_stats = []
    with output_path.open("w") as handle:
        for record in records:
            step_spans = record_step_spans(record)
            if threshold is None:
                selected = list(range(len(step_spans)))
            else:
                selected = select_steps_by_energy(strengths[record["id"]], threshold)

            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "input_ids": record["input_ids"],
                        "loss_mask": build_loss_mask(len(record["input_ids"]), step_spans, selected),
                    }
                )
                + "\n"
            )
            all_stats.append(selection_stats(step_spans, selected))
    return all_stats


def summarize(stats: list[dict]) -> dict:
    """Aggregate per-sample stats; step-level and token-level drops reported separately."""
    return {
        "samples": len(stats),
        "step_drop_mean": statistics.mean(s["step_drop"] for s in stats),
        "step_drop_median": statistics.median(s["step_drop"] for s in stats),
        "token_drop_mean": statistics.mean(s["token_drop"] for s in stats),
        "token_drop_median": statistics.median(s["token_drop"] for s in stats),
        "steps_dropped_total": sum(s["n_steps_dropped"] for s in stats),
        "steps_total": sum(s["n_steps"] for s in stats),
        "tokens_dropped_total": sum(s["n_tokens_dropped"] for s in stats),
        "tokens_total": sum(s["n_tokens"] for s in stats),
        "samples_keeping_final_step": statistics.mean(s["keeps_final_step"] for s in stats),
        "samples_with_under_3_steps_kept": statistics.mean(
            float(s["n_steps_kept"] < 3) for s in stats
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-config.yaml")
    parser.add_argument("--strengths", default="data/spectral-strengths.parquet")
    parser.add_argument("--vanilla", action="store_true", help="also emit the all-ones baseline")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    with open(config["output_path"]) as handle:
        records = [json.loads(line) for line in handle]

    frame = pd.read_parquet(args.strengths)
    strengths = {int(row.id): list(row.step_strengths) for row in frame.itertuples()}
    records = [record for record in records if record["id"] in strengths]

    data_dir = Path(config["output_path"]).parent
    threshold = config["energy_threshold_p"]
    summaries = {}

    if args.vanilla:
        stats = emit_masked_dataset(records, strengths, None, data_dir / "train-vanilla.jsonl")
        summaries["vanilla"] = summarize(stats)

    spectral_path = data_dir / "train-spectral.jsonl"
    summaries[f"spectral p={threshold}"] = summarize(
        emit_masked_dataset(records, strengths, threshold, spectral_path)
    )
    print(f"p={threshold} (paper value) -> {spectral_path}")

    header = f"\n{'variant':<18} {'steps dropped':>14} {'median':>8} {'tokens dropped':>15} {'median':>8} {'keeps final':>12}"
    print(header)
    for name, summary in summaries.items():
        print(
            f"{name:<18} {summary['step_drop_mean']:>13.1%} {summary['step_drop_median']:>7.1%} "
            f"{summary['token_drop_mean']:>14.1%} {summary['token_drop_median']:>7.1%} "
            f"{summary['samples_keeping_final_step']:>11.1%}"
        )
        print(
            f"{'':<18} corpus-level: {summary['steps_dropped_total']}/{summary['steps_total']} steps, "
            f"{summary['tokens_dropped_total']}/{summary['tokens_total']} tokens dropped"
        )

    (data_dir / "selection-stats.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
