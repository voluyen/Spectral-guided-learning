"""Phase 5: masked supervised fine-tuning (paper Eq. 9).

    python src/train_sft.py --config configs/train-vanilla.yaml
    python src/train_sft.py --config configs/train-spectral.yaml

Both runs share this code path; only the loss mask in the dataset differs.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data_collator import MaskedSFTCollator
from masked_loss import masked_cross_entropy


class MaskedSFTDataset(Dataset):
    """JSONL records of {input_ids, loss_mask} produced by build_masks.py."""

    def __init__(self, path: str):
        with open(path) as handle:
            self.records = [json.loads(line) for line in handle]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def supervised_token_count(self) -> int:
        return sum(sum(record["loss_mask"]) for record in self.records)


class MaskedSFTTrainer(Trainer):
    """Trainer applying the selective masked objective instead of the default LM loss.

    Declaring `model_accepts_loss_kwargs` makes Trainer count supervised tokens across
    every microbatch of a gradient-accumulation step and pass that count as
    `num_items_in_batch`; using it as Z yields one sum/Z per optimizer step (Eq. 9), and
    Trainer then skips its own division by the accumulation count. Averaging microbatch
    losses instead would over-weight sparsely supervised sequences — which is exactly what
    the spectral masks create.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True  # our loss shifts labels, so count over labels[..., 1:]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = masked_cross_entropy(outputs.logits, labels, denominator=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(config: dict) -> TrainingArguments:
    on_gpu = torch.cuda.is_available()  # CPU falls back to fp32 for local smoke runs
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": config["min_learning_rate"]},
        warmup_ratio=config["warmup_ratio"],
        bf16=on_gpu,
        use_cpu=not on_gpu,
        gradient_checkpointing=on_gpu,
        logging_steps=config.get("logging_steps", 5),
        save_strategy="epoch",
        save_total_limit=config.get("save_total_limit", 6),
        report_to=[],
        seed=config.get("seed", 42),
        remove_unused_columns=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the setup")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = MaskedSFTDataset(config["data_path"])

    if args.smoke:
        dataset.records = dataset.records[:8]
        config["epochs"] = 1

    print(f"{len(dataset)} samples, {dataset.supervised_token_count():,} supervised tokens")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False

    trainer = MaskedSFTTrainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=dataset,
        data_collator=MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    result = trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # evidence for the paper's "fewer supervised tokens" claim, recorded next to the checkpoint
    (Path(config["output_dir"]) / "run-summary.json").write_text(
        json.dumps(
            {
                "data_path": config["data_path"],
                "samples": len(dataset),
                "supervised_tokens": dataset.supervised_token_count(),
                "epochs": config["epochs"],
                "seed": config.get("seed", 42),
                "train_runtime_s": result.metrics.get("train_runtime"),
                "final_train_loss": result.metrics.get("train_loss"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
