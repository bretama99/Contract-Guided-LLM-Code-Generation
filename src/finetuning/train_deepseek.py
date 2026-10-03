from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


TARGET_MODULE_CANDIDATES = {
    "q_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def common_prefix_length(
    first: list[int],
    second: list[int],
) -> int:
    length = 0

    for left, right in zip(first, second):
        if left != right:
            break

        length += 1

    return length


class ContractDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.lengths: list[int] = []

        for record in load_jsonl(path):
            task_id = str(record.get("task_id") or "unknown")
            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(
                    f"Invalid messages for task {task_id}"
                )

            roles = [
                message.get("role")
                if isinstance(message, dict)
                else None
                for message in messages
            ]

            if roles != ["system", "user", "assistant"]:
                raise ValueError(
                    f"Invalid role order for task {task_id}: {roles}"
                )

            prompt_ids = tokenizer.apply_chat_template(
                messages[:2],
                tokenize=True,
                add_generation_prompt=True,
            )

            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)

            if not isinstance(full_ids, list):
                full_ids = list(full_ids)

            prefix_length = common_prefix_length(
                prompt_ids,
                full_ids,
            )

            if prefix_length == 0:
                raise ValueError(
                    f"Prompt-prefix mismatch for task {task_id}"
                )

            if prefix_length >= len(full_ids):
                raise ValueError(
                    f"Empty assistant target for task {task_id}"
                )

            if len(full_ids) > max_length:
                raise ValueError(
                    f"Task {task_id} has {len(full_ids)} tokens, "
                    f"exceeding max_length={max_length}"
                )

            labels = (
                [-100] * prefix_length
                + full_ids[prefix_length:]
            )

            self.items.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

            self.lengths.append(len(full_ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, list[int]]:
        return self.items[index]

    def summary(self) -> dict[str, int]:
        ordered = sorted(self.lengths)
        p95_index = max(
            0,
            min(
                len(ordered) - 1,
                int(len(ordered) * 0.95) - 1,
            ),
        )

        return {
            "records": len(ordered),
            "minimum_tokens": ordered[0],
            "median_tokens": int(
                statistics.median(ordered)
            ),
            "p95_tokens": ordered[p95_index],
            "maximum_tokens": ordered[-1],
            "truncated_records": 0,
        }


class ContractCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad(
        self,
        features: list[dict[str, list[int]]],
        field: str,
        padding_value: int,
    ) -> torch.Tensor:
        tensors = [
            torch.tensor(
                feature[field],
                dtype=torch.long,
            )
            for feature in features
        ]

        return pad_sequence(
            tensors,
            batch_first=True,
            padding_value=padding_value,
        )

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._pad(
                features,
                "input_ids",
                self.pad_token_id,
            ),
            "attention_mask": self._pad(
                features,
                "attention_mask",
                0,
            ),
            "labels": self._pad(
                features,
                "labels",
                -100,
            ),
        }


def detect_target_modules(
    model: torch.nn.Module,
) -> list[str]:
    targets = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if (
            isinstance(module, torch.nn.Linear)
            and name.rsplit(".", 1)[-1]
            in TARGET_MODULE_CANDIDATES
        )
    }

    if not targets:
        raise RuntimeError(
            "No supported DeepSeek attention projections were found."
        )

    return sorted(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU DeepSeek-Coder-V2-Lite LoRA "
            "fine-tuning for contract generation."
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    for path in (
        args.model_path,
        args.train_file,
        args.val_file,
    ):
        if not path.exists():
            parser.error(f"Path not found: {path}")

    if args.max_length <= 0:
        parser.error("--max-length must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.grad_accum <= 0:
        parser.error("--grad-accum must be positive")

    if args.save_steps % args.eval_steps != 0:
        parser.error(
            "--save-steps must be a multiple of --eval-steps "
            "when loading the best model."
        )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a padding token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    train_dataset = ContractDataset(
        args.train_file,
        tokenizer,
        args.max_length,
    )
    val_dataset = ContractDataset(
        args.val_file,
        tokenizer,
        args.max_length,
    )

    dataset_summary = {
        "max_length": args.max_length,
        "train": train_dataset.summary(),
        "validation": val_dataset.summary(),
    }

    print(
        json.dumps(
            dataset_summary,
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    target_modules = detect_target_modules(model)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        ),
    )

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=ContractCollator(
            tokenizer.pad_token_id
        ),
    )

    result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.save_metrics(
        "train",
        result.metrics,
    )
    trainer.save_state()

    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    summary = {
        "model_path": str(args.model_path),
        "train_file": str(args.train_file),
        "validation_file": str(args.val_file),
        "output_directory": str(args.output_dir),
        "target_modules": target_modules,
        "dataset": dataset_summary,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
        "training": {
            "backend": "single_gpu",
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": (
                args.batch_size * args.grad_accum
            ),
            "maximum_length": args.max_length,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
        },
        "metrics": result.metrics,
    }

    (
        args.output_dir
        / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
