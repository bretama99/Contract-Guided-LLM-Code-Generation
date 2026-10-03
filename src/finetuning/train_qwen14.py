from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
from collections import Counter
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

TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
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
                    f"Expected object at {path}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary.replace(path)


def token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")

    if hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()

    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]

    if not isinstance(value, list):
        raise TypeError(
            f"Unsupported tokenizer output: {type(value).__name__}"
        )

    if not all(isinstance(token, int) for token in value):
        raise TypeError("Tokenizer output is not a flat token list")

    return value


def encode_text(
    tokenizer: Any,
    text: str,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    return token_ids(encoded)


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


def encode_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> tuple[
    dict[str, list[int]] | None,
    str,
    dict[str, Any],
]:
    current_id = str(
        record.get("task_id") or "unknown"
    )
    metadata: dict[str, Any] = {
        "task_id": current_id,
    }

    messages = record.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        return None, "invalid_messages", metadata

    if not all(isinstance(message, dict) for message in messages):
        return None, "invalid_message_object", metadata

    roles = [
        message.get("role")
        for message in messages
    ]

    if roles != ["system", "user", "assistant"]:
        metadata["roles"] = roles
        return None, "invalid_role_order", metadata

    for index, message in enumerate(messages):
        content = message.get("content")

        if not isinstance(content, str):
            metadata["message_index"] = index
            return None, "non_string_message_content", metadata

    assistant_content = messages[2]["content"].strip()

    if not assistant_content:
        return None, "empty_assistant_content", metadata

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not isinstance(prompt_text, str):
            return None, "invalid_prompt_template_output", metadata

        if not isinstance(full_text, str):
            return None, "invalid_full_template_output", metadata

        prompt_ids = encode_text(
            tokenizer,
            prompt_text,
        )
        full_ids = encode_text(
            tokenizer,
            full_text,
        )
    except Exception as error:
        metadata["error"] = str(error)
        return None, "tokenization_error", metadata

    prefix_length = common_prefix_length(
        prompt_ids,
        full_ids,
    )
    target_tokens = len(full_ids) - prefix_length

    metadata.update(
        {
            "prompt_tokens": len(prompt_ids),
            "full_tokens": len(full_ids),
            "prefix_tokens": prefix_length,
            "assistant_target_tokens": target_tokens,
        }
    )

    if not full_ids:
        return None, "empty_full_sequence", metadata

    if prefix_length == 0:
        return None, "prompt_prefix_mismatch", metadata

    if target_tokens <= 0:
        return None, "empty_assistant_target", metadata

    if len(full_ids) > max_length:
        return None, "exceeds_max_length", metadata

    labels = (
        [-100] * prefix_length
        + full_ids[prefix_length:]
    )

    if all(label == -100 for label in labels):
        return None, "empty_training_labels", metadata

    return (
        {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        },
        "valid",
        metadata,
    )


class ContractDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.lengths: list[int] = []
        self.kept_records: list[dict[str, Any]] = []
        self.removed_records: list[dict[str, Any]] = []
        self.removal_reasons: Counter[str] = Counter()

        for record in load_jsonl(path):
            item, reason, metadata = encode_record(
                record,
                tokenizer,
                max_length,
            )

            if item is None:
                self.removal_reasons[reason] += 1
                self.removed_records.append(
                    {
                        **metadata,
                        "reason": reason,
                        "record": record,
                    }
                )
                continue

            self.items.append(item)
            self.lengths.append(
                len(item["input_ids"])
            )
            self.kept_records.append(record)

        if not self.items:
            raise ValueError(
                f"No usable records remain in {path}. "
                f"Removal reasons: {dict(self.removal_reasons)}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, list[int]]:
        return self.items[index]

    def summary(self) -> dict[str, Any]:
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
            "removed": len(self.removed_records),
            "removal_reasons": dict(
                sorted(self.removal_reasons.items())
            ),
            "minimum_tokens": ordered[0],
            "median_tokens": int(
                statistics.median(ordered)
            ),
            "p95_tokens": ordered[p95_index],
            "maximum_tokens": ordered[-1],
            "truncated_records": 0,
        }


class ContractCollator:
    def __init__(
        self,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id

    @staticmethod
    def pad(
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
            "input_ids": self.pad(
                features,
                "input_ids",
                self.pad_token_id,
            ),
            "attention_mask": self.pad(
                features,
                "attention_mask",
                0,
            ),
            "labels": self.pad(
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
            in TARGET_MODULES
        )
    }

    if not targets:
        raise RuntimeError(
            "No supported Qwen attention projections found"
        )

    return sorted(targets)


def make_training_arguments(
    args: argparse.Namespace,
) -> TrainingArguments:
    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "bf16": True,
        "fp16": False,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "logging_strategy": "steps",
        "logging_steps": 5,
        "logging_first_step": True,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "max_grad_norm": 1.0,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
    }

    if "eval_strategy" in parameters:
        values["eval_strategy"] = "steps"
    else:
        values["evaluation_strategy"] = "steps"

    if "overwrite_output_dir" in parameters:
        values["overwrite_output_dir"] = False

    if "save_safetensors" in parameters:
        values["save_safetensors"] = True

    return TrainingArguments(**values)


def make_trainer(
    *,
    model: torch.nn.Module,
    arguments: TrainingArguments,
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: Any,
) -> Trainer:
    values: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": ContractCollator(
            tokenizer.pad_token_id
        ),
    }

    parameters = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in parameters:
        values["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        values["tokenizer"] = tokenizer

    return Trainer(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU Qwen2.5-Coder-14B "
            "contract-generation LoRA training."
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

    if args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error(
            "--eval-steps and --save-steps must be positive"
        )

    if args.save_steps % args.eval_steps != 0:
        parser.error(
            "--save-steps must be a multiple of --eval-steps"
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
                "Tokenizer has no padding or EOS token"
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        args.output_dir / "filtered_train.jsonl",
        train_dataset.kept_records,
    )
    write_jsonl(
        args.output_dir / "filtered_val.jsonl",
        val_dataset.kept_records,
    )
    write_jsonl(
        args.output_dir / "removed_records.jsonl",
        [
            {
                "split": "train",
                **record,
            }
            for record in train_dataset.removed_records
        ]
        + [
            {
                "split": "validation",
                **record,
            }
            for record in val_dataset.removed_records
        ],
    )

    dataset_summary = {
        "max_length": args.max_length,
        "train": train_dataset.summary(),
        "validation": val_dataset.summary(),
        "total_retained": (
            len(train_dataset)
            + len(val_dataset)
        ),
        "total_removed": (
            len(train_dataset.removed_records)
            + len(val_dataset.removed_records)
        ),
    }

    (
        args.output_dir / "dataset_summary.json"
    ).write_text(
        json.dumps(
            dataset_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            dataset_summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
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

    trainer = make_trainer(
        model=model,
        arguments=make_training_arguments(args),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    result = trainer.train(
        resume_from_checkpoint=(
            args.resume_from_checkpoint
        )
    )

    trainer.save_metrics(
        "train",
        result.metrics,
    )
    trainer.save_state()

    final_dir = (
        args.output_dir / "final_adapter"
    )
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
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": (
                args.batch_size
                * args.grad_accum
            ),
            "maximum_length": args.max_length,
            "best_checkpoint": (
                trainer.state.best_model_checkpoint
            ),
            "best_eval_loss": (
                trainer.state.best_metric
            ),
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
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()