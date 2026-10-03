from __future__ import annotations

import argparse
import inspect
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed


TRAIN_SIZE = 2056
VAL_SIZE = 109
SEED = 42

PROJECTIONS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}


# ---------------------------------------------------------------------
# Files / dataset
# ---------------------------------------------------------------------

def project_root() -> Path:
    for path in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (path / "dataset/training_data").is_dir():
            return path.resolve()

    raise FileNotFoundError("Project root not found")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON: {path}:{line_no}"
                ) from e

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object: {path}:{line_no}"
                )

            rows.append(row)

    return rows


def locate_dataset(root: Path) -> tuple[Path, Path]:
    base = root / "dataset/training_data"

    train = base / "train.jsonl"
    val = base / "val.jsonl"

    if not train.is_file():
        raise FileNotFoundError(train)

    if not val.is_file():
        raise FileNotFoundError(val)

    return train, val


def task_id(row: dict[str, Any]) -> str:
    return str(
        row.get("task_id")
        or row.get("id")
        or ""
    ).strip()


def validate_rows(
    rows: list[dict[str, Any]],
    expected: int,
    split: str,
) -> set[str]:
    if len(rows) != expected:
        raise ValueError(
            f"{split}: expected {expected}, found {len(rows)}"
        )

    ids: set[str] = set()

    for row in rows:
        tid = task_id(row)
        messages = row.get("messages")

        if not tid:
            raise ValueError(f"{split}: missing task ID")

        if tid in ids:
            raise ValueError(f"{split}: duplicate task ID {tid}")

        ids.add(tid)

        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(
                f"{tid}: expected exactly 3 messages"
            )

        roles = [
            m.get("role") if isinstance(m, dict) else None
            for m in messages
        ]

        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"{tid}: expected roles "
                "system,user,assistant; found {roles}"
            )

        for m in messages:
            if not isinstance(m.get("content"), str):
                raise ValueError(
                    f"{tid}: message content must be a string"
                )

        if not messages[2]["content"].strip():
            raise ValueError(
                f"{tid}: assistant contract is empty"
            )

    return ids


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]

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
        raise TypeError("Expected token list")

    if not all(isinstance(x, int) for x in value):
        raise TypeError("Expected flat integer token list")

    return value


def common_prefix(
    a: list[int],
    b: list[int],
) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i

    return min(len(a), len(b))


class ContractDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        chat: Any,
        max_length: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.lengths: list[int] = []

        for row in rows:
            tid = task_id(row)
            messages = row["messages"]

            # Prompt = system + user, ending exactly where
            # assistant generation should start.
            prompt_ids = token_ids(
                chat.apply_chat_template(
                    messages[:2],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )

            # Full training sequence = system + user + assistant contract.
            full_ids = token_ids(
                chat.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )

            boundary = common_prefix(
                prompt_ids,
                full_ids,
            )

            if boundary <= 0:
                raise ValueError(
                    f"{tid}: prompt/full tokenization mismatch"
                )

            if boundary >= len(full_ids):
                raise ValueError(
                    f"{tid}: assistant target is empty"
                )

            # Absolutely no truncation.
            if len(full_ids) > max_length:
                raise ValueError(
                    f"{tid}: {len(full_ids)} tokens exceeds "
                    f"max_length={max_length}; "
                    "dataset is NOT truncated"
                )

            labels = (
                [-100] * boundary
                + full_ids[boundary:]
            )

            if all(x == -100 for x in labels):
                raise ValueError(
                    f"{tid}: no supervised assistant tokens"
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
        lengths = sorted(self.lengths)

        p95 = max(
            0,
            int(len(lengths) * 0.95) - 1,
        )

        return {
            "records": len(lengths),
            "minimum_tokens": lengths[0],
            "median_tokens": int(
                statistics.median(lengths)
            ),
            "p95_tokens": lengths[p95],
            "maximum_tokens": lengths[-1],
            "truncated_records": 0,
        }


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @staticmethod
    def pad(
        features: list[dict[str, list[int]]],
        key: str,
        value: int,
    ) -> torch.Tensor:
        return pad_sequence(
            [
                torch.tensor(
                    x[key],
                    dtype=torch.long,
                )
                for x in features
            ],
            batch_first=True,
            padding_value=value,
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


# ---------------------------------------------------------------------
# Tokenizer / model
# ---------------------------------------------------------------------

def load_tokenizer(
    kind: str,
    model_path: Path,
) -> tuple[Any, Any]:
    if kind == "ministral":
        from transformers import AutoProcessor

        chat = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )

        tokenizer = chat.tokenizer

    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        chat = tokenizer

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither PAD nor EOS token"
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    if not getattr(chat, "chat_template", None) and not getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        raise ValueError(
            "Tokenizer/processor has no chat template"
        )

    return chat, tokenizer


def load_model(
    kind: str,
    model_path: Path,
) -> torch.nn.Module:
    if kind == "ministral":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    else:
        from transformers import AutoModelForCausalLM

        # CodeLlama-13B-Instruct path.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    return model


# ---------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------

def lora_targets(
    model: torch.nn.Module,
    kind: str,
) -> list[str]:
    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        suffix = name.rsplit(".", 1)[-1]
        parts = set(name.split("."))

        if suffix not in PROJECTIONS:
            continue

        if "self_attn" not in parts:
            continue

        if (
            kind == "ministral"
            and not (
                {"language_model", "text_model"}
                & parts
            )
        ):
            continue

        targets.append(name)

    found = {
        x.rsplit(".", 1)[-1]
        for x in targets
    }

    if found != PROJECTIONS:
        raise RuntimeError(
            f"Expected attention projections "
            f"{sorted(PROJECTIONS)}, "
            f"found {sorted(found)}"
        )

    return sorted(targets)


# ---------------------------------------------------------------------
# Trainer arguments
# ---------------------------------------------------------------------

def training_arguments(
    args: argparse.Namespace,
) -> TrainingArguments:
    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": False,

        "num_train_epochs": args.epochs,

        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,

        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        ("warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters else "warmup_steps"): 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",

        "bf16": True,
        "fp16": False,

        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 20,

        "logging_strategy": "steps",
        "logging_steps": 5,
        "logging_first_step": True,

        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        "prediction_loss_only": True,

        "max_grad_norm": 1.0,

        "remove_unused_columns": False,

        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,

        "report_to": [],

        "seed": SEED,
        "data_seed": SEED,
    }

    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in parameters
        else "evaluation_strategy"
    )

    values[strategy_key] = "steps"

    if args.max_steps > 0:
        values["max_steps"] = args.max_steps

    return TrainingArguments(
        **{
            key: value
            for key, value in values.items()
            if key in parameters
        }
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LoRA SFT for Ministral or Llama/CodeLlama "
            "using the fixed TACO 2165 contract dataset"
        )
    )

    p.add_argument(
        "--kind",
        choices=["ministral", "llama"],
        required=True,
    )

    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--epochs",
        type=float,
        default=4,
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--resume",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root = project_root()

    args.model_path = (
        args.model_path.expanduser()
        if args.model_path.is_absolute()
        else root / args.model_path
    ).resolve()

    args.output_dir = (
        args.output_dir.expanduser()
        if args.output_dir.is_absolute()
        else root / args.output_dir
    ).resolve()

    train_file, val_file = locate_dataset(root)

    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)

    train_ids = validate_rows(
        train_rows,
        TRAIN_SIZE,
        "train",
    )

    val_ids = validate_rows(
        val_rows,
        VAL_SIZE,
        "validation",
    )

    overlap = train_ids & val_ids

    if overlap:
        raise ValueError(
            f"Train/validation overlap: "
            f"{sorted(overlap)[:10]}"
        )

    print(
        json.dumps(
            {
                "train_file": str(train_file),
                "validation_file": str(val_file),
                "train_records": len(train_rows),
                "validation_records": len(val_rows),
                "total_records": (
                    len(train_rows)
                    + len(val_rows)
                ),
                "overlap": 0,
            },
            indent=2,
        ),
        flush=True,
    )

    set_seed(SEED)

    chat, tokenizer = load_tokenizer(
        args.kind,
        args.model_path,
    )

    train_dataset = ContractDataset(
        train_rows,
        chat,
        args.max_length,
    )

    val_dataset = ContractDataset(
        val_rows,
        chat,
        args.max_length,
    )

    print(
        json.dumps(
            {
                "train": train_dataset.summary(),
                "validation": val_dataset.summary(),
                "total_retained": (
                    len(train_dataset)
                    + len(val_dataset)
                ),
                "total_removed": 0,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        print("DRY RUN PASSED", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU using CUDA_VISIBLE_DEVICES"
        )

    model = load_model(
        args.kind,
        args.model_path,
    )

    targets = lora_targets(
        model,
        args.kind,
    )

    print(
        json.dumps(
            {
                "lora_target_count": len(targets),
                "lora_target_modules": targets,
            },
            indent=2,
        ),
        flush=True,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=targets,
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

    if hasattr(
        model,
        "enable_input_require_grads",
    ):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_arguments(args),
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": Collator(
            tokenizer.pad_token_id
        ),
    }

    trainer_parameters = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = chat
    else:
        trainer_kwargs["tokenizer"] = chat

    trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = None

    if args.resume:
        checkpoints = [
            p
            for p in args.output_dir.glob("checkpoint-*")
            if (
                p.is_dir()
                and p.name.rsplit("-", 1)[-1].isdigit()
            )
        ]

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {args.output_dir}"
            )

        resume_checkpoint = str(
            max(
                checkpoints,
                key=lambda p: int(
                    p.name.rsplit("-", 1)[-1]
                ),
            )
        )

        print(
            f"Resuming from {resume_checkpoint}",
            flush=True,
        )

    result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_metrics(
        "train",
        result.metrics,
    )

    trainer.save_state()

    # Because load_best_model_at_end=True, trainer.model
    # corresponds to the best validation-loss checkpoint here.
    final_dir = (
        args.output_dir
        / "final_adapter_eval_loss_unselected"
    )

    trainer.save_model(str(final_dir))
    chat.save_pretrained(final_dir)

    summary = {
        "kind": args.kind,
        "model_path": str(args.model_path),

        "train_file": str(train_file),
        "validation_file": str(val_file),

        "train_records": len(train_dataset),
        "validation_records": len(val_dataset),
        "total_records": (
            len(train_dataset)
            + len(val_dataset)
        ),

        "epochs": args.epochs,

        "learning_rate": 2e-5,
        "effective_batch_size": 16,

        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_projections": sorted(PROJECTIONS),

        "attention_implementation": "sdpa",

        "best_checkpoint_by_eval_loss": (
            trainer.state.best_model_checkpoint
        ),

        "best_eval_loss": (
            trainer.state.best_metric
        ),

        "saved_adapter": str(final_dir),

        "selection": (
            "Best validation-loss model is saved here; "
            "retained checkpoints may additionally be compared "
            "using downstream pass@1."
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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