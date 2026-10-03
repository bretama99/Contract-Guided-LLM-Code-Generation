from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

GPU_ARGUMENT = "--gpu-id"

for index, argument in enumerate(sys.argv):
    if argument == GPU_ARGUMENT and index + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
        break
    if argument.startswith(GPU_ARGUMENT + "="):
        os.environ["CUDA_VISIBLE_DEVICES"] = argument.split("=", 1)[1]
        break

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


N_TRAIN = 2056
N_VAL = 109

TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

DEFAULT_MAX_LENGTH = 4096
DEFAULT_EPOCHS = 4.0
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_EVAL_STEPS = 50
DEFAULT_SAVE_STEPS = 50
DEFAULT_SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            rows.append(value)

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def project_root() -> Path:
    candidates = [
        Path.cwd().resolve(),
        *Path(__file__).resolve().parents,
    ]

    for candidate in candidates:
        if (
            (candidate / "dataset/training_data").is_dir()
            and (candidate / "outputs").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the contract_guided_llm_code_generation project root."
    )


def absolute_path(
    root: Path,
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def task_id(row: dict[str, Any]) -> str:
    return str(
        row.get("task_id")
        or row.get("id")
        or ""
    ).strip()


def validate_rows(
    rows: list[dict[str, Any]],
    expected_count: int,
    split: str,
) -> set[str]:
    if len(rows) != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} records, "
            f"found {len(rows)}"
        )

    identifiers: set[str] = set()

    for index, row in enumerate(rows, 1):
        current_id = task_id(row)

        if not current_id:
            raise ValueError(
                f"{split}:{index}: missing task_id"
            )

        if current_id in identifiers:
            raise ValueError(
                f"{split}:{index}: duplicate task_id {current_id}"
            )

        identifiers.add(current_id)

        messages = row.get("messages")

        if not isinstance(messages, list):
            raise ValueError(
                f"{current_id}: messages must be a list"
            )

        if len(messages) != 3:
            raise ValueError(
                f"{current_id}: expected exactly three messages"
            )

        roles = [
            message.get("role")
            if isinstance(message, dict)
            else None
            for message in messages
        ]

        if roles != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(
                f"{current_id}: roles must be "
                "system, user, assistant"
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    f"{current_id}: message must be an object"
                )

            if not isinstance(
                message.get("content"),
                str,
            ):
                raise ValueError(
                    f"{current_id}: message content must be text"
                )

        if not messages[2]["content"].strip():
            raise ValueError(
                f"{current_id}: empty assistant contract"
            )

    return identifiers


def check_evaluation_leakage(
    root: Path,
    train_ids: set[str],
    val_ids: set[str],
) -> None:
    evaluation_candidates = [
        root / "dataset/testing_data.jsonl",
        root / "dataset/testing_data.jsonl",
    ]

    evaluation_file = next(
        (
            path
            for path in evaluation_candidates
            if path.is_file()
        ),
        None,
    )

    if evaluation_file is None:
        return

    evaluation_rows = read_jsonl(evaluation_file)

    evaluation_ids = {
        task_id(row)
        for row in evaluation_rows
    }

    train_leakage = train_ids & evaluation_ids
    val_leakage = val_ids & evaluation_ids

    if train_leakage:
        raise ValueError(
            "Training/evaluation leakage detected: "
            f"{sorted(train_leakage)[:20]}"
        )

    if val_leakage:
        raise ValueError(
            "Validation/evaluation leakage detected: "
            f"{sorted(val_leakage)[:20]}"
        )


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
            "Tokenizer output is not a token list"
        )

    if not all(
        isinstance(item, int)
        for item in value
    ):
        raise TypeError(
            "Tokenizer output contains non-integer token IDs"
        )

    return value


def assistant_boundary(
    prompt_tokens: list[int],
    complete_tokens: list[int],
) -> int:
    boundary = 0

    for left, right in zip(
        prompt_tokens,
        complete_tokens,
    ):
        if left != right:
            break

        boundary += 1

    if boundary <= 0:
        raise ValueError(
            "Assistant boundary could not be identified"
        )

    if boundary >= len(complete_tokens):
        raise ValueError(
            "Assistant content could not be identified"
        )

    return boundary


class ContractDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.samples: list[dict[str, list[int]]] = []
        self.kept_records: list[dict[str, Any]] = []
        self.removed_records: list[dict[str, Any]] = []

        token_lengths: list[int] = []

        for row in rows:
            current_id = task_id(row)
            messages = row["messages"]

            try:
                prompt_tokens = token_ids(
                    tokenizer.apply_chat_template(
                        messages[:-1],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )

                complete_tokens = token_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )

                boundary = assistant_boundary(
                    prompt_tokens,
                    complete_tokens,
                )

                total_tokens = len(complete_tokens)

                if total_tokens > max_length:
                    self.removed_records.append(
                        {
                            "task_id": current_id,
                            "reason": "exceeds_max_length",
                            "token_count": total_tokens,
                        }
                    )
                    continue

                labels = (
                    [-100] * boundary
                    + complete_tokens[boundary:]
                )

                assistant_tokens = sum(
                    token != -100
                    for token in labels
                )

                if assistant_tokens <= 0:
                    raise ValueError(
                        "No assistant tokens remain after masking"
                    )

                self.samples.append(
                    {
                        "input_ids": complete_tokens,
                        "attention_mask": [
                            1
                        ] * total_tokens,
                        "labels": labels,
                    }
                )

                self.kept_records.append(row)
                token_lengths.append(total_tokens)

            except Exception as error:
                self.removed_records.append(
                    {
                        "task_id": current_id,
                        "reason": "tokenization_error",
                        "error": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )

        self.token_lengths = token_lengths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, list[int]]:
        return self.samples[index]

    def summary(self) -> dict[str, Any]:
        if not self.token_lengths:
            return {
                "records": 0,
                "removed": len(self.removed_records),
                "removal_reasons": self._removal_reasons(),
                "minimum_tokens": None,
                "median_tokens": None,
                "p95_tokens": None,
                "maximum_tokens": None,
                "truncated_records": 0,
            }

        ordered = sorted(self.token_lengths)

        p95_index = min(
            len(ordered) - 1,
            max(
                0,
                int(
                    0.95 * len(ordered)
                ) - 1,
            ),
        )

        return {
            "records": len(self.samples),
            "removed": len(self.removed_records),
            "removal_reasons": self._removal_reasons(),
            "minimum_tokens": min(ordered),
            "median_tokens": int(
                statistics.median(ordered)
            ),
            "p95_tokens": ordered[p95_index],
            "maximum_tokens": max(ordered),
            "truncated_records": 0,
        }

    def _removal_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}

        for record in self.removed_records:
            reason = str(
                record.get("reason", "unknown")
            )
            reasons[reason] = reasons.get(reason, 0) + 1

        return reasons


class ContractCollator:
    def __init__(
        self,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        samples: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        width = max(
            len(sample["input_ids"])
            for sample in samples
        )

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for sample in samples:
            padding = (
                width
                - len(sample["input_ids"])
            )

            input_ids.append(
                sample["input_ids"]
                + [self.pad_token_id] * padding
            )

            attention_mask.append(
                sample["attention_mask"]
                + [0] * padding
            )

            labels.append(
                sample["labels"]
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def detect_target_modules(
    model: torch.nn.Module,
) -> list[str]:
    found: set[str] = set()

    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]

        if leaf in TARGET_MODULES:
            found.add(leaf)

    missing = TARGET_MODULES - found

    if missing:
        raise RuntimeError(
            "Required LoRA target modules were not found: "
            f"{sorted(missing)}"
        )

    return sorted(found)


def load_tokenizer(
    model_path: Path,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.eos_token_id is None:
        raise RuntimeError(
            "Qwen tokenizer has no EOS token"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model(
    args: argparse.Namespace,
) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one GPU must be visible. "
            f"Visible GPUs: {torch.cuda.device_count()}"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    from transformers import AutoConfig, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    model_loader = AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM

    model = model_loader.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        device_map={"": 0},
        max_memory={
            0: args.max_memory,
            "cpu": "64GiB",
        },
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    return model


def make_training_arguments(
    args: argparse.Namespace,
) -> TrainingArguments:
    values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        ("warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters else "warmup_steps"): 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": 5,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "max_grad_norm": 1.0,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "tf32": True,
    }

    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" not in parameters:
        values["evaluation_strategy"] = values.pop(
            "eval_strategy"
        )

    values = {
        key: value
        for key, value in values.items()
        if key in parameters
    }

    return TrainingArguments(**values)


def make_trainer(
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


def latest_checkpoint(
    output_dir: Path,
) -> Path:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir()
    ]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {output_dir}"
        )

    checkpoints.sort(
        key=lambda path: int(
            path.name.rsplit("-", 1)[-1]
        )
    )

    return checkpoints[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-Coder-32B QLoRA training "
            "for high-quality TACO contract generation."
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
        "--gpu-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-memory",
        default="42GiB",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--attention-implementation",
        choices=(
            "flash_attention_2",
            "sdpa",
            "eager",
        ),
        default="flash_attention_2",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = project_root()

    args.model_path = absolute_path(
        root,
        args.model_path,
    )

    args.train_file = absolute_path(
        root,
        args.train_file,
    )

    args.val_file = absolute_path(
        root,
        args.val_file,
    )

    args.output_dir = absolute_path(
        root,
        args.output_dir,
    )

    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_path}"
        )

    if not args.train_file.is_file():
        raise FileNotFoundError(
            f"Training file not found: {args.train_file}"
        )

    if not args.val_file.is_file():
        raise FileNotFoundError(
            f"Validation file not found: {args.val_file}"
        )

    if args.max_length <= 0:
        raise ValueError(
            "--max-length must be positive"
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    if args.grad_accum <= 0:
        raise ValueError(
            "--grad-accum must be positive"
        )

    set_seed(args.seed)
    random.seed(args.seed)

    train_rows = read_jsonl(
        args.train_file
    )

    val_rows = read_jsonl(
        args.val_file
    )

    train_ids = validate_rows(
        train_rows,
        N_TRAIN,
        "train",
    )

    val_ids = validate_rows(
        val_rows,
        N_VAL,
        "validation",
    )

    overlap = train_ids & val_ids

    if overlap:
        raise ValueError(
            "Training/validation overlap detected: "
            f"{sorted(overlap)[:20]}"
        )

    check_evaluation_leakage(
        root,
        train_ids,
        val_ids,
    )

    tokenizer = load_tokenizer(
        args.model_path
    )

    train_dataset = ContractDataset(
        train_rows,
        tokenizer,
        args.max_length,
    )

    val_dataset = ContractDataset(
        val_rows,
        tokenizer,
        args.max_length,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty"
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty"
        )

    dataset_summary = {
        "train_file": str(args.train_file),
        "validation_file": str(args.val_file),
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
        "train_validation_overlap": len(overlap),
        "evaluation_leakage_checked": True,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        args.output_dir
        / "dataset_summary.json"
    ).write_text(
        json.dumps(
            dataset_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
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

    print(
        json.dumps(
            dataset_summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        print(
            "DRY RUN PASSED",
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU after "
            f"--gpu-id={args.gpu_id}; found "
            f"{torch.cuda.device_count()}"
        )

    model = load_model(args)

    target_modules = detect_target_modules(
        model
    )

    print(
        json.dumps(
            {
                "physical_gpu": args.gpu_id,
                "visible_cuda_device": 0,
                "target_modules": target_modules,
                "quantization": "4bit_nf4",
                "compute_dtype": "bfloat16",
                "lora": {
                    "rank": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "bias": "none",
                },
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
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
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

    if hasattr(
        model,
        "enable_input_require_grads",
    ):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    resume_checkpoint: str | None = None

    if args.resume:
        if args.resume == "latest":
            resume_checkpoint = str(
                latest_checkpoint(
                    args.output_dir
                )
            )
        else:
            resume_checkpoint = str(
                absolute_path(
                    root,
                    args.resume,
                )
            )

        if not Path(
            resume_checkpoint
        ).is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{resume_checkpoint}"
            )

    training_arguments = make_training_arguments(
        args
    )

    trainer = make_trainer(
        model=model,
        arguments=training_arguments,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    trainer.save_metrics(
        "train",
        result.metrics,
    )

    trainer.save_state()

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    best_eval_loss = (
        trainer.state.best_metric
    )

    final_dir = (
        args.output_dir
        / "final_adapter_eval_loss"
    )

    trainer.save_model(
        str(final_dir)
    )

    tokenizer.save_pretrained(
        final_dir
    )

    summary = {
        "model": {
            "path": str(args.model_path),
            "architecture": model.config.model_type,
            "quantization": "4bit_nf4",
            "compute_dtype": "bfloat16",
        },
        "dataset": dataset_summary,
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": "none",
            "target_modules": target_modules,
        },
        "training": {
            "physical_gpu": args.gpu_id,
            "visible_cuda_device": 0,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": (
                args.batch_size
                * args.grad_accum
            ),
            "max_length": args.max_length,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "seed": args.seed,
            "attention_implementation": (
                args.attention_implementation
            ),
        },
        "selection": {
            "best_checkpoint_by_eval_loss": (
                best_checkpoint
            ),
            "best_eval_loss": best_eval_loss,
            "saved_adapter": str(final_dir),
            "downstream_selection_required": True,
            "criterion": (
                "Validation loss is used for checkpoint "
                "selection during training. Final model "
                "selection for the research comparison must "
                "use downstream TACO contract-generation "
                "quality and code-generation Pass@1."
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