from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from src.generation.generation_core import (
    canonical_contract,
    load_tasks,
    normalize_prompt_text,
    now,
    output_path,
    read_json,
    save_json,
    selected_tasks,
    taco_function_name,
    taco_question,
    task_id,
)
from src.prompts.generation_prompts import CODE_GUIDANCE_RULES

MAX_OUTPUT = 4096
NO_REPEAT_NGRAM_SIZE = 32
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate contract-guided TACO solutions without starter code."
    )

    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)

    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )

    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not args.task_file.is_file():
        parser.error(f"Task file not found: {args.task_file}")

    if not args.contract_dir.is_dir():
        parser.error(f"Contract directory not found: {args.contract_dir}")

    if not args.model_path.is_dir():
        parser.error(f"Model path not found: {args.model_path}")

    if args.gpu_id < 0:
        parser.error("--gpu-id must be >= 0")

    if args.start < 0:
        parser.error("--start must be >= 0")

    if args.count is not None and args.count <= 0:
        parser.error("--count must be > 0")

    return args


def clean_code(text: str) -> str:
    text = str(text or "").strip()

    text = re.sub(
        r"^\s*\[PYTHON\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*\[/PYTHON\]\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*<python>\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*</python>\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    python_blocks = re.findall(
        r"```(?:python|py)\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if python_blocks:
        return max(python_blocks, key=len).strip()

    fenced_blocks = re.findall(
        r"```\s*(.*?)```",
        text,
        flags=re.DOTALL,
    )

    if fenced_blocks:
        return max(fenced_blocks, key=len).strip()

    text = re.sub(
        r"^\s*```(?:python|py)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


def index_contracts(directory: Path) -> dict[str, Path]:
    contracts: dict[str, Path] = {}

    for path in sorted(directory.rglob("taco_*.json")):
        try:
            record = read_json(path)
        except Exception:
            continue

        if not isinstance(record, dict):
            continue

        stored_task_id = record.get("task_id")

        if isinstance(stored_task_id, str) and stored_task_id.strip():
            tid = stored_task_id.strip()
        else:
            tid = path.stem

        previous = contracts.get(tid)

        if previous is not None and previous != path:
            raise ValueError(
                f"Duplicate contract for {tid}: "
                f"{previous} and {path}"
            )

        contracts[tid] = path

    return contracts


def get_contract(
    path: Path,
) -> tuple[str, str, bool | None]:
    record = read_json(path)

    if not isinstance(record, dict):
        raise ValueError(f"Invalid contract record: {path}")

    contract = canonical_contract(record.get("contract"))

    if contract is None:
        contract = canonical_contract(record.get("parsed_contract"))

    if contract is None:
        contract = canonical_contract(record)

    if contract is not None:
        return (
            json.dumps(
                contract,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "parsed_contract",
            record.get("schema_ok"),
        )

    raw = str(
        record.get("raw_response")
        or record.get("response")
        or ""
    ).strip()

    if raw:
        return (
            raw,
            "raw_response",
            record.get("schema_ok"),
        )

    raise ValueError(f"No usable contract content: {path}")


def build_user_prompt(
    task: dict[str, Any],
    contract_text: str,
) -> str:
    question = taco_question(task)
    function_name = taco_function_name(task)

    if function_name:
        interface = (
            "Call-Based format.\n"
            f"Required function name: {function_name}"
        )
    else:
        interface = "Standard Input/Output format."

    return normalize_prompt_text(
        "\n".join(
            (
                "TASK:",
                question,
                "",
                "CONTRACT:",
                contract_text,
                "",
                "REQUIRED INTERFACE:",
                interface,
                "",
                "ANSWER:",
            )
        )
    )


def encode_prompt(
    tokenizer: Any,
    task: dict[str, Any],
    contract_text: str,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": normalize_prompt_text(
                CODE_GUIDANCE_RULES
            ),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                task,
                contract_text,
            ),
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    encoded = dict(encoded)

    if "input_ids" not in encoded:
        raise ValueError(
            "Tokenizer chat template returned no input_ids"
        )

    return encoded


def output_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False

    try:
        record = read_json(path)
    except Exception:
        return False

    if not isinstance(record, dict):
        return False

    code = record.get("code")

    return (
        isinstance(code, str)
        and bool(code.strip())
        and record.get("generation_issue") is None
    )


def get_context_length(config: Any) -> int:
    candidates = (
        config,
        getattr(config, "text_config", None),
    )

    for cfg in candidates:
        if cfg is None:
            continue

        value = getattr(
            cfg,
            "max_position_embeddings",
            None,
        )

        if isinstance(value, int) and value > 0:
            return value

    return 8192


def add_token_ids(
    target: set[int],
    value: Any,
) -> None:
    if isinstance(value, int):
        target.add(value)
        return

    if isinstance(value, (list, tuple, set)):
        for token_id in value:
            if isinstance(token_id, int):
                target.add(token_id)


def add_named_stop_token(
    tokenizer: Any,
    stop_ids: set[int],
    token: str,
) -> None:
    token_id = tokenizer.convert_tokens_to_ids(token)

    if not isinstance(token_id, int):
        return

    if (
        tokenizer.unk_token_id is not None
        and token_id == tokenizer.unk_token_id
    ):
        return

    stop_ids.add(token_id)


def patch_legacy_transformers_compatibility(
    torch: Any,
) -> None:
    import transformers.utils.import_utils as import_utils

    if not hasattr(
        import_utils,
        "is_torch_fx_available",
    ):
        import_utils.is_torch_fx_available = (
            lambda: hasattr(torch, "fx")
        )


def patch_deepseek_cache() -> None:
    from transformers.cache_utils import DynamicCache

    def from_legacy_cache(
        cls,
        past_key_values=None,
    ):
        if past_key_values is None:
            return cls()

        return cls(
            past_key_values
        )

    def to_legacy_cache(self):
        return tuple(
            (
                layer.keys,
                layer.values,
            )
            for layer in self.layers
            if getattr(
                layer,
                "is_initialized",
                False,
            )
        )

    def get_usable_length(
        self,
        new_seq_length,
        layer_idx=0,
    ):
        return self.get_seq_length(
            layer_idx
        )

    DynamicCache.from_legacy_cache = classmethod(
        from_legacy_cache
    )

    DynamicCache.to_legacy_cache = (
        to_legacy_cache
    )

    DynamicCache.get_usable_length = (
        get_usable_length
    )


def patch_deepseek_generation(model) -> None:
    from types import MethodType
    from transformers.cache_utils import Cache

    original_prepare = (
        model.prepare_inputs_for_generation
    )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if isinstance(
            past_key_values,
            Cache,
        ):
            if (
                past_key_values.get_seq_length()
                == 0
            ):
                past_key_values = None
            else:
                past_key_values = (
                    past_key_values
                    .to_legacy_cache()
                )

        kwargs.pop(
            "position_ids",
            None,
        )

        return original_prepare(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    model.prepare_inputs_for_generation = MethodType(
        prepare_inputs_for_generation,
        model,
    )
def load_model(
    args: argparse.Namespace,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"
    )

    import torch

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    patch_legacy_transformers_compatibility(torch)

    common = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }

    config = AutoConfig.from_pretrained(
        str(args.model_path),
        **common,
    )

    model_type = str(
        getattr(config, "model_type", "")
    ).lower()

    if model_type.startswith("deepseek"):
        patch_deepseek_cache()
    if model_type == "mistral3":
        config.tie_word_embeddings = False

        text_config = getattr(
            config,
            "text_config",
            None,
        )

        if text_config is not None:
            text_config.tie_word_embeddings = False

    tokenizer_args = dict(common)

    if model_type == "mistral3":
        tokenizer_args["mode"] = "test"

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        **tokenizer_args,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer has no EOS token"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention = args.attention_implementation

    if model_type.startswith("deepseek"):
        attention = "eager"

    elif (
        model_type == "mistral3"
        and attention == "auto"
    ):
        attention = "sdpa"

    model_args: dict[str, Any] = {
        **common,
        "config": config,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }

    if attention != "auto":
        model_args["attn_implementation"] = attention

    if model_type == "mistral3":
        from transformers import (
            Mistral3ForConditionalGeneration,
        )

        model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(args.model_path),
                **model_args,
            )
        )

    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(args.model_path),
                **model_args,
            )
        )

    if model_type.startswith("deepseek"):
        patch_deepseek_generation(
            model
        )

    model.eval()
    model.config.use_cache = True

    text_model_config = getattr(
        model.config,
        "text_config",
        None,
    )

    if text_model_config is not None:
        text_model_config.use_cache = True

    stop_ids: set[int] = set()

    add_token_ids(
        stop_ids,
        tokenizer.eos_token_id,
    )

    generation_config = getattr(
        model,
        "generation_config",
        None,
    )

    if generation_config is not None:
        add_token_ids(
            stop_ids,
            generation_config.eos_token_id,
        )

    if model_type.startswith("qwen"):
        add_named_stop_token(
            tokenizer,
            stop_ids,
            "<|im_end|>",
        )

    elif model_type == "llama":
        add_named_stop_token(
            tokenizer,
            stop_ids,
            "<|eot_id|>",
        )

    elif model_type == "mistral3":
        add_named_stop_token(
            tokenizer,
            stop_ids,
            "<|end_of_turn|>",
        )

    if not stop_ids:
        raise ValueError(
            "No valid generation stop token found"
        )

    max_context = get_context_length(config)

    decoding_name = (
        "low_temperature_sampling"
        if model_type == "mistral3"
        else "greedy"
    )

    ngram_size = (
        16
        if model_type == "mistral3"
        else NO_REPEAT_NGRAM_SIZE
    )

    print(
        f"model_type={model_type} "
        f"attention={attention} "
        f"context={max_context} "
        f"use_cache=True "
        f"decoding={decoding_name} "
        f"no_repeat_ngram_size={ngram_size}",
        flush=True,
    )

    return (
        tokenizer,
        model,
        torch,
        model_type,
        attention,
        max_context,
        sorted(stop_ids),
    )


def main() -> None:
    args = parse_args()

    run_started = time.perf_counter()

    tasks = selected_tasks(
        load_tasks(args.task_file),
        args.start,
        args.count,
    )

    contracts = index_contracts(
        args.contract_dir
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        tokenizer,
        model,
        torch,
        model_type,
        attention,
        max_context,
        stop_ids,
    ) = load_model(args)

    device = next(model.parameters()).device

    saved = 0
    skipped = 0
    issues = 0

    print(
        f"selected={len(tasks)} "
        f"contracts={len(contracts)} "
        f"gpu={args.gpu_id} "
        f"model_type={model_type} "
        f"attention={attention} "
        f"context={max_context} "
        f"max_output={MAX_OUTPUT}",
        flush=True,
    )

    for index, task in tasks:
        tid = task_id(task, index)

        destination = output_path(
            args.output_dir,
            tid,
        )

        if (
            not args.overwrite
            and output_is_complete(destination)
        ):
            skipped += 1
            continue

        contract_path = contracts.get(tid)

        raw_response = ""

        try:
            if contract_path is None:
                raise FileNotFoundError(
                    f"No contract found for {tid}"
                )

            (
                contract_text,
                contract_source,
                contract_schema_ok,
            ) = get_contract(contract_path)

            encoded = encode_prompt(
                tokenizer,
                task,
                contract_text,
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            prompt_tokens = int(
                encoded["input_ids"].shape[-1]
            )

            available_tokens = (
                max_context - prompt_tokens
            )

            if available_tokens <= 0:
                raise ValueError(
                    f"Prompt exceeds context: "
                    f"prompt={prompt_tokens}, "
                    f"context={max_context}"
                )

            max_new_tokens = min(
                MAX_OUTPUT,
                available_tokens,
            )

            seed = SEED + index

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            generation_kwargs: dict[str, Any] = {
                **encoded,
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": (
                    stop_ids[0]
                    if len(stop_ids) == 1
                    else stop_ids
                ),
            }

            if model_type == "mistral3":
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": 0.08,
                        "top_p": 0.95,
                        "no_repeat_ngram_size": 16,
                        "logits_to_keep": 1,
                    }
                )
            else:
                generation_kwargs.update(
                    {
                        "do_sample": False,
                        "no_repeat_ngram_size": 32,
                    }
                )

            generation_started = (
                time.perf_counter()
            )

            with torch.inference_mode():
                output = model.generate(
                    **generation_kwargs
                )

            input_length = int(
                encoded["input_ids"].shape[-1]
            )

            generated_ids = output[
                0,
                input_length:,
            ]

            generated_tokens = int(
                generated_ids.shape[-1]
            )

            raw_response = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            code = clean_code(raw_response)

            if not code:
                raise ValueError(
                    "Generated solution is empty"
                )

            hit_output_limit = (
                generated_tokens
                >= max_new_tokens
            )

            save_json(
                destination,
                {
                    "task_id": tid,
                    "dataset": "taco",
                    "workflow": "contract_guided",
                    "provider": "local_transformers",
                    "model": args.model_name,
                    "model_type": model_type,
                    "attention_implementation": (
                        attention
                    ),
                    "starter_code_included": False,
                    "reference_solution_included": False,
                    "contract_path": str(
                        contract_path
                    ),
                    "contract_source": (
                        contract_source
                    ),
                    "contract_schema_ok": (
                        contract_schema_ok
                    ),
                    "raw_response": raw_response,
                    "code": code,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": (
                        generated_tokens
                    ),
                    "max_new_tokens_used": (
                        max_new_tokens
                    ),
                    "hit_output_limit": (
                        hit_output_limit
                    ),
                    "generation_seconds": round(
                        time.perf_counter()
                        - generation_started,
                        3,
                    ),
                    "generation_issue": None,
                    "generated_at": now(),
                },
            )

            saved += 1

            print(
                f"SAVED {tid} "
                f"tokens={generated_tokens} "
                f"limit={hit_output_limit}",
                flush=True,
            )

        except Exception as exc:
            issues += 1

            if issues == 1:
                traceback.print_exc()

            save_json(
                destination,
                {
                    "task_id": tid,
                    "dataset": "taco",
                    "workflow": "contract_guided",
                    "provider": "local_transformers",
                    "model": args.model_name,
                    "model_type": model_type,
                    "contract_path": (
                        str(contract_path)
                        if contract_path is not None
                        else None
                    ),
                    "raw_response": raw_response,
                    "code": "",
                    "generation_issue": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "generated_at": now(),
                },
            )

            print(
                f"GENERATION_ISSUE {tid}: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

    summary = {
        "selected": len(tasks),
        "saved": saved,
        "skipped": skipped,
        "generation_issues": issues,
        "model": args.model_name,
        "model_type": model_type,
        "attention_implementation": attention,
        "max_output": MAX_OUTPUT,
        "decoding": (
        "low_temperature_sampling"
            if model_type == "mistral3"
            else "greedy"
                ),
        "temperature": (
            0.08
            if model_type == "mistral3"
            else None
        ),
        "top_p": (
            0.95
            if model_type == "mistral3"
            else None
        ),
        "no_repeat_ngram_size": (
            16
            if model_type == "mistral3"
            else NO_REPEAT_NGRAM_SIZE
        ),
        "use_cache": True,
        "single_generation_per_task": True,
        "starter_code_used": False,
        "reference_solution_used": False,
        "question_truncation_used": False,
        "contract_truncation_used": False,
        "verification_used": False,
        "repair_used": False,
        "elapsed_seconds": round(
            time.perf_counter()
            - run_started,
            2,
        ),
    }

    save_json(
        args.output_dir
        / (
            f"_summary_{args.start}_"
            f"{args.count if args.count is not None else 'all'}"
            ".json"
        ),
        summary,
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