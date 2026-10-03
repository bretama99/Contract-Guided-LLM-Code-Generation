from __future__ import annotations

import argparse
import ast
import math
import os
import time
from pathlib import Path

from src.generation import generation_core as core
from src.prompts.generation_prompts import CONTRACT_SYSTEM_PROMPT


CONTRACT_TOP_KEYS = {
    "interface",
    "preconditions",
    "postconditions",
    "invariants",
}

CLAUSE_KEYS = {"id", "expression", "description"}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument(
        "--max-generation-seconds",
        type=float,
        default=100.0,
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--merge-adapter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--family-stop-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--schema-validation",
        choices=("strict", "basic"),
        default="strict",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not args.task_file.is_file():
        parser.error(f"Task file not found: {args.task_file}")

    for path in (args.model_path, args.adapter_path):
        if path is not None and not path.is_dir():
            parser.error(f"Directory not found: {path}")

    if args.gpu_id < 0 or args.start < 0 or args.seed < 0:
        parser.error("GPU ID, start and seed must be non-negative")

    if args.count is not None and args.count < 1:
        parser.error("Count must be positive")

    if args.max_output_tokens < 1:
        parser.error("Maximum output tokens must be positive")

    if (
        not math.isfinite(args.max_generation_seconds)
        or args.max_generation_seconds < 0
    ):
        parser.error(
            "Maximum generation seconds must be finite and non-negative"
        )

    return args


def build_prompt(task, tid):
    function_name = core.taco_function_name(task)

    parts = [
        f"TASK_ID: {tid}",
        "",
        "QUESTION:",
        core.taco_question(task),
        "",
    ]

    if function_name:
        parts.extend(
            [
                "Use Call-Based format.",
                f"FUNCTION NAME: {function_name}",
            ]
        )
    else:
        parts.append("Use Standard Input format.")

    parts.extend(["", "CONTRACT:"])

    return core.normalize_prompt_text("\n".join(parts))


def clauses_ok(clauses, prefix, require_nonempty=False):
    if not isinstance(clauses, list):
        return False

    if require_nonempty and not clauses:
        return False

    for index, clause in enumerate(clauses, start=1):
        if not isinstance(clause, dict):
            return False

        if set(clause) != CLAUSE_KEYS:
            return False

        if clause.get("id") != f"{prefix}{index}":
            return False

        expression = clause.get("expression")
        description = clause.get("description")

        if not isinstance(expression, str) or not expression.strip():
            return False

        if not isinstance(description, str) or not description.strip():
            return False

        try:
            ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, TypeError):
            return False

    return True


def basic_schema_ok(contract):
    return (
        isinstance(contract, dict)
        and set(contract) == CONTRACT_TOP_KEYS
        and isinstance(contract.get("interface"), dict)
        and all(
            isinstance(contract.get(field), list)
            for field in ("preconditions", "postconditions", "invariants")
        )
        and bool(contract["postconditions"])
    )


def schema_ok(contract):
    if not basic_schema_ok(contract):
        return False

    return (
        clauses_ok(contract["preconditions"], "P")
        and clauses_ok(
            contract["postconditions"],
            "Q",
            require_nonempty=True,
        )
        and clauses_ok(contract["invariants"], "I")
    )


def completed(path, tid):
    if not path.is_file():
        return False

    try:
        record = core.read_json(path)
    except (OSError, ValueError):
        return False

    if not isinstance(record, dict):
        return False

    if record.get("task_id") != tid:
        raise ValueError(
            f"Existing file belongs to a different task: {path}"
        )

    return (
        bool(str(record.get("raw_response") or "").strip())
        and record.get("generation_issue") is None
    )


def context_length(config):
    for candidate in (
        config,
        getattr(config, "text_config", None),
    ):
        value = getattr(candidate, "max_position_embeddings", None)

        if isinstance(value, int) and value > 0:
            return value

    return 8192


def add_ids(target, value):
    if isinstance(value, int):
        target.add(value)
    elif isinstance(value, (list, tuple, set)):
        target.update(
            token_id
            for token_id in value
            if isinstance(token_id, int)
        )


def valid_token_id(tokenizer, token):
    value = tokenizer.convert_tokens_to_ids(token)

    if not isinstance(value, int):
        return None

    if (
        tokenizer.unk_token_id is not None
        and value == tokenizer.unk_token_id
    ):
        return None

    return value


def stop_ids(tokenizer, model, model_type, family_stop_tokens=True):
    ids = set()

    add_ids(ids, tokenizer.eos_token_id)

    generation_config = getattr(model, "generation_config", None)

    if generation_config is not None:
        add_ids(ids, generation_config.eos_token_id)

    family_token = {
        "qwen2": "<|im_end|>",
        "llama": "<|eot_id|>",
        "mistral3": "<|end_of_turn|>",
    }.get(model_type)

    if family_stop_tokens and family_token:
        value = valid_token_id(tokenizer, family_token)

        if value is not None:
            ids.add(value)

    if not ids:
        raise ValueError("No valid stop token found")

    return sorted(ids)


def patch_deepseek_cache():
    from transformers.cache_utils import DynamicCache

    def from_legacy_cache(cls, past_key_values=None):
        if past_key_values is None:
            return cls()

        return cls(past_key_values)

    def to_legacy_cache(self):
        return tuple(
            (layer.keys, layer.values)
            for layer in self.layers
            if getattr(layer, "is_initialized", False)
        )

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)
    DynamicCache.to_legacy_cache = to_legacy_cache
    DynamicCache.get_usable_length = get_usable_length


def patch_deepseek_generation(model):
    from types import MethodType
    from transformers.cache_utils import Cache

    original_prepare = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if isinstance(past_key_values, Cache):
            if past_key_values.get_seq_length() == 0:
                past_key_values = None
            else:
                past_key_values = past_key_values.to_legacy_cache()

        kwargs.pop("position_ids", None)

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


def load_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers.utils import import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    common = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }

    config = AutoConfig.from_pretrained(
        str(args.model_path),
        **common,
    )

    model_type = str(getattr(config, "model_type", "")).lower()

    if model_type.startswith("deepseek"):
        patch_deepseek_cache()

    if model_type == "mistral3":
        config.tie_word_embeddings = False

        if hasattr(config, "text_config"):
            config.text_config.tie_word_embeddings = False

    tokenizer_args = dict(common)

    if model_type == "mistral3":
        tokenizer_args["mode"] = "test"

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        **tokenizer_args,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention = args.attention_implementation

    if model_type.startswith("deepseek"):
        attention = "eager"
    elif attention == "auto" and model_type == "mistral3":
        attention = "sdpa"

    model_args = {
        **common,
        "config": config,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }

    if attention != "auto":
        model_args["attn_implementation"] = attention

    if model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(args.model_path),
            **model_args,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            **model_args,
        )

    if model_type.startswith("deepseek"):
        patch_deepseek_generation(model)

    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_path.resolve()),
            is_trainable=False,
            local_files_only=True,
        )

        merge_adapter = (
            args.merge_adapter
            if args.merge_adapter is not None
            else model_type == "mistral3"
        )

        if merge_adapter:
            model = model.merge_and_unload()

    model.eval()
    model.config.use_cache = True

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True

    return (
        tokenizer,
        model,
        torch,
        model_type,
        attention,
        context_length(config),
        stop_ids(
            tokenizer,
            model,
            model_type,
            args.family_stop_tokens,
        ),
    )


def encode(tokenizer, task, tid):
    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": core.normalize_prompt_text(
                    CONTRACT_SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": build_prompt(task, tid),
            },
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    encoded = dict(encoded)

    if "input_ids" not in encoded:
        raise ValueError("Chat template did not return input_ids")

    return encoded


def generate_record(runtime, args, index, task, tid):
    (
        tokenizer,
        model,
        torch,
        model_type,
        attention,
        max_context,
        eos_ids,
    ) = runtime

    record = {
        "task_id": tid,
        "task_index": index,
        "model": args.model_name,
        "model_type": model_type,
        "attention_implementation": attention,
        "contract": None,
        "raw_response": "",
        "parse_ok": False,
        "schema_ok": False,
        "schema_validation": args.schema_validation,
        "prompt_tokens": None,
        "generated_tokens": 0,
        "hit_output_limit": False,
        "time_budget_reached": False,
        "ended_with_eos": False,
        "generation_seconds": None,
        "max_output_tokens": args.max_output_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "seed": args.seed + index,
        "generation_issue": None,
        "generated_at": None,
    }

    try:
        device = next(model.parameters()).device

        encoded = encode(tokenizer, task, tid)
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        prompt_tokens = int(encoded["input_ids"].shape[-1])
        record["prompt_tokens"] = prompt_tokens

        required = prompt_tokens + args.max_output_tokens

        if required > max_context:
            raise ValueError(
                f"{tid}: prompt={prompt_tokens}, "
                f"required={required}, context={max_context}"
            )

        torch.manual_seed(record["seed"])
        torch.cuda.manual_seed_all(record["seed"])

        generation = {
            **encoded,
            "max_new_tokens": args.max_output_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_ids,
        }

        if model_type == "mistral3":
            generation["logits_to_keep"] = 1

        if args.max_generation_seconds > 0:
            generation["max_time"] = args.max_generation_seconds

        started = time.perf_counter()

        try:
            with torch.inference_mode():
                output = model.generate(**generation)

            ids = (
                output[0, prompt_tokens:]
                .detach()
                .cpu()
                .tolist()
            )
        finally:
            elapsed = time.perf_counter() - started
            record["generation_seconds"] = round(elapsed, 3)

        record["generated_tokens"] = len(ids)
        record["hit_output_limit"] = (
            len(ids) >= args.max_output_tokens
        )
        record["ended_with_eos"] = bool(
            ids and ids[-1] in eos_ids
        )
        record["time_budget_reached"] = bool(
            args.max_generation_seconds > 0
            and elapsed >= args.max_generation_seconds
        )
        record["raw_response"] = tokenizer.decode(
            ids,
            skip_special_tokens=True,
        ).strip()

        if not record["raw_response"]:
            raise ValueError("Empty contract response")

        record["contract"] = core.extract_json_object(
            record["raw_response"]
        )
        record["parse_ok"] = isinstance(
            record["contract"],
            dict,
        )

        validate = (
            schema_ok
            if args.schema_validation == "strict"
            else basic_schema_ok
        )

        record["schema_ok"] = validate(record["contract"])

    except Exception as error:
        record["generation_issue"] = (
            f"{type(error).__name__}: {error}"
        )

    record["generated_at"] = core.now()

    return record


def main():
    args = parse_args()

    tasks = core.selected_tasks(
        core.load_tasks(args.task_file),
        args.start,
        args.count,
    )

    if not tasks:
        raise ValueError("No tasks selected")

    jobs = []
    destinations = set()

    for index, task in tasks:
        tid = core.task_id(task, index)
        path = core.output_path(args.output_dir, tid)
        resolved = path.resolve()

        if path.name.startswith("_") or resolved in destinations:
            raise ValueError(
                f"Reserved or duplicate output path: {path}"
            )

        destinations.add(resolved)
        jobs.append((index, task, tid, path))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "selected": len(jobs),
        "saved": 0,
        "skipped": 0,
        "parsed": 0,
        "schema_ok": 0,
        "generation_issues": 0,
        "output_limit_hits": 0,
        "time_budget_hits": 0,
        "generation_seconds": 0.0,
        "max_output": args.max_output_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "schema_validation": args.schema_validation,
        "greedy": True,
        "started_at": core.now(),
    }

    runtime = None

    for index, task, tid, destination in jobs:
        if not args.overwrite and completed(destination, tid):
            summary["skipped"] += 1
            continue

        if runtime is None:
            runtime = load_model(args)
            summary["model_type"] = runtime[3]

        record = generate_record(
            runtime,
            args,
            index,
            task,
            tid,
        )

        core.save_json(destination, record)

        failed = record["generation_issue"] is not None

        summary["saved"] += int(not failed)
        summary["generation_issues"] += int(failed)
        summary["parsed"] += int(record["parse_ok"])
        summary["schema_ok"] += int(record["schema_ok"])
        summary["output_limit_hits"] += int(
            record["hit_output_limit"]
        )
        summary["time_budget_hits"] += int(
            record["time_budget_reached"]
        )
        summary["generation_seconds"] += (
            record["generation_seconds"] or 0.0
        )

    summary["generation_seconds"] = round(
        summary["generation_seconds"],
        3,
    )
    summary["finished_at"] = core.now()

    count = args.count if args.count is not None else "all"

    core.save_json(
        args.output_dir / f"_summary_{args.start}_{count}.json",
        summary,
    )

    return int(summary["generation_issues"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())