from __future__ import annotations

import argparse
import json
import time
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from src.generation.generation_core import (
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


SYSTEM_PROMPT = """
Solve the programming task exactly.
Return one complete Python solution only.
No tests, explanations, reasoning, Markdown, or code fences.
Preserve the required interface and I/O behavior.
Use an algorithm efficient enough for the stated constraints.
""".strip()

MAX_OUTPUT = 4096
SEED = 42


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int)
    p.add_argument(
        "--attention-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def prompt(task):
    fn = taco_function_name(task)

    parts = [
        "QUESTION:",
        taco_question(task),
        "",
        "Use Call-Based format." if fn else "Use Standard Input format.",
    ]

    if fn:
        parts.append(f"FUNCTION NAME: {fn}")

    parts += ["", "ANSWER:"]

    return normalize_prompt_text("\n".join(parts))


def clean(text):
    text = str(text or "").strip()

    m = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        re.I | re.S,
    )
    if m:
        return m.group(1).strip()

    return re.sub(
        r"^\s*\[PYTHON\]\s*|\s*\[/PYTHON\]\s*$",
        "",
        text,
        flags=re.I,
    ).strip()


def done(path):
    try:
        r = read_json(path)
        return bool(
            isinstance(r, dict)
            and str(r.get("code") or "").strip()
            and r.get("generation_issue") is None
        )
    except Exception:
        return False
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

    def get_usable_length(
        self,
        new_seq_length,
        layer_idx=0,
    ):
        return self.get_seq_length(layer_idx)

    DynamicCache.from_legacy_cache = classmethod(
        from_legacy_cache
    )
    DynamicCache.to_legacy_cache = to_legacy_cache
    DynamicCache.get_usable_length = get_usable_length


def patch_deepseek_generation(model):
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
        if isinstance(past_key_values, Cache):
            if past_key_values.get_seq_length() == 0:
                past_key_values = None
            else:
                past_key_values = (
                    past_key_values.to_legacy_cache()
                )

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
    
def load_model(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_id)
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
    import transformers.utils.import_utils as import_utils

    if not hasattr(
        import_utils,
        "is_torch_fx_available",
    ):
        import_utils.is_torch_fx_available = (
            lambda: hasattr(torch, "fx")
        )

    common = {
        "local_files_only": True,
        "trust_remote_code": True,
    }

    config = AutoConfig.from_pretrained(
        str(a.model_path),
        **common,
    )

    model_type = str(
        getattr(
            config,
            "model_type",
            "",
        )
    ).lower()
    if model_type.startswith("deepseek"):
        patch_deepseek_cache()

    if model_type == "mistral3":
        config.tie_word_embeddings = False

        if hasattr(
            config,
            "text_config",
        ):
            config.text_config.tie_word_embeddings = False

    tokenizer_args = dict(common)

    if model_type == "mistral3":
        tokenizer_args["mode"] = "test"

    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model_path),
        **tokenizer_args,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer has no EOS token"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention = a.attention_implementation

    if model_type == "mistral3":
        if attention == "auto":
            attention = "sdpa"

    elif model_type.startswith("deepseek"):
        attention = "eager"

    model_args = {
        **common,
        "config": config,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }

    if attention != "auto":
        model_args[
            "attn_implementation"
        ] = attention

    if model_type == "mistral3":
        from transformers import (
            Mistral3ForConditionalGeneration,
        )

        model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(a.model_path),
                **model_args,
            )
        )

    else:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(a.model_path),
                **model_args,
            )
        )

    if model_type.startswith("deepseek"):
        patch_deepseek_generation(model)

    model.eval()

    model.eval()
    model.config.use_cache = True

    if hasattr(
        model.config,
        "text_config",
    ):
        model.config.text_config.use_cache = True

    max_context = None

    for cfg in (
        config,
        getattr(
            config,
            "text_config",
            None,
        ),
    ):
        value = getattr(
            cfg,
            "max_position_embeddings",
            None,
        )

        if isinstance(value, int) and value > 0:
            max_context = value
            break

    if max_context is None:
        max_context = 8192

    stop_ids = set()

    if isinstance(
        tokenizer.eos_token_id,
        int,
    ):
        stop_ids.add(
            tokenizer.eos_token_id
        )

    generation_config = getattr(
        model,
        "generation_config",
        None,
    )

    if generation_config is not None:
        eos = generation_config.eos_token_id

        if isinstance(eos, int):
            stop_ids.add(eos)

        elif isinstance(
            eos,
            (list, tuple, set),
        ):
            stop_ids.update(
                token_id
                for token_id in eos
                if isinstance(token_id, int)
            )

    family_token = None

    if model_type.startswith("qwen"):
        family_token = "<|im_end|>"

    elif model_type == "llama":
        family_token = "<|eot_id|>"

    elif model_type == "mistral3":
        family_token = "<|end_of_turn|>"

    if family_token is not None:
        token_id = (
            tokenizer.convert_tokens_to_ids(
                family_token
            )
        )

        if (
            isinstance(token_id, int)
            and (
                tokenizer.unk_token_id is None
                or token_id != tokenizer.unk_token_id
            )
        ):
            stop_ids.add(token_id)

    if not stop_ids:
        raise ValueError(
            "No valid stop token found"
        )

    print(
        f"model_type={model_type} "
        f"attention={attention} "
        f"use_cache=True "
        f"context={max_context}",
        flush=True,
    )

    return (
        tokenizer,
        model,
        torch,
        model_type,
        int(max_context),
        sorted(stop_ids),
    )
    
def main():
    a = args()

    tasks = selected_tasks(
        load_tasks(a.task_file),
        a.start,
        a.count,
    )

    a.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, torch, model_type, context, stop_ids = load_model(a)
    device = next(model.parameters()).device

    saved = skipped = issues = 0

    for index, task in tasks:
        tid = task_id(task, index)
        path = output_path(a.output_dir, tid)

        if path.is_file() and not a.overwrite:
            skipped += 1
            continue

        raw = ""

        try:
            encoded = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt(task)},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

            encoded = {
                k: v.to(device)
                for k, v in dict(encoded).items()
            }

            width = int(encoded["input_ids"].shape[-1])

            if width + MAX_OUTPUT > context:
                raise ValueError(
                    f"prompt+output={width + MAX_OUTPUT} > context={context}"
                )

            torch.manual_seed(SEED + index)
            torch.cuda.manual_seed_all(SEED + index)

            generation = {
                **encoded,
                "max_new_tokens": MAX_OUTPUT,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": stop_ids,
            }

            if model_type == "mistral3":
                generation["logits_to_keep"] = 1

            with torch.inference_mode():
                started = time.perf_counter()
                output = model.generate(max_time=100.0, **generation)
                generation_seconds = time.perf_counter() - started

            ids = output[0, width:]

            raw = tokenizer.decode(
                ids,
                skip_special_tokens=True,
            ).strip()

            code = clean(raw)

            if not code:
                raise ValueError("empty generated code")

            save_json(
                path,
                {
                    "task_id": tid,
                    "model": a.model_name,
                    "raw_response": raw,
                    "code": code,
                    "generated_tokens": len(ids),
                    "generation_seconds": generation_seconds,
                    "generation_issue": None,
                    "generated_at": now(),
                },
            )

            saved += 1
            print(f"SAVED {tid} tokens={len(ids)}", flush=True)

        except Exception as e:
            issues += 1

            save_json(
                path,
                {
                    "task_id": tid,
                    "model": a.model_name,
                    "raw_response": raw,
                    "code": "",
                    "generation_issue": f"{type(e).__name__}: {e}",
                    "generated_at": now(),
                },
            )

            print(
                f"GENERATION_ISSUE {tid}: {type(e).__name__}: {e}",
                flush=True,
            )

    summary = {
        "selected": len(tasks),
        "saved": saved,
        "skipped": skipped,
        "generation_issues": issues,
        "max_output": MAX_OUTPUT,
    }

    save_json(
        a.output_dir / f"_summary_{a.start}_{a.count or 'all'}.json",
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()