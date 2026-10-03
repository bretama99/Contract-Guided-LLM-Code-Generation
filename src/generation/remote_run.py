
from __future__ import annotations

import json
import os
import runpy
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.generation import generation_core_extended as core  # noqa: E402

def _remote_post(args, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    if getattr(args, "remote_api_key", None):
        headers["Authorization"] = f"Bearer {args.remote_api_key}"
        headers["X-API-Key"] = args.remote_api_key

    last = None
    retries = int(getattr(args, "remote_retries", 5) or 5)
    timeout = float(getattr(args, "remote_timeout", 900.0) or 900.0)

    for attempt in range(retries):
        request = urllib.request.Request(
            args.remote_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return json.loads(reply.read().decode("utf-8"))

        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            last = f"HTTP {error.code}: {detail}"

            if 400 <= error.code < 500:
                raise RuntimeError(f"Remote rejected request: {last}")

        except Exception as error:
            last = f"{type(error).__name__}: {error}"

        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Remote failed after {retries} attempts: {last}")


# ------------------------------------------------------------------ runtime

def _eos_ids(args, config, text_config, tokenizer):

    from transformers import GenerationConfig

    try:
        generation_config = GenerationConfig.from_pretrained(
            str(args.model_path),
            local_files_only=True,
        )
    except Exception:
        generation_config = None
        print(
            "WARNING no generation_config.json beside --model-path; "
            "EOS derived from tokenizer and config only.",
            flush=True,
        )

    eos = set()

    for value in (
        tokenizer.eos_token_id,
        getattr(text_config, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
    ):
        if isinstance(value, int):
            eos.add(value)
        elif isinstance(value, (list, tuple)):
            eos.update(i for i in value if isinstance(i, int))

    if not eos:
        raise ValueError("Invalid EOS IDs")

    return eos


def remote_runtime(args, stage):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    config = AutoConfig.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )

    if config.model_type not in ("qwen2", "llama", "qwen3_5"):
        raise ValueError(f"Unsupported model type: {config.model_type}")

    text_config = (
        config.text_config if config.model_type == "qwen3_5" else config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path or args.model_path),
        local_files_only=True,
    )

    # MUST match load_runtime_32. Without it, encode_chat_32 silently stops
    # passing enable_thinking=False and every prompt gains a thinking block.
    tokenizer._contract_qwen35 = config.model_type == "qwen3_5"

    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError(
            "Tokenizer must provide EOS and the model's chat template"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos = _eos_ids(args, config, text_config, tokenizer)

    context = int(text_config.max_position_embeddings)
    tokenizer_context = getattr(tokenizer, "model_max_length", None)

    if isinstance(tokenizer_context, int) and 0 < tokenizer_context < 10**9:
        context = min(context, tokenizer_context)

    # --- defaults copied verbatim from load_runtime_32 ---
    temperature = args.temperature

    if temperature is None:
        temperature = 0.7 if stage == "code" else 0.0

    penalty = args.repetition_penalty

    if penalty is None:
        penalty = 1.05 if stage == "code" else 1.0

    ngram = args.no_repeat_ngram_size

    if ngram is None:
        ngram = 16 if stage == "code" else 0

    output_tokens = args.max_output_tokens or (
        4096 if stage == "code" else 2048
    )

    generation = GenerationConfig(
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=args.top_p if temperature > 0 else 1.0,
        top_k=args.top_k if temperature > 0 else 50,
        repetition_penalty=penalty,
        no_repeat_ngram_size=0,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        eos_token_id=sorted(eos),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        return_dict_in_generate=False,
    )

    settings = {
        "attention_implementation": f"remote:{args.remote_url}",
        "model_type": config.model_type,
        "enable_thinking": (
            False if config.model_type == "qwen3_5" else None
        ),
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": penalty,
        "generated_no_repeat_ngram_size": ngram,
        "max_output_tokens": output_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "stop_ids": sorted(eos),
        "adapter_path": (
            str(args.adapter_path)
            if getattr(args, "adapter_path", None)
            else None
        ),
        "remote_url": args.remote_url,
    }

    print(
        f"Remote generation via {args.remote_url} "
        f"(stage={stage}, context={context})",
        flush=True,
    )

    # Fail fast. One token now beats a broken endpoint discovered on
    # task 400 of 1,007.
    probe = _remote_post(
        args,
        {
            "input_ids": [int(tokenizer.eos_token_id)],
            "max_new_tokens": 1,
            "seed": args.seed,
            "temperature": 0.0,
            "do_sample": False,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "generated_no_repeat_ngram_size": 0,
            "stop_ids": sorted(eos),
            "max_generation_seconds": 30.0,
            "adapter": settings["adapter_path"],
        },
    )

    if "output_ids" not in probe:
        raise RuntimeError(
            "Remote reply has no 'output_ids'. The server does not "
            f"implement the expected contract. Keys returned: {sorted(probe)}"
        )

    if probe.get("attention_implementation"):
        settings["attention_implementation"] = probe[
            "attention_implementation"
        ]

    if probe.get("model_type") and probe["model_type"] != config.model_type:
        raise RuntimeError(
            f"Remote serves {probe['model_type']} but local config says "
            f"{config.model_type}. Wrong model on the server."
        )

    print(json.dumps(settings, indent=2), flush=True)

    return {
        "tokenizer": tokenizer,
        "model": None,
        "device": None,
        "max_context": context,
        "output_tokens": output_tokens,
        "generation_kwargs": {"generation_config": generation},
        "settings": settings,
        "remote": True,
    }


# --------------------------------------------------------------- generation


def remote_generate(runtime, args, encoded, index, stage):
    prompt_tokens = encoded["input_ids"].shape[-1]

    budget = min(
        runtime["output_tokens"],
        runtime["max_context"] - prompt_tokens,
    )

    if budget <= 0:
        raise ValueError(
            f"Prompt exceeds the model context: {prompt_tokens} tokens"
        )

    seed = args.seed + index
    settings = runtime["settings"]

    started = time.perf_counter()

    payload = _remote_post(
        args,
        {
            "input_ids": encoded["input_ids"][0].tolist(),
            "max_new_tokens": int(budget),
            "seed": int(seed),
            "temperature": settings["temperature"],
            "do_sample": settings["do_sample"],
            "top_p": settings["top_p"],
            "top_k": settings["top_k"],
            "repetition_penalty": settings["repetition_penalty"],
            "generated_no_repeat_ngram_size": settings[
                "generated_no_repeat_ngram_size"
            ],
            "stop_ids": settings["stop_ids"],
            "max_generation_seconds": settings["max_generation_seconds"],
            "adapter": settings["adapter_path"],
        },
    )

    wall = time.perf_counter() - started

    ids = [int(i) for i in payload["output_ids"]]

    # Server clock preferred; it excludes network time.
    elapsed = float(payload.get("generation_seconds") or wall)

    generation = runtime["generation_kwargs"]["generation_config"]
    eos = set(generation.eos_token_id)
    ended_with_eos = bool(ids and ids[-1] in eos)

    decode = runtime["tokenizer"].decode

    raw = decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    final = decode(
        ids[:-1] if ended_with_eos else ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()

    exhausted = payload.get("time_budget_exhausted")

    if exhausted is None:
        # Neither EOS nor the token budget stopped it, so the
        # server-side deadline must have.
        exhausted = not ended_with_eos and len(ids) < budget

    metadata = {
        "generation_seconds": round(elapsed, 3),
        "seed": seed,
        "generated_tokens": len(ids),
        "ended_with_eos": ended_with_eos,
        "hit_output_limit": len(ids) >= budget and not ended_with_eos,
        "time_budget_exhausted": bool(exhausted) and not ended_with_eos,
        "tokens_per_second": (
            round(len(ids) / elapsed, 3) if elapsed else None
        ),
        "remote_wall_seconds": round(wall, 3),
    }

    return raw, final, metadata


# ------------------------------------------------------------------ patching


_original_load_runtime = core.load_runtime_32
_original_generate = core.generate_response_32
_original_add_arguments = core.add_runtime_arguments


def patched_add_runtime_arguments(parser):
    _original_add_arguments(parser)
    parser.add_argument("--remote-url")
    parser.add_argument("--remote-api-key")
    parser.add_argument("--remote-timeout", type=float, default=900.0)
    parser.add_argument("--remote-retries", type=int, default=5)


def patched_load_runtime_32(args, stage):
    if getattr(args, "remote_url", None):
        if stage not in ("contracts", "code"):
            raise ValueError("stage must be contracts or code")
        return remote_runtime(args, stage)

    return _original_load_runtime(args, stage)


def patched_generate_response_32(runtime, args, encoded, index, stage):
    if runtime.get("model") is None:
        return remote_generate(runtime, args, encoded, index, stage)

    return _original_generate(runtime, args, encoded, index, stage)


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(__doc__, file=sys.stderr)
        print(
            "ERROR first argument must be the generation module name, "
            "for example generate_taco_vanilla_code_32",
            file=sys.stderr,
        )
        return 2

    module_name = sys.argv[1].removesuffix(".py")
    script = HERE / f"{module_name}.py"

    if not script.exists():
        print(f"ERROR no such script: {script}", file=sys.stderr)
        return 2

    core.add_runtime_arguments = patched_add_runtime_arguments
    core.load_runtime_32 = patched_load_runtime_32
    core.generate_response_32 = patched_generate_response_32

    # The generation script sees only its own arguments.
    sys.argv = [str(script)] + sys.argv[2:]

    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_code:
        return exit_code.code if exit_code.code is not None else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())