from __future__ import annotations

import ast
import inspect
import json
import math
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(directory, task_id_value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id_value))
    return Path(directory) / f"{name.strip('_') or 'task'}.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_tasks(path):
    text = Path(path).read_text(encoding="utf-8-sig")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if isinstance(data, dict):
        if isinstance(data.get("question"), str):
            data = [data]
        else:
            data = next(
                (
                    data[key]
                    for key in (
                        "tasks", "data", "items", "problems", "questions"
                    )
                    if isinstance(data.get(key), list)
                ),
                None,
            )

    if (
        not isinstance(data, list)
        or any(not isinstance(row, dict) for row in data)
    ):
        raise ValueError(f"Expected task objects in {path}")

    return data


def selected_tasks(tasks, start, count):
    if start < 0 or (count is not None and count < 1):
        raise ValueError("start must be non-negative and count positive")

    end = None if count is None else start + count
    return list(enumerate(tasks[start:end], start=start))


def task_id(task, index):
    value = task.get("task_id")

    if value not in (None, "", "None"):
        return str(value)

    if (
        task.get("split") is not None
        and task.get("split_index") is not None
    ):
        return f"taco_{task['split']}_{task['split_index']}"

    return f"taco_{index}"


def taco_question(task):
    value = task.get("question")
    return value.strip() if isinstance(value, str) else ""


def as_mapping(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}

    return value if isinstance(value, dict) else {}


def taco_interface(task):
    sources = (
        task,
        as_mapping(task.get("interface")),
        as_mapping(task.get("input_output")),
    )

    def field(*keys):
        for source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    name = field("function_name", "fn_name")
    signature = field("signature")
    mode = field("mode")
    class_name = field("class_name")

    if (
        name
        or mode in ("functional", "call_based")
        or class_name
        or (signature and signature != "stdin -> stdout")
    ):
        lines = [
            "Call-based format. "
            "Preserve the callable interface stated in the task."
        ]

        for label, value in (
            ("Required function name", name),
            ("Required class", class_name),
            ("Required signature", signature),
        ):
            if value:
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    return "Standard input/output format: read stdin and write stdout."


def prepare_jobs(selected, args):
    jobs = []
    destinations = set()

    for index, task in selected:
        tid = task_id(task, index)
        destination = output_path(args.output_dir, tid)
        resolved = destination.resolve()

        if (
            destination.name.startswith("_")
            or resolved in destinations
        ):
            raise ValueError(
                f"Reserved or duplicate output path: {destination}"
            )

        destinations.add(resolved)

        job = {
            "index": index,
            "task_id": tid,
            "task": task,
            "destination": destination,
        }

        if hasattr(args, "contract_dir"):
            job["contract_path"] = output_path(args.contract_dir, tid)

            if resolved == job["contract_path"].resolve():
                raise ValueError(
                    f"Code output would overwrite its contract: "
                    f"{destination}"
                )

        jobs.append(job)

    return jobs


def existing_record(job, field, overwrite):
    path = job["destination"]

    if overwrite or not path.exists():
        return None

    try:
        record = read_json(path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read existing output {path}: {error}"
        ) from error

    if (
        not isinstance(record, dict)
        or record.get("task_id") != job["task_id"]
        or (
            "task_index" in record
            and record["task_index"] != job["index"]
        )
    ):
        raise ValueError(
            f"Existing output belongs to a different task: {path}"
        )

    value = record.get(field)

    if value is None or (
        isinstance(value, str) and not value.strip()
    ):
        return None

    return record


def add_runtime_arguments(parser):
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu-id", type=int)
    gpu.add_argument("--gpu-ids")

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-memory-per-gpu", default="40GiB")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--max-generation-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Default: code 0.7; contracts 0.",
    )
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float)
    
    parser.add_argument(
        "--generated-only-repetition-penalty",
        action="store_true",
        help=(
            "Apply repetition penalty only to generated tokens. "
            "Intended for DeepSeek-Coder-33B when explicitly requested."
        ),
    )
    
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        help=(
            "Generated tokens only. Default: code 16; contracts 0. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )


def finish_arguments(parser, args):
    if args.start < 0 or (
        args.count is not None and args.count < 1
    ):
        parser.error("--start must be non-negative and --count positive")

    for name in ("max_generation_seconds", "repetition_penalty"):
        value = getattr(args, name)

        if value is not None and (
            not math.isfinite(value) or value <= 0
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be finite and positive"
            )

    if args.temperature is not None and (
        not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        parser.error("--temperature must be finite and non-negative")

    if not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("--top-p must be in (0, 1] and --top-k non-negative")

    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")

    if (
        args.no_repeat_ngram_size is not None
        and args.no_repeat_ngram_size < 0
    ):
        parser.error("--no-repeat-ngram-size must be non-negative")

    visible = (
        args.gpu_ids
        if args.gpu_ids is not None
        else str(args.gpu_id if args.gpu_id is not None else 3)
    )
    parts = visible.split(",")

    if any(not part.strip().isdecimal() for part in parts):
        parser.error(
            "GPU IDs must be comma-separated non-negative integers"
        )

    numbers = [int(part.strip()) for part in parts]

    if len(set(numbers)) != len(numbers):
        parser.error("GPU IDs must not contain duplicates")

    args.gpu_ids = ",".join(map(str, numbers))

    if not re.fullmatch(
        r"[1-9]\d*(?:\.\d+)?(?:GiB|MiB|GB|MB)",
        args.max_memory_per_gpu,
    ):
        parser.error("--max-memory-per-gpu must look like 40GiB")

    if not args.task_file.is_file():
        parser.error(f"Missing task file: {args.task_file}")

    for name in (
        "model_path", "tokenizer_path", "adapter_path", "contract_dir"
    ):
        path = getattr(args, name, None)

        if path is not None and not path.is_dir():
            parser.error(f"Missing directory: {path}")

    if (
        hasattr(args, "contract_dir")
        and args.output_dir.resolve() == args.contract_dir.resolve()
    ):
        parser.error("--output-dir must differ from --contract-dir")

    return args

def encode_chat_32(tokenizer, system, user):
    if getattr(tokenizer, "_contract_devstral", False):
        import torch
        from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        ids = list(tokenizer.mistral_tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[
                SystemMessage(content=system),
                UserMessage(content=user),
            ])
        ).tokens)

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    processor = getattr(
        tokenizer,
        "_contract_processor",
        None,
    )

    if processor is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
        )

        encoded = dict(encoded)
        encoded.pop("token_type_ids", None)

        return encoded

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **(
            {"enable_thinking": False}
            if getattr(tokenizer, "_contract_qwen35", False)
            else {}
        ),
    )

    return dict(
        tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
        )
    )


def load_runtime_32(args, stage):
    if stage not in ("contracts", "code"):
        raise ValueError("stage must be contracts or code")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )
    is_deepseek33 = (
        config.model_type == "llama"
        and "deepseek-coder-33b" in str(args.model_path).lower()
        )
    is_devstral24 = (
        config.model_type == "mistral"
        and "MistralForCausalLM" in (getattr(config, "architectures", None) or [])
        and (args.tokenizer_path or args.model_path).joinpath("tekken.json").is_file()
    )
    if not (
        config.model_type in ("qwen2", "llama", "qwen3_5")
        or is_devstral24
    ):
        raise ValueError(
            f"Unsupported model type: {config.model_type}. "
            "Expected Qwen2.5-Coder-Instruct, DeepSeek-Coder-33B-Instruct, "
            "Qwen3.5-27B, or Devstral-Small-24B."
        )

    text_config = (
        config.text_config
        if config.model_type == "qwen3_5"
        else config
    )
    model_class = AutoModelForCausalLM
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model_class = Qwen3_5ForConditionalGeneration
    
    if is_devstral24:
        from types import SimpleNamespace
        from mistral_common.protocol.instruct.validator import ValidationMode
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tekken = (args.tokenizer_path or args.model_path) / "tekken.json"
        mistral_tokenizer = MistralTokenizer.from_file(
            tekken,
            mode=ValidationMode.agnostic,
        )
        inner = mistral_tokenizer.instruct_tokenizer.tokenizer

        if (
            inner.bos_id != config.bos_token_id
            or inner.eos_id != config.eos_token_id
            or inner.n_words > config.vocab_size
        ):
            raise ValueError("Devstral tokenizer does not match the checkpoint")

        tokenizer = SimpleNamespace(
            mistral_tokenizer=mistral_tokenizer,
            decode=mistral_tokenizer.decode,
            bos_token_id=inner.bos_id,
            eos_token_id=inner.eos_id,
            pad_token_id=inner.eos_id,
            vocab_size=inner.n_words,
            model_max_length=config.max_position_embeddings,
            _contract_devstral=True,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_path or args.model_path),
            local_files_only=True,
        )

        if is_deepseek33 and args.tokenizer_path is None:
            raise ValueError(
                "DeepSeek-Coder-33B requires the verified tokenizer. "
                "Pass --tokenizer-path explicitly."
            )

        tokenizer._contract_devstral = False
        tokenizer._contract_qwen35 = (
            config.model_type == "qwen3_5"
        )

    if is_devstral24:
        if tokenizer.eos_token_id is None:
            raise ValueError("Devstral tokenizer does not provide EOS")
    else:
        if tokenizer.eos_token_id is None or not tokenizer.chat_template:
            raise ValueError(
                "Tokenizer must provide EOS and the model's chat template"
            )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    attention = (
        "sdpa"
        if args.attention_implementation == "auto"
        else args.attention_implementation
    )

    memory = {
        i: args.max_memory_per_gpu
        for i in range(torch.cuda.device_count())
    }
    memory["cpu"] = 0

    options = dict(
        config=config,
        dtype=dtype,
        device_map="auto",
        max_memory=memory,
        local_files_only=True,
        attn_implementation=attention,
    )

    if args.load_in_4bit:
        if getattr(config, "quantization_config", None):
            raise ValueError(
                "--load-in-4bit requires an unquantized checkpoint"
            )

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print(
        f"Loading {args.model_path}: GPUs={args.gpu_ids} "
        f"dtype={args.dtype} 4bit={args.load_in_4bit} "
        f"attention={attention}",
        flush=True,
    )

    model = model_class.from_pretrained(
        str(args.model_path),
        **options,
    )

    device_map = getattr(model, "hf_device_map", None) or {}

    has_offloading = any(
        str(device) in ("cpu", "disk")
        for device in device_map.values()
    )

    non_cuda_devices = {
        str(parameter.device)
        for parameter in model.parameters()
        if parameter.device.type != "cuda"
    }

    if has_offloading or non_cuda_devices:
        raise RuntimeError(
            "Model weights are not fully loaded on the selected GPUs. "
            f"Non-CUDA parameter devices: {sorted(non_cuda_devices)}. "
            "Use 4-bit loading or more GPU memory."
        )

    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_path),
            is_trainable=False,
            local_files_only=True,
        )

    model.eval()
    model.config.use_cache = True
    if config.model_type == "qwen3_5":
        model.config.text_config.use_cache = True

    eos = set()

    for value in (
        tokenizer.eos_token_id,
        getattr(text_config, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
        model.generation_config.eos_token_id,
    ):
        if isinstance(value, int):
            eos.add(value)
        elif isinstance(value, (list, tuple)):
            eos.update(value)

    vocab_size = model.get_output_embeddings().weight.shape[0]

    if not eos or any(
        not isinstance(i, int) or not 0 <= i < vocab_size
        for i in eos
    ):
        raise ValueError("Invalid EOS IDs")

    if is_devstral24:
        tokenizer_vocab_size = tokenizer.vocab_size
    else:
        tokenizer_vocab_size = len(tokenizer)

    if tokenizer_vocab_size > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary exceeds model embeddings")

    context = int(text_config.max_position_embeddings)
    tokenizer_context = getattr(tokenizer, "model_max_length", None)

    if (
        isinstance(tokenizer_context, int)
        and 0 < tokenizer_context < 10**9
    ):
        context = min(context, tokenizer_context)

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
        repetition_penalty=(
            1.0
            if args.generated_only_repetition_penalty
            else penalty
        ),
        no_repeat_ngram_size=0,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        eos_token_id=sorted(eos),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        return_dict_in_generate=False,
    )

    kwargs = {"generation_config": generation}
    base = (
        model.get_base_model()
        if hasattr(model, "get_base_model")
        else model
    )

    if "use_model_defaults" in inspect.signature(base.generate).parameters:
        kwargs["use_model_defaults"] = False

    settings = {
        "attention_implementation": attention,
        "model_type": config.model_type,
        "is_deepseek33": is_deepseek33,
        "is_devstral24": is_devstral24,
        "enable_thinking": False if config.model_type == "qwen3_5" else None,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": penalty,
        "generated_only_repetition_penalty": (
            args.generated_only_repetition_penalty
        ),
        "generated_no_repeat_ngram_size": ngram,
        "max_output_tokens": output_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "stop_ids": sorted(eos),
        "adapter_path": (
            str(args.adapter_path) if args.adapter_path else None
        ),
    }

    print(json.dumps(settings, indent=2), flush=True)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": model.get_input_embeddings().weight.device,
        "max_context": context,
        "output_tokens": output_tokens,
        "generation_kwargs": kwargs,
        "settings": settings,
    }


def generate_response_32(runtime, args, encoded, index, stage):
    import torch
    from transformers import (
        LogitsProcessorList,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        StoppingCriteria,
        StoppingCriteriaList,
    )

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
    torch.manual_seed(seed)

    inputs = {
        key: value.to(runtime["device"])
        for key, value in encoded.items()
    }

    kwargs = dict(runtime["generation_kwargs"])
    generation = deepcopy(kwargs["generation_config"])
    generation.max_new_tokens = budget
    kwargs["generation_config"] = generation



    class GeneratedOnlyRepetitionPenalty:
        def __init__(self, penalty):
            self.processor = RepetitionPenaltyLogitsProcessor(penalty)
    
        def __call__(self, input_ids, scores):
            generated_ids = input_ids[:, prompt_tokens:]
    
            if generated_ids.shape[-1] == 0:
                return scores
    
            return self.processor(generated_ids, scores)
    
    
    class GeneratedNGramGuard:
        def __init__(self, size):
            self.processor = NoRepeatNGramLogitsProcessor(size)
    
        def __call__(self, input_ids, scores):
            return self.processor(
                input_ids[:, prompt_tokens:],
                scores,
            )
    
    
    processors = []
    
    penalty = runtime["settings"]["repetition_penalty"]
    
    if (
        runtime["settings"]["generated_only_repetition_penalty"]
        and penalty != 1.0
    ):
        processors.append(
            GeneratedOnlyRepetitionPenalty(penalty)
        )
    
    ngram = runtime["settings"]["generated_no_repeat_ngram_size"]
    
    if ngram:
        processors.append(
            GeneratedNGramGuard(ngram)
        )
    
    if processors:
        kwargs["logits_processor"] = LogitsProcessorList(processors)

    class Deadline(StoppingCriteria):
        expired = False

        def __call__(self, input_ids, scores, **unused):
            self.expired = (
                time.perf_counter() - started
                >= args.max_generation_seconds
            )
            return torch.full(
                (input_ids.shape[0],),
                self.expired,
                dtype=torch.bool,
                device=input_ids.device,
            )

    deadline = Deadline()
    kwargs["stopping_criteria"] = StoppingCriteriaList([deadline])

    started = time.perf_counter()

    with torch.inference_mode():
        output = runtime["model"].generate(
            **inputs,
            **kwargs,
        )

    ids = output[0, prompt_tokens:].detach().cpu().tolist()
    elapsed = time.perf_counter() - started

    eos = set(generation.eos_token_id)
    ended_with_eos = bool(ids and ids[-1] in eos)

    tokenizer = runtime["tokenizer"]

    if runtime["settings"].get("is_devstral24", False):
        raw = tokenizer.decode(ids)
        final = tokenizer.decode(
            ids[:-1] if ended_with_eos else ids
        ).strip()

        if stage == "code":
            match = re.fullmatch(
                r"\s*```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```\s*",
                final,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                final = match.group(1).strip()

    else:
        decode = tokenizer.decode

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
    if runtime["settings"].get("is_deepseek33", False):
        if stage == "code":
            match = re.fullmatch(
                r"\s*```(?:python|py)?[ \t]*\r?\n?(.*?)\r?\n?```\s*",
                final,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                final = match.group(1).strip()

        elif stage == "contracts":
            match = re.fullmatch(
                r"\s*```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\s*",
                final,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                final = match.group(1).strip()

    metadata = {
        "generation_seconds": round(elapsed, 3),
        "seed": seed,
        "generated_tokens": len(ids),
        "ended_with_eos": ended_with_eos,
        "hit_output_limit": (
            len(ids) >= budget and not ended_with_eos
        ),
        "time_budget_exhausted": (
            deadline.expired and not ended_with_eos
        ),
        "tokens_per_second": (
            round(len(ids) / elapsed, 3) if elapsed else None
        ),
    }

    return raw, final, metadata