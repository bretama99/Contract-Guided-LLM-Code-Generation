from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from typing import Any


def _complete_json_object(text: str) -> bool:
    started = False
    depth = 0
    in_string = False
    escaped = False

    for char in str(text or ""):
        if not started:
            if char == "{":
                started = True
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return True

    return False


class LocalModel:
    """Greedy SFT+LoRA validation runtime for Qwen2, Qwen3.5 and Mistral3."""

    def __init__(
        self,
        model_path: Path,
        adapter_path: Path,
        gpu_id: int,
        attention_implementation: str = "sdpa",
        *,
        load_in_4bit: bool = False,
        max_memory_per_gpu: str = "42GiB",
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self.torch = torch
        self.model_path = Path(
            model_path
        ).expanduser().resolve()

        self.adapter_path = Path(
            adapter_path
        ).expanduser().resolve()

        self.load_in_4bit = bool(
            load_in_4bit
        )

        self.max_memory_per_gpu = str(
            max_memory_per_gpu
        )

        common = {
            "local_files_only": True,
            "trust_remote_code": True,
        }

        config = AutoConfig.from_pretrained(
            str(self.model_path),
            **common,
        )

        self.model_type = str(
            getattr(config, "model_type", "")
        ).lower()

        if self.model_type not in {
            "qwen2",
            "qwen3_5",
            "mistral3",
        }:
            raise ValueError(
                f"Unsupported validation model type: "
                f"{self.model_type!r}. "
                "Expected qwen2, qwen3_5, or mistral3."
            )

        if self.model_type == "mistral3":
            config.tie_word_embeddings = False

            if (
                getattr(
                    config,
                    "text_config",
                    None,
                )
                is not None
            ):
                config.text_config.tie_word_embeddings = False

        self.processor = None

        if self.model_type == "mistral3":
            self.processor = (
                AutoProcessor.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
            )

            self.tokenizer = (
                self.processor.tokenizer
            )

        else:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    **common,
                )
            )

        if (
            self.tokenizer.eos_token_id
            is None
        ):
            raise ValueError(
                "Tokenizer has no EOS token"
            )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        attention = (
            attention_implementation
            or "sdpa"
        )

        if attention == "auto":
            attention = "sdpa"

        model_args: dict[str, Any] = {
            **common,
            "config": config,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
            "max_memory": {
                0: self.max_memory_per_gpu
            },
            "low_cpu_mem_usage": True,
            "attn_implementation": attention,
        }

        if self.load_in_4bit:
            if getattr(
                config,
                "quantization_config",
                None,
            ):
                raise ValueError(
                    "--load-in-4bit requires an "
                    "unquantized checkpoint. "
                    "For Devstral use the BF16 "
                    "dequantized copy, not the "
                    "native-FP8 directory."
                )

            model_args[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                ),
            )

        model_class = (
            AutoModelForCausalLM
        )

        if self.model_type == "qwen3_5":
            from transformers import (
                Qwen3_5ForConditionalGeneration,
            )

            model_class = (
                Qwen3_5ForConditionalGeneration
            )

        elif self.model_type == "mistral3":
            from transformers import (
                Mistral3ForConditionalGeneration,
            )

            model_class = (
                Mistral3ForConditionalGeneration
            )

        model = model_class.from_pretrained(
            str(self.model_path),
            **model_args,
        )

        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(self.adapter_path),
            is_trainable=False,
            local_files_only=True,
        )

        # Keep the SFT LoRA active throughout:
        # verification -> repair -> re-verification.
        self.model = model.eval()

        self.model.config.use_cache = True

        if (
            getattr(
                self.model.config,
                "text_config",
                None,
            )
            is not None
        ):
            self.model.config.text_config.use_cache = True

        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )

        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        self.attention = attention

        self.context_length = (
            self._context_length(
                config
            )
        )

        self.stop_ids = (
            self._stop_ids(
                config
            )
        )

        print(
            f"model_type={self.model_type} "
            f"attention={self.attention} "
            f"4bit={self.load_in_4bit} "
            f"context={self.context_length} "
            f"adapter=True",
            flush=True,
        )

    @staticmethod
    def _context_length(
        config: Any,
    ) -> int:
        values = []

        for source in (
            config,
            getattr(
                config,
                "text_config",
                None,
            ),
        ):
            if source is None:
                continue

            value = getattr(
                source,
                "max_position_embeddings",
                None,
            )

            if (
                isinstance(value, int)
                and 0 < value < 10**9
            ):
                values.append(value)

        return (
            min(values)
            if values
            else 8192
        )

    def _stop_ids(
        self,
        config: Any,
    ) -> list[int]:
        ids: set[int] = set()

        def add(
            value: Any,
        ) -> None:
            if isinstance(
                value,
                int,
            ):
                ids.add(value)

            elif isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                ids.update(
                    item
                    for item in value
                    if isinstance(
                        item,
                        int,
                    )
                )

        add(
            self.tokenizer.eos_token_id
        )

        add(
            getattr(
                config,
                "eos_token_id",
                None,
            )
        )

        text_config = getattr(
            config,
            "text_config",
            None,
        )

        if text_config is not None:
            add(
                getattr(
                    text_config,
                    "eos_token_id",
                    None,
                )
            )

        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )

        if generation_config is not None:
            add(
                generation_config.eos_token_id
            )

        special = (
            "<|end_of_turn|>"
            if self.model_type
            == "mistral3"
            else "<|im_end|>"
        )

        token_id = (
            self.tokenizer
            .convert_tokens_to_ids(
                special
            )
        )

        if (
            isinstance(
                token_id,
                int,
            )
            and token_id >= 0
            and (
                self.tokenizer.unk_token_id
                is None
                or token_id
                != self.tokenizer.unk_token_id
            )
        ):
            ids.add(token_id)

        vocab_size = (
            self.model
            .get_output_embeddings()
            .weight
            .shape[0]
        )

        ids = {
            token_id
            for token_id in ids
            if 0
            <= token_id
            < vocab_size
        }

        if not ids:
            raise ValueError(
                "No valid EOS/stop token found"
            )

        return sorted(ids)

    def _encode(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": str(
                    system_prompt
                ),
            },
            {
                "role": "user",
                "content": str(
                    user_prompt
                ),
            },
        ]

        if self.processor is not None:
            prompt = (
                self.processor
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        else:
            chat_args: dict[
                str,
                Any,
            ] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            if (
                self.model_type
                == "qwen3_5"
            ):
                chat_args[
                    "enable_thinking"
                ] = False

            prompt = (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    **chat_args,
                )
            )

        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
        )

        encoded = dict(
            encoded
        )

        encoded.pop(
            "token_type_ids",
            None,
        )

        if (
            "input_ids"
            not in encoded
        ):
            raise ValueError(
                "Chat template returned "
                "no input_ids"
            )

        return encoded

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        max_context: int,
        max_time: float | None = None,
    ) -> str:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
        )

        if (
            max_time is not None
            and max_time <= 0
        ):
            raise TimeoutError(
                "Generation time budget exhausted"
            )

        deadline = (
            time.perf_counter()
            + float(max_time)
            if max_time is not None
            else None
        )

        encoded = self._encode(
            system_prompt,
            user_prompt,
        )

        prompt_tokens = int(
            encoded[
                "input_ids"
            ].shape[-1]
        )

        effective_context = min(
            int(max_context),
            int(
                self.context_length
            ),
        )

        if (
            prompt_tokens
            + int(max_new_tokens)
            > effective_context
        ):
            raise ValueError(
                f"prompt={prompt_tokens}, "
                f"output={max_new_tokens}, "
                f"context={effective_context}"
            )

        device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        encoded = {
            key: value.to(
                device
            )
            for key, value
            in encoded.items()
        }

        tokenizer = (
            self.tokenizer
        )

        prompt_width = (
            prompt_tokens
        )

        class CompleteJsonStoppingCriteria(
            StoppingCriteria
        ):
            def __init__(
                self,
            ) -> None:
                self.last_check = 0

            def __call__(
                self,
                input_ids,
                scores,
                **unused,
            ):
                generated_len = (
                    input_ids.shape[-1]
                    - prompt_width
                )

                if generated_len <= 0:
                    return False

                if (
                    generated_len
                    < max_new_tokens
                    and generated_len
                    - self.last_check
                    < 16
                ):
                    return False

                self.last_check = (
                    generated_len
                )

                generated = (
                    tokenizer.decode(
                        input_ids[
                            0,
                            prompt_width:
                        ],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )

                return (
                    _complete_json_object(
                        generated
                    )
                )

        generation: dict[
            str,
            Any,
        ] = {
            **encoded,
            "max_new_tokens": int(
                max_new_tokens
            ),
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": (
                self.tokenizer
                .pad_token_id
            ),
            "eos_token_id": (
                self.stop_ids[0]
                if len(
                    self.stop_ids
                )
                == 1
                else self.stop_ids
            ),
            "stopping_criteria": (
                StoppingCriteriaList(
                    [
                        CompleteJsonStoppingCriteria()
                    ]
                )
            ),
        }

        if deadline is not None:
            remaining = (
                deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Generation time "
                    "budget exhausted"
                )

            generation[
                "max_time"
            ] = max(
                0.1,
                remaining,
            )

        base = (
            self.model.get_base_model()
            if hasattr(
                self.model,
                "get_base_model",
            )
            else self.model
        )

        parameters = (
            inspect.signature(
                base.forward
            ).parameters
        )

        if (
            "logits_to_keep"
            in parameters
        ):
            generation[
                "logits_to_keep"
            ] = 1

        elif (
            "num_logits_to_keep"
            in parameters
        ):
            generation[
                "num_logits_to_keep"
            ] = 1

        with (
            self.torch
            .inference_mode()
        ):
            output = (
                self.model.generate(
                    **generation
                )
            )

        ids = output[
            0,
            prompt_width:
        ]

        text = (
            self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            .strip()
        )

        if (
            deadline is not None
            and time.perf_counter()
            >= deadline
            and not
            _complete_json_object(
                text
            )
        ):
            raise TimeoutError(
                "Generation time budget exhausted"
            )

        return text