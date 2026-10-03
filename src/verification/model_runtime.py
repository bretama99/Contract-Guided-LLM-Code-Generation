from __future__ import annotations

import json
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
    def __init__(
        self,
        model_path: Path,
        adapter_path: Path | None,
        gpu_id: int,
        attention_implementation: str | None = None,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        import torch
        import transformers.utils.import_utils as import_utils
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = (
                lambda: hasattr(torch, "fx")
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self.torch = torch
        self.model_path = Path(model_path)
        self.adapter_path = (
            Path(adapter_path)
            if adapter_path is not None
            else None
        )

        common = {
            "local_files_only": True,
            "trust_remote_code": True,
        }

        raw_model_type = ""
        config_json = self.model_path / "config.json"

        if config_json.is_file():
            try:
                raw_model_type = str(
                    json.loads(
                        config_json.read_text(encoding="utf-8")
                    ).get("model_type", "")
                ).lower()
            except Exception:
                raw_model_type = ""

        config = AutoConfig.from_pretrained(
            str(self.model_path),
            **common,
        )

        self.model_type = str(
            getattr(config, "model_type", raw_model_type)
        ).lower()

        if self.model_type.startswith("deepseek"):
            self._patch_deepseek_cache()

        if self.model_type == "mistral3":
            config.tie_word_embeddings = False
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                text_config.tie_word_embeddings = False

        tokenizer_args = dict(common)

        if self.model_type == "mistral3":
            tokenizer_args["mode"] = "test"

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            **tokenizer_args,
        )

        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attention = attention_implementation or "auto"

        if self.model_type.startswith("deepseek"):
            attention = "eager"
        elif self.model_type == "mistral3" and attention == "auto":
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

        if self.model_type == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **model_args,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_args,
            )

        if self.model_type.startswith("deepseek"):
            self._patch_deepseek_generation(model)

        if self.adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(self.adapter_path.resolve()),
                is_trainable=False,
                local_files_only=True,
            )

            if self.model_type == "mistral3":
                model = model.merge_and_unload()

        self.model = model.eval()
        self.model.config.use_cache = True

        text_config = getattr(self.model.config, "text_config", None)

        if text_config is not None:
            text_config.use_cache = True

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
        self.context_length = self._context_length(config)
        self.stop_ids = self._stop_ids()

        print(
            f"model_type={self.model_type} "
            f"attention={self.attention} "
            f"use_cache=True "
            f"context={self.context_length} "
            f"adapter={self.adapter_path is not None}",
            flush=True,
        )

    @staticmethod
    def _context_length(config: Any) -> int:
        for cfg in (
            config,
            getattr(config, "text_config", None),
        ):
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

    def _stop_ids(self) -> list[int]:
        ids: set[int] = set()

        def add(value: Any) -> None:
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, (list, tuple, set)):
                ids.update(
                    item
                    for item in value
                    if isinstance(item, int)
                )

        add(self.tokenizer.eos_token_id)

        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )

        if generation_config is not None:
            add(generation_config.eos_token_id)

        family_token = None

        if self.model_type.startswith("qwen"):
            family_token = "<|im_end|>"
        elif self.model_type == "llama":
            family_token = "<|eot_id|>"
        elif self.model_type == "mistral3":
            family_token = "<|end_of_turn|>"

        if family_token:
            token_id = self.tokenizer.convert_tokens_to_ids(
                family_token
            )

            if (
                isinstance(token_id, int)
                and (
                    self.tokenizer.unk_token_id is None
                    or token_id != self.tokenizer.unk_token_id
                )
            ):
                ids.add(token_id)

        if not ids:
            raise ValueError("No valid stop token found")

        return sorted(ids)

    @staticmethod
    def _patch_deepseek_cache() -> None:
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

    @staticmethod
    def _patch_deepseek_generation(model: Any) -> None:
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

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        max_context: int,
        max_time: float | None = None,
    ) -> str:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
        )

        deadline = (
            time.perf_counter() + float(max_time)
            if max_time is not None
            else None
        )

        if max_time is not None and max_time <= 0:
            raise TimeoutError("Generation time budget exhausted")

        encoded = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": str(user)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        encoded = dict(encoded)

        if "input_ids" not in encoded:
            raise ValueError("Chat template returned no input_ids")

        prompt_tokens = int(
            encoded["input_ids"].shape[-1]
        )

        effective_context = min(
            int(max_context),
            int(self.context_length),
        )

        if prompt_tokens + max_new_tokens > effective_context:
            raise ValueError(
                f"prompt={prompt_tokens}, "
                f"output={max_new_tokens}, "
                f"context={effective_context}"
            )

        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Generation time budget exhausted")

        device = next(self.model.parameters()).device

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        tokenizer = self.tokenizer
        prompt_width = prompt_tokens

        class CompleteJsonStoppingCriteria(StoppingCriteria):
            def __init__(self):
                self.last_check = 0

            def __call__(
                self,
                input_ids,
                scores,
                **kwargs,
            ):
                generated_len = (
                    input_ids.shape[-1] - prompt_width
                )

                if generated_len <= 0:
                    return False

                if (
                    generated_len < max_new_tokens
                    and generated_len - self.last_check < 16
                ):
                    return False

                self.last_check = generated_len

                generated = tokenizer.decode(
                    input_ids[0, prompt_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                return _complete_json_object(generated)

        generation: dict[str, Any] = {
            **encoded,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": (
                self.stop_ids[0]
                if len(self.stop_ids) == 1
                else self.stop_ids
            ),
            "stopping_criteria": StoppingCriteriaList(
                [CompleteJsonStoppingCriteria()]
            ),
        }

        if deadline is not None:
            remaining = deadline - time.perf_counter()

            if remaining <= 0:
                raise TimeoutError(
                    "Generation time budget exhausted"
                )

            generation["max_time"] = max(
                0.1,
                remaining,
            )

        if self.model_type == "mistral3":
            generation["logits_to_keep"] = 1

        with self.torch.inference_mode():
            output = self.model.generate(**generation)

        ids = output[0, prompt_width:]

        text = self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if (
            deadline is not None
            and time.perf_counter() >= deadline
            and not _complete_json_object(text)
        ):
            raise TimeoutError(
                "Generation time budget exhausted"
            )

        return text