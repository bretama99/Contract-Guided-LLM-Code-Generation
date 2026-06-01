from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.core import (
    JsonDict,
    as_dict,
    as_list,
    class_name,
    ensure_no_reference_leak,
    extract_class_code,
    method_profiles,
    skeleton,
    task_id,
    text,
    verify_class_code,
)
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.generate_from_contracts import compact_contract
from src.classeval.generate_from_contracts import generation_path as v1_generation_path
from src.classeval.rl_generate_optimized_contracts import optimized_path
from src.classeval.rl_reward import contract_record_status
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

STAGE = "2R_optimized_contract_guided_generation"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contract_guided"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contract_guided.log"
DEFAULT_MAX_TOKENS = 8192

SYSTEM_PROMPT = (
    "You are an expert Python class-level code generation assistant. "
    "Return only one complete, syntactically valid Python source file. "
    "Do not include markdown, explanations, tests, or partial code."
)

logger = logging.getLogger(__name__)


def generation_path(
    task: JsonDict,
    provider: str,
    model: str,
    contract_provider: str | None = None,
    contract_model: str | None = None,
) -> Path:
    base = OUT_DIR / safe_name(provider) / safe_name(model)
    if contract_provider and contract_model and (contract_provider != provider or contract_model != model):
        base = base / "from_contract" / safe_name(contract_provider) / safe_name(contract_model)
    return base / f"{safe_name(task_id(task))}.json"


def safe_class_name(task: JsonDict) -> str:
    return text(task.get("class_name")) or text(task.get("entry_point")) or class_name(task)


def classify_generation_error(error: Any) -> tuple[str, str]:
    detail = str(error).strip()
    lower = detail.lower()

    if not detail:
        return "optimized_generation_failed", ""
    if "missing optimized contract file" in lower:
        return "contract_missing", detail
    if "not usable for code generation" in lower:
        return "contract_unusable_or_fallback", detail
    if "has no contract object" in lower:
        return "contract_missing_object", detail
    if "incomplete method bodies" in lower or "missing methods" in lower:
        return "missing method", detail
    if "changed method signatures" in lower or "staticmethod" in lower:
        return "signature error", detail
    if "invalid python source" in lower or "syntaxerror" in lower or "invalid syntax" in lower:
        return "syntax error", detail
    if "modulenotfounderror" in lower or "no module named" in lower:
        return "import error", detail
    if "timed out" in lower or "timeout" in lower:
        return "timeout", detail
    if "rate limit" in lower:
        return "provider_rate_limit", detail
    if "authentication" in lower or "api key" in lower:
        return "provider_authentication_error", detail
    if "connection" in lower or "network" in lower:
        return "provider_connection_error", detail

    return "optimized_generation_failed", detail


def task_imports(task: JsonDict) -> list[str]:
    imports = task.get("import_statement")

    if isinstance(imports, list):
        return [text(item) for item in imports if text(item)]
    if isinstance(imports, str):
        return [line.strip() for line in imports.splitlines() if line.strip()]

    return []


def strip_leaky_content(value: Any) -> Any:
    blocked_keys = (
        "test",
        "tests",
        "unit_test",
        "solution",
        "reference",
        "ground_truth",
        "oracle",
        "stdout",
        "stderr",
        "traceback",
        "raw_response",
        "prompt",
        "api_result",
    )

    if isinstance(value, dict):
        return {
            str(key): strip_leaky_content(child)
            for key, child in value.items()
            if not any(blocked in str(key).lower() for blocked in blocked_keys)
        }

    if isinstance(value, list):
        return [clean for item in value if (clean := strip_leaky_content(item)) not in ("", None, [], {})]

    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in ("self.assert", "unittest", "def test_", "traceback", "candidate.py")):
            return ""
        return value[:1800]

    return value


def visible_method_examples(description: str, limit: int = 8) -> list[str]:
    lines = text(description).replace('"""', "").replace("'''", "").splitlines()
    blocked = ("self.assert", "unittest", "def test_", "candidate.py", "traceback")

    examples: list[str] = []
    capture_next = False

    for raw in lines:
        line = raw.strip()
        lowered = line.lower()

        if not line or any(marker in lowered for marker in blocked):
            capture_next = False
            continue

        if line.startswith(">>>"):
            examples.append(line[:300])
            capture_next = True
        elif capture_next and not line.startswith(("def ", ":param", ":return:")):
            examples.append(line[:300])
            capture_next = False
        else:
            capture_next = False

        if len(examples) >= limit:
            break

    return examples


def method_descriptions(task: JsonDict) -> list[JsonDict]:
    blocked = ("test", "assert", "unittest", "expected", "candidate.py", "traceback")
    result: list[JsonDict] = []

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue

        name = text(info.get("method_name"))
        description = text(info.get("method_description"))

        lines: list[str] = []
        for line in description.replace('"""', "").replace("'''", "").splitlines():
            line = line.strip()
            lowered = line.lower()

            if line and not line.startswith(("def ", ">>>")) and not any(marker in lowered for marker in blocked):
                lines.append(line)

        if name:
            result.append(
                {
                    "method": name,
                    "description": " ".join(lines)[:1200],
                    "visible_examples": visible_method_examples(description),
                }
            )

    return result


def required_api(task: JsonDict) -> JsonDict:
    profiles = method_profiles(skeleton(task), safe_class_name(task))
    return {
        "class_name": safe_class_name(task),
        "methods": [
            {
                "method_name": name,
                "signature_profile": profile[0],
                "is_static": profile[1],
            }
            for name, profile in profiles.items()
        ],
    }


def load_v1_generation(task: JsonDict, provider: str, model: str) -> tuple[str, str | None]:
    path = v1_generation_path(task, provider, model)

    if not path.exists():
        return "", None

    record = load_json(path)
    if not isinstance(record, dict):
        return "", str(path)

    code = text(record.get("generated_code"))
    if record.get("status") == "success" and code:
        return code, str(path)

    return "", str(path)


def load_contract_bundle(task: JsonDict, provider: str, model: str) -> tuple[JsonDict, JsonDict]:
    path = optimized_path(task, provider, model)

    if not path.exists():
        raise FileNotFoundError(f"Missing optimized contract file: {path}")

    record = load_json(path)
    status = contract_record_status(record)

    if status.get("usable_for_optimized_metrics") is not True:
        raise RuntimeError(
            f"Optimized contract is not usable for code generation: "
            f"path={path} status={status.get('status')} fallback={status.get('fallback')} error={status.get('error')}"
        )

    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Optimized contract record has no contract object: {path}")

    return compact_contract(contract), {
        "contract_path": str(path),
        "contract_status": record.get("status"),
        "contract_type": record.get("contract_type") or "rl_optimized",
        "contract_version": record.get("contract_version") or "v2_rl_optimized",
        "fallback_to_previous_contract": record.get("fallback_to_previous_contract") is True,
        "excluded_from_optimized_metrics": record.get("excluded_from_optimized_metrics") is True,
        "rl_policy": record.get("rl_policy"),
    }


def previous_feedback(
    task: JsonDict,
    provider: str,
    model: str,
    contract_provider: str | None = None,
    contract_model: str | None = None,
) -> JsonDict:
    from src.classeval.evaluate_contract_guided_optimized import evaluation_path

    path = evaluation_path(task, provider, model, contract_provider, contract_model)

    if not path.exists():
        return {}

    record = load_json(path)
    feedback = as_dict(record.get("feedback_for_next_contract"))
    keys = ("passed", "failure_type", "tests_passed", "tests_total", "tests_failed", "failures", "errors", "skipped")

    return {key: feedback.get(key) for key in keys if feedback.get(key) not in (None, "", [], {})}


def build_prompt(payload: JsonDict) -> str:
    return "\n".join(
        [
            "Generate exactly ONE complete Python class implementation for ClassEval.",
            "Return only Python source code. No markdown. No explanations. No tests.",
            "",
            "ABSOLUTE SOURCE OF TRUTH:",
            "The skeleton and REQUIRED_API define the exact class name, imports, constructor, fields, decorators, and method signatures.",
            "The optimized contract is only guidance and must never override the skeleton.",
            "",
            "HARD REJECTION RULES:",
            "- Every required method must exist.",
            "- Every required method must have executable implementation code.",
            "- Do not use pass, ellipsis, TODO, NotImplementedError, or empty method bodies.",
            "- Do not omit __init__ when it exists in the skeleton.",
            "- Do not change @staticmethod.",
            "- Do not rename, remove, reorder, or add methods or parameters.",
            "- Do not add tests, unittest code, self.assert, examples as code, or markdown fences.",
            "- Do not include reference/canonical solutions, hidden tests, or test answers.",
            "",
            "BEHAVIOR RULES:",
            "- Start from the v1 generated implementation when available.",
            "- Preserve correct v1 behavior.",
            "- Modify behavior only when the visible task or optimized contract clearly supports the change.",
            "- Prefer simple deterministic Python and standard-library-only code.",
            "- Preserve class state fields initialized by the constructor.",
            "",
            "INPUT:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def make_prompt(task: JsonDict, contract: JsonDict, feedback: JsonDict, v1_code: str, v1_code_path: str | None, rl_policy: Any) -> str:
    payload = {
        "required_api": required_api(task),
        "class_name": safe_class_name(task),
        "skeleton": skeleton(task),
        "imports": task_imports(task),
        "class_constructor": text(task.get("class_constructor"))[:1200],
        "class_description": text(task.get("class_description")).replace('"""', "").replace("'''", "")[:1800],
        "fields": as_list(task.get("fields")),
        "method_descriptions": method_descriptions(task),
        "v1_generated_code_path": v1_code_path,
        "v1_generated_code": v1_code[:12000],
        "optimized_contract": strip_leaky_content(contract),
        "selected_rl_action": strip_leaky_content(rl_policy),
        "previous_optimized_evaluation_feedback": strip_leaky_content(feedback),
    }

    prompt = build_prompt(payload)
    ensure_no_reference_leak(task, prompt)
    return prompt


def call_model(client: Any, model: str, prompt: str, task: JsonDict, args: argparse.Namespace) -> tuple[str, str, JsonDict | None]:
    response, api_result = call_chat_model(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        json_mode=False,
    )

    code = extract_class_code(response, task)
    verify_class_code(code, task, check_methods=True)

    return code, response, api_result


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "class_name": safe_class_name(task),
        "entry_point": safe_class_name(task),
        "stage": STAGE,
        "provider": args.provider,
        "model_name": model,
        "contract_provider": args.contract_provider or args.provider,
        "contract_model": args.contract_model or model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any) -> JsonDict:
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    contract, contract_meta = load_contract_bundle(task, contract_provider, contract_model)
    v1_code, v1_code_path = load_v1_generation(task, base_provider, base_model)
    feedback = previous_feedback(task, args.provider, model, contract_provider, contract_model) if args.use_previous_eval_feedback else {}

    prompt = make_prompt(task, contract, feedback, v1_code, v1_code_path, contract_meta.get("rl_policy"))
    code, response, api_result = call_model(client, model, prompt, task, args)

    record = make_record(
        task,
        args,
        model,
        status="success",
        failure_type=None,
        failure_detail=None,
        contract=contract,
        **contract_meta,
        v1_generated_code_path=v1_code_path,
        used_v1_generated_code=bool(v1_code),
        generated_code=code,
        copied_from_v1=False,
        error=None,
    )

    if args.debug:
        record.update({"prompt": prompt, "raw_response": response, "api_result": api_result})

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval solutions from optimized contracts.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-provider")
    parser.add_argument("--contract-model")
    parser.add_argument("--base-provider")
    parser.add_argument("--base-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-previous-eval-feedback", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = 0
    failed = 0
    skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = generation_path(task, args.provider, model, contract_provider, contract_model)

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        try:
            record = generate_one(task, args, model, client)
        except Exception as exc:
            logger.exception("Optimized-contract-guided generation failed: %s", task_id(task))
            failure_type, failure_detail = classify_generation_error(exc)
            record = make_record(
                task,
                args,
                model,
                status="failed",
                failure_type=failure_type,
                failure_detail=failure_detail,
                contract=None,
                contract_path=str(optimized_path(task, contract_provider, contract_model)),
                contract_status=None,
                contract_type="rl_optimized",
                contract_version="v2_rl_optimized",
                generated_code=None,
                copied_from_v1=False,
                error=failure_detail,
            )

        save_json(path, record)

        if record.get("status") == "success":
            done += 1
            print(f"[{index}] DONE {task_id(task)}")
        else:
            failed += 1
            print(f"[{index}] FAILED/{record.get('failure_type')} {task_id(task)}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval optimized-contract-guided generation finished.")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()