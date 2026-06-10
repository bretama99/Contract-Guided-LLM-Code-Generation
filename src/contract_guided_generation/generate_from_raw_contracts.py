#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json, load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import add_missing_common_alias_imports, extract_python_code
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_path
from src.common.task_utils import select_tasks, task_entry_point, task_identifier, task_prompt
from src.rl_method_level.rl_generate_contract import optimized_contract_code_path, optimized_contract_path

RAW_STAGE = "stage_2b_raw_contract_guided_generation"
OPT_STAGE = "stage_2e_optimized_rl_contract_guided_generation"
LOG_FILE = LOG_ROOT / "stage2_contract_guided_generation.log"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DELAY = 0.5
CONTRACT_SOURCES = ("raw", "optimized_rl")

SYSTEM_PROMPT = "Generate a complete executable Python benchmark solution. Return only Python source code."
PROMPT_TEMPLATE = """
You are generating a Python solution for a benchmark programming task.

The original benchmark prompt is the source of truth. The selected contract is a helper checklist.

Rules:
- Return only Python code: no markdown, explanations, or tests.
- Preserve the required function name and visible signature exactly.
- Include needed imports and helper functions.
- Solve the general task, not only examples.
- Do not copy contract clauses as runtime assertions.
- Do not add validation/exceptions unless the prompt explicitly requires them.
- If contract and prompt conflict, follow the prompt.
- For runtime-sensitive tasks, handle prompt-supported empty inputs, variable-length inputs, and boundary cases.
- For timeout-sensitive tasks, avoid unbounded brute force when the prompt implies a direct or bounded algorithm.

Contract source: {contract_source}
Entry point: {entry_point}

Original benchmark prompt:
{prompt}

Contract metadata:
{metadata}

Selected contract:
{contract}
""".strip()
logger = logging.getLogger(__name__)


def require_source(value: str) -> str:
    if value not in CONTRACT_SOURCES:
        raise ValueError(f"Unsupported contract source: {value}")
    return value


def stage_for(source: str) -> str:
    return RAW_STAGE if require_source(source) == "raw" else OPT_STAGE


def contract_type(source: str) -> str:
    return "raw" if require_source(source) == "raw" else "optimized_rl"


def contract_version(source: str) -> str:
    return "stage2_raw_contract_schema_v1" if require_source(source) == "raw" else "stage2_rl_optimized_contract_schema_v1"


def contract_path(source: str, dataset: str, task_id: str, provider: str, model: str) -> Path:
    return raw_contract_path(dataset, task_id, provider, model) if require_source(source) == "raw" else optimized_contract_path(dataset, task_id, provider, model)


def output_path(source: str, dataset: str, task_id: str, provider: str, model: str) -> Path:
    return raw_contract_code_path(dataset, task_id, provider, model) if require_source(source) == "raw" else optimized_contract_code_path(dataset, task_id, provider, model)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []

def extract_contract_payload(record: Any, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = require_source(source)
    label = contract_type(source)
    if not isinstance(record, dict):
        raise ValueError(f"{label} contract file must contain a JSON object.")

    status = record.get("status")
    if status not in (None, "success"):
        # Backward compatibility for old artifacts that wrongly rejected identical raw contracts.
        if source == "optimized_rl" and isinstance(record.get("candidate_contract"), dict) and "identical" in str(record.get("error", "")).lower():
            return record["candidate_contract"], {"optimization_status": "legacy_unchanged", "fallback_used": True}
        raise RuntimeError(f"{label} contract status is not success: {record.get('error')}")

    contract = record.get("contract", record)
    if not isinstance(contract, dict):
        raise ValueError(f"{label} contract record does not contain a contract object.")

    meta = {}
    if source == "optimized_rl":
        for key in (
            "optimized", "optimization_status", "optimization_changed", "optimized_success", "fallback_used",
            "issue", "reason", "solution", "selected_action", "raw_contract_path", "contract_hash",
        ):
            if key in record:
                meta[key] = record.get(key)
        meta["rl"] = record.get("rl") if isinstance(record.get("rl"), dict) else None
    return contract, meta


def compact_items(items: Any, keep: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    rows = []
    for item in as_list(items):
        if isinstance(item, dict):
            row = {k: item.get(k) for k in keep if item.get(k) not in (None, "", [], {})}
            if row:
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    task = contract.get("task") if isinstance(contract.get("task"), dict) else {}
    interface = contract.get("interface") if isinstance(contract.get("interface"), dict) else {}
    return {
        "task_summary": task.get("summary", ""),
        "signature": task.get("signature", ""),
        "imports_required": task.get("imports_required", []),
        "helper_functions_required": task.get("helper_functions_required", []),
        "interface": interface,
        "preconditions": compact_items(contract.get("preconditions"), ("target", "kind", "description", "source"), 8),
        "postconditions": compact_items(contract.get("postconditions"), ("target", "kind", "description", "source"), 12),
        "invariants": compact_items(contract.get("invariants"), ("target", "description", "source"), 8),
        "edge_cases": compact_items(contract.get("edge_cases"), ("case", "expected_behavior", "source"), 12),
        "invalid_input_behavior": contract.get("invalid_input_behavior", {}),
    }


def build_prompt(task: dict[str, Any], contract: dict[str, Any], source: str, metadata: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        contract_source=contract_type(source),
        entry_point=task_entry_point(task),
        prompt=task_prompt(task),
        metadata=json.dumps(metadata, indent=2, ensure_ascii=False),
        contract=json.dumps(compact_contract(contract), indent=2, ensure_ascii=False),
    )


def generated_record(
    *, task: dict[str, Any], dataset: str, benchmark: str, provider: str, model: str,
    source: str, metadata: dict[str, Any], temperature: float, max_tokens: int,
    status: str, contract_file: Path, prompt: str, raw_response: str,
    code: str | None, api_result: dict[str, Any] | None, error: str | None,
    reused_raw_generation: bool = False, raw_generation_path: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task), "benchmark": benchmark, "dataset": dataset,
        "entry_point": task_entry_point(task), "stage": stage_for(source), "status": status,
        "provider": provider, "model_name": model, "temperature": temperature, "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(), "contract_path": str(contract_file),
        "contract_type": contract_type(source), "contract_version": contract_version(source),
        "contract_metadata": metadata,
        "contract_optimized": metadata.get("optimized"),
        "contract_optimization_status": metadata.get("optimization_status"),
        "contract_optimization_changed": metadata.get("optimization_changed"),
        "contract_optimized_success": metadata.get("optimized_success"),
        "contract_fallback_used": metadata.get("fallback_used"),
        "contract_issue": metadata.get("issue"), "contract_reason": metadata.get("reason"), "contract_solution": metadata.get("solution"),
        "reused_raw_generation": reused_raw_generation, "raw_generation_path": raw_generation_path,
        "generation_prompt": prompt, "raw_response": raw_response, "generated_code": code,
        "api_result": api_result, "api_latency_seconds": api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        "error": error, "source": task.get("source"), "source_version": task.get("source_version"),
    }


def extract_solution(raw_response: str, entry_point: str) -> str:
    code = extract_python_code(raw_response, entry_point=entry_point, validate=True)
    code = add_missing_common_alias_imports(code)
    return extract_python_code(code, entry_point=entry_point, validate=True)

def reuse_raw_if_allowed(task: dict[str, Any], dataset: str, provider: str, model: str, metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    if metadata.get("optimization_status") not in {"preserved_raw_passed", "keep_raw_if_passed", "kept_good_raw_contract", "no_optimized_candidate"}:
        return None, None
    path = raw_contract_code_path(dataset, task_identifier(task), provider, model)
    if not path.exists():
        return None, None
    record = load_json(path)
    if isinstance(record, dict) and record.get("status") == "success" and isinstance(record.get("generated_code"), str):
        return record["generated_code"], str(path)
    return None, None

def generate_one(
    *, task: dict[str, Any], dataset: str, benchmark: str, provider: str, model: str,
    source: str, client: Any, temperature: float, max_tokens: int,
) -> dict[str, Any]:
    tid = task_identifier(task)
    cpath = contract_path(source, dataset, tid, provider, model)
    prompt = ""; raw_response = ""; api_result = None; code = None; metadata: dict[str, Any] = {}
    try:
        if not cpath.exists():
            raise FileNotFoundError(f"Missing {contract_type(source)} contract file: {cpath}")
        contract, metadata = extract_contract_payload(load_json(cpath), source)

        if source == "optimized_rl":
            code, raw_path = reuse_raw_if_allowed(task, dataset, provider, model, metadata)
            if code is not None:
                return generated_record(task=task, dataset=dataset, benchmark=benchmark, provider=provider, model=model,
                                        source=source, metadata=metadata, temperature=temperature, max_tokens=max_tokens,
                                        status="success", contract_file=cpath,
                                        prompt="Reused raw contract-guided code because optimized contract preserved/fell back to raw.",
                                        raw_response="", code=code, api_result=None, error=None,
                                        reused_raw_generation=True, raw_generation_path=raw_path)

        prompt = build_prompt(task, contract, source, metadata)
        raw_response, api_result = call_chat_model(
            client=client, model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, json_mode=False,
        )
        code = extract_solution(raw_response, task_entry_point(task))
        return generated_record(task=task, dataset=dataset, benchmark=benchmark, provider=provider, model=model,
                                source=source, metadata=metadata, temperature=temperature, max_tokens=max_tokens,
                                status="success", contract_file=cpath, prompt=prompt, raw_response=raw_response,
                                code=code, api_result=api_result, error=None)
    except Exception as exc:
        logger.exception("Stage 2 contract-guided generation failed for %s using %s", tid, source)
        return generated_record(task=task, dataset=dataset, benchmark=benchmark, provider=provider, model=model,
                                source=source, metadata=metadata, temperature=temperature, max_tokens=max_tokens,
                                status="failed", contract_file=cpath, prompt=prompt, raw_response=raw_response,
                                code=code, api_result=api_result, error=str(exc))


def run(args: argparse.Namespace) -> None:
    info = DATASETS[args.dataset]
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(info["path"]), start=args.start, count=args.count)
    counts = {"completed": 0, "failed": 0, "skipped": 0, "reused": 0}
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=args.start):
        tid = task_identifier(task)
        out = output_path(args.contract_source, args.dataset, tid, args.provider, model)
        if out.exists() and not args.overwrite:
            counts["skipped"] += 1; print(f"[{index}] SKIP {tid}"); continue
        rec = generate_one(task=task, dataset=args.dataset, benchmark=info.get("label", args.dataset),
                           provider=args.provider, model=model, source=args.contract_source,
                           client=client, temperature=args.temperature, max_tokens=args.max_tokens)
        save_json(out, rec)
        ok = rec.get("status") == "success"
        counts["completed" if ok else "failed"] += 1
        if rec.get("reused_raw_generation"):
            counts["reused"] += 1
        suffix = " — reused raw code" if rec.get("reused_raw_generation") else ("" if ok else f": {rec.get('error')}")
        print(f"[{index}] {'DONE' if ok else 'FAIL'} {tid} [{contract_type(args.contract_source)}]{suffix}")
        if args.delay > 0:
            time.sleep(args.delay)
    print(f"\nStage 2 contract-guided generation finished [{contract_type(args.contract_source)}]")
    print(json.dumps({**counts, "elapsed": round(time.perf_counter() - started, 4)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2B/2E: generate code from raw or optimized contracts")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-source", choices=CONTRACT_SOURCES, default="raw")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging(LOG_FILE)
    run(parse_args())


if __name__ == "__main__":
    main()
