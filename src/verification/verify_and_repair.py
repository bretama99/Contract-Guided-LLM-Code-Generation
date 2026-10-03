from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.prompts.generation_prompts import CONTRACT_SYSTEM_PROMPT
from src.verification.model_runtime import LocalModel
from src.verification.contract_repair import REPAIR_VERSION, repair_defective_contract
from src.verification.contract_verifier import FINAL_STATUSES, VERIFIER_VERSION, verify_contract

JsonDict = dict[str, Any]
WORKFLOW_VERSION = "three_file_expression_led_verify_repair_v3_actionable_u1"

SELECTIONS = frozenset({
    "PRESERVED_CORRECT",
    "PRESERVED_UNCERTAIN",
    "REPAIRED_CORRECT",
    "REPAIRED_CORRECT_FROM_UNCERTAIN",
    "DEFECTIVE_UNRESOLVED",
    "UNCERTAIN_REPAIR_UNRESOLVED",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify every TACO contract against the original task and repair "
            "DEFECTIVE contracts plus evidence-backed actionable UNCERTAIN contracts. Repaired candidates are "
            "accepted only after fresh CORRECT verification."
        )
    )
    p.add_argument("--task-file", type=Path, required=True)
    p.add_argument("--contract-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path)
    p.add_argument("--model-name", required=True)

    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int)

    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--json-retries", type=int, default=1)
    p.add_argument("--verify-tokens", type=int, default=768)
    p.add_argument("--confirm-tokens", type=int, default=256)
    p.add_argument("--repair-tokens", type=int, default=768)
    p.add_argument("--contract-generation-tokens", type=int, default=1536)
    p.add_argument("--contract-generation-retries", type=int, default=5)
    p.add_argument("--max-context", type=int, default=32768)

    p.add_argument(
        "--attention-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    p.add_argument("--overwrite", action="store_true")

    a = p.parse_args()

    for path, label in (
        (a.task_file, "Task file"),
        (a.model_path, "Model path"),
    ):
        if not path.exists():
            p.error(f"{label} not found: {path}")

    if not a.contract_dir.is_dir():
        p.error(f"Contract directory not found: {a.contract_dir}")
    if a.adapter_path is not None and not a.adapter_path.exists():
        p.error(f"Adapter path not found: {a.adapter_path}")
    if a.gpu_id < 0 or a.start < 0:
        p.error("--gpu-id and --start must be >= 0")
    if a.count is not None and a.count <= 0:
        p.error("--count must be > 0")
    if a.retries < 0 or a.json_retries < 0 or a.contract_generation_retries <= 0:
        p.error("--retries/--json-retries must be >= 0 and --contract-generation-retries must be > 0")
    for name in (
        "verify_tokens",
        "confirm_tokens",
        "repair_tokens",
        "contract_generation_tokens",
        "max_context",
    ):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be > 0")
    return a


def fingerprint(contract: JsonDict) -> str:
    text = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_contracts(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("taco_*.json")):
        try:
            record = read_json(path)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue

        stored = record.get("task_id")
        tid = stored.strip() if isinstance(stored, str) and stored.strip() else path.stem
        resolved = path.resolve()
        previous = indexed.get(tid)
        if previous is not None and previous != resolved:
            raise ValueError(f"Duplicate contract for {tid}: {previous} and {resolved}")
        indexed[tid] = resolved
    return indexed


def load_contract(path: Path) -> tuple[JsonDict, JsonDict | None]:
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError(f"Contract record is not an object: {path}")

    # If the original generation explicitly stored contract=null, do not
    # salvage its malformed raw response. Generate a fresh contract from the
    # original task/question instead.
    if "contract" in record and record.get("contract") is None:
        return record, None

    for value in (
        record.get("contract"),
        record.get("parsed_contract"),
        record,
    ):
        contract = canonical_contract(value)
        if isinstance(contract, dict):
            return record, contract

    return record, None


def build_contract_generation_prompt(task: JsonDict, tid: str) -> str:
    fn = taco_function_name(task)
    parts = [
        f"TASK_ID: {tid}",
        "",
        "QUESTION:",
        taco_question(task),
        "",
    ]
    if fn:
        parts += ["Use Call-Based format.", f"FUNCTION NAME: {fn}"]
    else:
        parts.append("Use Standard Input format.")
    parts += ["", "CONTRACT:"]
    return normalize_prompt_text("\n".join(parts))


def parse_generated_contract(raw: str) -> JsonDict | None:
    """Parse freshly generated contract JSON without changing its semantics."""
    text = str(raw or "").strip()
    if not text:
        return None

    # Remove common Markdown fences.
    fenced = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```\s*$", "", fenced).strip()

    candidates = [text]
    if fenced != text:
        candidates.append(fenced)

    # Fresh model generations often contain regex-style backslashes such as
    # \d, \s, \w. They are valid contract-string content but invalid JSON
    # escapes unless the backslash is doubled. Fix JSON encoding only.
    for candidate in list(candidates):
        escaped = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        if escaped != candidate:
            candidates.append(escaped)

    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            contract = canonical_contract(value)
            if isinstance(contract, dict):
                return contract
        except Exception:
            pass

        # Accept one JSON object embedded in surrounding model text.
        for pos, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[pos:])
            except Exception:
                continue
            contract = canonical_contract(value)
            if isinstance(contract, dict):
                return contract

    return None

def generate_contract_from_question(
    model: LocalModel,
    task: JsonDict,
    *,
    tid: str,
    args: argparse.Namespace,
) -> JsonDict:
    prompt = build_contract_generation_prompt(task, tid)
    errors: list[str] = []
    raws: list[str] = []
    calls = 0

    for retry in range(int(args.contract_generation_retries)):
        current_prompt = prompt

        if retry:
            previous = raws[-1] if raws else ""
            current_prompt += (
                "\n\nPREVIOUS ATTEMPT COULD NOT BE PARSED AS THE REQUIRED CONTRACT JSON."
                "\nGenerate the contract again from the ORIGINAL QUESTION."
                "\nReturn exactly one complete JSON object matching the contract schema."
                "\nDo not use Markdown fences or commentary."
            )
            # Include only a short tail of the previous response as formatting
            # feedback; ORIGINAL QUESTION remains the semantic source of truth.
            if previous:
                current_prompt += (
                    "\nPrevious malformed output tail (formatting reference only):\n"
                    + previous[-1200:]
                )

        raw = model.generate(
            CONTRACT_SYSTEM_PROMPT,
            current_prompt,
            max_new_tokens=args.contract_generation_tokens,
            max_context=args.max_context,
        )
        calls += 1
        raws.append(str(raw or ""))

        contract = parse_generated_contract(raw)
        if isinstance(contract, dict):
            return {
                "contract": contract,
                "model_calls": calls,
                "raw_response": raw,
                "errors": errors,
            }

        errors.append(
            f"attempt {retry + 1}: generated response was not a usable contract JSON object"
        )

    raise RuntimeError(
        f"Fresh contract generation failed for {tid} after {calls} model calls: "
        f"{errors[-1] if errors else 'unknown error'}"
    )

def output_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = read_json(path)
    except Exception:
        return False
    return (
        isinstance(record, dict)
        and record.get("workflow_version") == WORKFLOW_VERSION
        and record.get("verifier_version") == VERIFIER_VERSION
        and record.get("repair_version") == REPAIR_VERSION
        and record.get("selection_status") in SELECTIONS
        and isinstance(record.get("contract"), dict)
        and record.get("processing_error") is None
    )


def strip_raw(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_raw(item)
            for key, item in value.items()
            if key not in {"raw_response", "confirmation_raw_response"}
        }
    if isinstance(value, list):
        return [strip_raw(item) for item in value]
    return value


def repair_call_count(result: JsonDict | None) -> int:
    if not isinstance(result, dict):
        return 0
    total = 0
    for item in result.get("attempts") or []:
        if isinstance(item, dict):
            total += int(item.get("repair_model_calls") or 0)
            total += int(item.get("verification_model_calls") or 0)
    return total


def final_verification(
    original: JsonDict,
    repair_result: JsonDict | None,
) -> JsonDict:
    if isinstance(repair_result, dict):
        accepted = repair_result.get("accepted_verification")
        if isinstance(accepted, dict):
            return accepted
    return original


def model_metadata(args: argparse.Namespace, model: LocalModel) -> JsonDict:
    return {
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "model_type": getattr(model, "model_type", None),
        "attention_implementation": getattr(model, "attention", args.attention_implementation),
        "context_length": getattr(model, "context_length", None),
        "greedy": True,
    }


def save_selected(
    *,
    destination: Path,
    tid: str,
    index: int,
    source: Path,
    original_contract: JsonDict,
    original_verification: JsonDict,
    selected_contract: JsonDict,
    selection_status: str,
    repair_result: JsonDict | None,
    args: argparse.Namespace,
    model: LocalModel,
    elapsed: float,
    source_contract_origin: str = "original_sft_contract",
    source_generation: JsonDict | None = None,
) -> None:
    final = final_verification(original_verification, repair_result)
    final_status = str(final.get("status") or original_verification.get("status") or "")

    selected_source = (
        "repaired"
        if selection_status in {
            "REPAIRED_CORRECT",
            "REPAIRED_CORRECT_FROM_UNCERTAIN",
        }
        else (
            "regenerated_contract_fallback"
            if source_contract_origin == "regenerated_from_question"
            else "original_fallback"
        )
        if selection_status in {
            "DEFECTIVE_UNRESOLVED",
            "UNCERTAIN_REPAIR_UNRESOLVED",
        }
        else (
            "regenerated_from_question"
            if source_contract_origin == "regenerated_from_question"
            else "original"
        )
    )

    save_json(
        destination,
        {
            "task_id": tid,
            "task_index": index,
            "dataset": "taco",
            "workflow": "contract_verification_repair",
            "workflow_version": WORKFLOW_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "repair_version": REPAIR_VERSION,
            "selection_status": selection_status,
            "selected_source": selected_source,
            "downstream_usable": True,

            # The downstream code generator reads this field.
            "contract": selected_contract,

            "original_contract": original_contract,
            "original_contract_fingerprint": fingerprint(original_contract),
            "selected_contract_fingerprint": fingerprint(selected_contract),
            "original_contract_path": str(source),
            "source_contract_origin": source_contract_origin,
            "source_contract_regenerated": (
                source_contract_origin == "regenerated_from_question"
            ),
            "source_contract_generation": (
                {
                    "model_calls": int(source_generation.get("model_calls") or 0),
                    "errors": list(source_generation.get("errors") or []),
                }
                if isinstance(source_generation, dict)
                else None
            ),

            "original_status": original_verification.get("status"),
            "original_workflow_action": original_verification.get("workflow_action"),
            "original_repair_recommended": bool(
                original_verification.get("repair_recommended")
            ),
            "original_repair_basis": original_verification.get("repair_basis"),
            "final_status": final_status,
            "original_verification": strip_raw(original_verification),
            "final_verification": strip_raw(final),
            "repair": strip_raw(repair_result) if repair_result is not None else None,
            "repair_attempts_used": (
                int(repair_result.get("attempts_used") or 0)
                if isinstance(repair_result, dict)
                else 0
            ),

            "model": model_metadata(args, model),
            "config": {
                "retries": args.retries,
                "json_retries": args.json_retries,
                "verify_tokens": args.verify_tokens,
                "confirm_tokens": args.confirm_tokens,
                "repair_tokens": args.repair_tokens,
                "contract_generation_tokens": args.contract_generation_tokens,
                "contract_generation_retries": args.contract_generation_retries,
                "max_context": args.max_context,
                "code_execution_used": False,
                "code_result_used": False,
                "reference_solution_used": False,
                "tests_used": False,
            },
            "elapsed_seconds": round(elapsed, 3),
            "processing_error": None,
            "finished_at": now(),
        },
    )


def save_error(
    destination: Path,
    *,
    tid: str,
    index: int,
    source: Path,
    contract: JsonDict | None,
    error: BaseException,
) -> None:
    save_json(
        destination,
        {
            "task_id": tid,
            "task_index": index,
            "dataset": "taco",
            "workflow": "contract_verification_repair",
            "workflow_version": WORKFLOW_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "repair_version": REPAIR_VERSION,
            "selection_status": "PROCESSING_ERROR",
            "selected_source": "none",
            "downstream_usable": False,
            "contract": contract,
            "original_contract_path": str(source),
            "processing_error": f"{type(error).__name__}: {error}",
            "finished_at": now(),
        },
    )


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()

    tasks = selected_tasks(load_tasks(args.task_file), args.start, args.count)
    contracts = index_contracts(args.contract_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight completeness: verification must not silently drop selected tasks.
    missing = []
    scheduled = []
    skipped = 0

    for index, task in tasks:
        tid = task_id(task, index)
        source = contracts.get(tid)
        if source is None:
            missing.append(tid)
            continue

        destination = output_path(args.output_dir, tid)
        if not args.overwrite and output_is_complete(destination):
            skipped += 1
            continue
        scheduled.append((index, task, tid, source, destination))

    if missing:
        raise RuntimeError(
            f"Missing source contracts for {len(missing)} selected tasks. "
            f"First missing: {missing[:20]}"
        )

    if not scheduled:
        print(
            f"selected={len(tasks)} scheduled=0 skipped={skipped} "
            f"contracts_indexed={len(contracts)}",
            flush=True,
        )
        return

    # The same SFT+LoRA runtime is reused for verification, repair, and
    # re-verification. The downstream BASE code model is not used here.
    model = LocalModel(
        args.model_path,
        args.adapter_path,
        args.gpu_id,
        args.attention_implementation,
    )

    original_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    processed = 0
    errors = 0
    repair_attempts_total = 0
    model_calls_total = 0
    regenerated_from_question = 0

    print(
        f"selected={len(tasks)} scheduled={len(scheduled)} skipped={skipped} "
        f"contracts_indexed={len(contracts)} gpu={args.gpu_id} "
        f"model={args.model_name} adapter={args.adapter_path is not None} "
        f"verifier={VERIFIER_VERSION} repair={REPAIR_VERSION}",
        flush=True,
    )

    for position, (index, task, tid, source, destination) in enumerate(
        scheduled,
        start=1,
    ):
        started = time.perf_counter()
        original_contract: JsonDict | None = None

        try:
            _, original_contract = load_contract(source)
            source_contract_origin = "original_sft_contract"
            source_generation: JsonDict | None = None
            source_generation_calls = 0

            if original_contract is None:
                source_generation = generate_contract_from_question(
                    model,
                    task,
                    tid=tid,
                    args=args,
                )
                original_contract = source_generation["contract"]
                source_generation_calls = int(source_generation.get("model_calls") or 0)
                source_contract_origin = "regenerated_from_question"
                regenerated_from_question += 1
                print(
                    f"[{position}/{len(scheduled)}] REGENERATED_SOURCE_CONTRACT {tid} "
                    f"calls={source_generation_calls}",
                    flush=True,
                )

            original_verification = verify_contract(
                model,
                task,
                original_contract,
                tid=tid,
                verify_tokens=args.verify_tokens,
                confirm_tokens=args.confirm_tokens,
                max_context=args.max_context,
                json_retries=args.json_retries,
            )

            original_status = str(original_verification.get("status") or "")
            if original_status not in FINAL_STATUSES:
                raise RuntimeError(f"Invalid verifier status: {original_status!r}")

            original_counts[original_status] += 1
            model_calls = (
                source_generation_calls
                + int(original_verification.get("model_calls") or 0)
            )
            repair_result: JsonDict | None = None

            repair_allowed = (
                original_status == "DEFECTIVE"
                or (
                    original_status == "UNCERTAIN"
                    and original_verification.get("repair_recommended") is True
                    and isinstance(original_verification.get("repair_issue"), dict)
                )
            )

            if original_status == "CORRECT":
                selected_contract = original_contract
                selection_status = "PRESERVED_CORRECT"

            elif original_status == "UNCERTAIN" and not repair_allowed:
                selected_contract = original_contract
                selection_status = "PRESERVED_UNCERTAIN"

            else:
                repair_result = repair_defective_contract(
                    model,
                    task,
                    original_contract,
                    original_verification,
                    args,
                    tid=tid,
                )

                repair_attempts_total += int(repair_result.get("attempts_used") or 0)
                model_calls += repair_call_count(repair_result)

                if repair_result.get("resolved") is True:
                    candidate = repair_result.get("accepted_contract")
                    accepted_verification = repair_result.get("accepted_verification")

                    if (
                        not isinstance(candidate, dict)
                        or not isinstance(accepted_verification, dict)
                        or accepted_verification.get("status") != "CORRECT"
                    ):
                        raise RuntimeError(
                            "Repair marked resolved without a CORRECT verified candidate"
                        )

                    selected_contract = candidate
                    selection_status = (
                        "REPAIRED_CORRECT_FROM_UNCERTAIN"
                        if original_status == "UNCERTAIN"
                        else "REPAIRED_CORRECT"
                    )

                else:
                    selected_contract = original_contract
                    selection_status = (
                        "UNCERTAIN_REPAIR_UNRESOLVED"
                        if original_status == "UNCERTAIN"
                        else "DEFECTIVE_UNRESOLVED"
                    )

            model_calls_total += model_calls
            elapsed = time.perf_counter() - started

            save_selected(
                destination=destination,
                tid=tid,
                index=index,
                source=source,
                original_contract=original_contract,
                original_verification=original_verification,
                selected_contract=selected_contract,
                selection_status=selection_status,
                repair_result=repair_result,
                args=args,
                model=model,
                elapsed=elapsed,
                source_contract_origin=source_contract_origin,
                source_generation=source_generation,
            )

            processed += 1
            selection_counts[selection_status] += 1

            final_status = (
                "CORRECT"
                if selection_status in {
                    "REPAIRED_CORRECT",
                    "REPAIRED_CORRECT_FROM_UNCERTAIN",
                }
                else original_status
            )
            attempts = int(repair_result.get("attempts_used") or 0) if repair_result else 0

            print(
                f"[{position}/{len(scheduled)}] {selection_status} {tid} "
                f"original={original_status} final={final_status} "
                f"repairs={attempts} calls={model_calls} seconds={elapsed:.3f}",
                flush=True,
            )

        except BaseException as error:
            errors += 1
            if errors == 1:
                traceback.print_exc()
            save_error(
                destination,
                tid=tid,
                index=index,
                source=source,
                contract=original_contract,
                error=error,
            )
            print(
                f"[{position}/{len(scheduled)}] PROCESSING_ERROR {tid}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    summary = {
        "workflow_version": WORKFLOW_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "repair_version": REPAIR_VERSION,
        "selected": len(tasks),
        "scheduled": len(scheduled),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "contracts_indexed": len(contracts),
        "original_status_counts": dict(sorted(original_counts.items())),
        "selection_counts": dict(sorted(selection_counts.items())),
        "repair_attempts_total": repair_attempts_total,
        "model_calls_total": model_calls_total,
        "regenerated_from_question": regenerated_from_question,
        "retries": args.retries,
        "json_retries": args.json_retries,
        "verify_tokens": args.verify_tokens,
        "confirm_tokens": args.confirm_tokens,
        "repair_tokens": args.repair_tokens,
        "contract_generation_tokens": args.contract_generation_tokens,
        "contract_generation_retries": args.contract_generation_retries,
        "max_context": args.max_context,
        "model_name": args.model_name,
        "model_type": getattr(model, "model_type", None),
        "attention_implementation": getattr(
            model,
            "attention",
            args.attention_implementation,
        ),
        "verifier_py_sha256": file_sha256(HERE / "verifier.py"),
        "repair_py_sha256": file_sha256(HERE / "repair.py"),
        "verify_and_repair_taco_py_sha256": file_sha256(Path(__file__)),
        "code_execution_used": False,
        "code_result_used": False,
        "reference_solution_used": False,
        "tests_used": False,
        "elapsed_seconds": round(time.perf_counter() - run_started, 2),
        "finished_at": now(),
    }

    save_json(
        args.output_dir / f"_summary_{args.start}_{args.count if args.count is not None else 'all'}.json",
        summary,
    )

    print(json.dumps(summary, indent=2), flush=True)

    if errors:
        raise RuntimeError(
            f"Verification/repair infrastructure errors occurred for {errors} tasks. "
            "Those tasks were not silently converted into semantic outcomes."
        )


if __name__ == "__main__":
    main()
