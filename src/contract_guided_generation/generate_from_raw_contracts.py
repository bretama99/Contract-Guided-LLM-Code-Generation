import argparse, json, logging, time
from datetime import datetime, timezone
from typing import Any
from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json, load_json_list, save_json, setup_logging
from src.common.llm_clients import call_chat_model, default_model, get_client
from src.common.parsing import extract_python_code, add_missing_common_alias_imports
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_path
from src.common.task_utils import select_tasks, task_identifier, task_entry_point, task_prompt

STAGE = "stage_2b_raw_contract_guided_generation"
SYSTEM_PROMPT = (
    "Generate complete executable Python benchmark solutions. "
    "Return only Python code."
)

PROMPT_TEMPLATE = """
You are generating a complete Python solution for a benchmark programming task.

The original benchmark prompt, function signature, examples, and tests implied by the prompt are the source of truth.
The raw contract is only a helper checklist. It may be incomplete or partially wrong.

OUTPUT RULES
- Return only Python code.
- Do not include markdown, explanations, or tests.
- Preserve the required function name exactly.
- Preserve the required function signature when given.
- Include imports only when needed.
- Define all helper functions used by the solution.
- The final code must be syntactically valid Python.

CONTRACT USAGE RULES
Use the raw contract to better understand:
- required behavior
- input assumptions
- output guarantees
- edge cases
- ordering or structural constraints

Do not blindly copy the contract.
Do not copy contract assertions into the final code.
Do not add input validation, type checks, assertions, exceptions, or error returns unless the original benchmark prompt explicitly requires them.
If the raw contract conflicts with the prompt, examples, signature, or required output behavior, ignore the contract and follow the original benchmark prompt.

BEHAVIOR RULES
Preserve exact behavior required by the task, including:
- return type
- ordering
- duplicate handling
- boundary cases
- empty-input behavior
- string/list/tuple formatting
- numeric precision when relevant
- required output structure
- exact API usage when relevant

Do not invent constraints beyond the prompt and contract.
Do not overfit to only the examples.
Solve the general task described by the prompt.

EXECUTION RULES
- Produce deterministic executable code.
- Avoid unnecessary filesystem, network, subprocess, or environment checks.
- Do not call external services.
- Avoid infinite loops or long-running operations.
- Prefer simple, direct, reliable implementations.

ENTRY POINT
{entry_point}

ORIGINAL BENCHMARK PROMPT
{prompt}

RAW CONTRACT
{contract}
""".strip()

def extract_contract(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Raw contract file must contain a JSON object.")
    if record.get("status") not in (None, "success"):
        raise RuntimeError("Raw contract status is not success.")
    contract = record.get("contract", record)
    if not isinstance(contract, dict):
        raise ValueError("Raw contract record does not contain a valid contract object.")
    return contract

def compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    task = contract.get("task") if isinstance(contract.get("task"), dict) else {}
    interface = contract.get("interface") if isinstance(contract.get("interface"), dict) else {}

    postconditions = contract.get("postconditions", [])
    if not isinstance(postconditions, list):
        postconditions = []

    edge_cases = contract.get("edge_cases", [])
    if not isinstance(edge_cases, list):
        edge_cases = []

    safe_postconditions = []
    for item in postconditions:
        if not isinstance(item, dict):
            continue

        description = str(item.get("description", "")).strip()
        if not description:
            continue

        if description.lower() in {
            "returns the expected result",
            "returns the correct output",
            "result satisfies the task behavior",
        }:
            continue

        safe_postconditions.append(
            {
                "target": item.get("target", "return"),
                "kind": item.get("kind", "semantic"),
                "description": description,
                "source": item.get("source", "inferred"),
            }
        )

        if len(safe_postconditions) >= 3:
            break

    safe_edge_cases = []
    for item in edge_cases:
        if not isinstance(item, dict):
            continue

        case = str(item.get("case", "")).strip()
        expected = str(item.get("expected_behavior", "")).strip()

        if not case or not expected:
            continue

        safe_edge_cases.append(
            {
                "case": case,
                "expected_behavior": expected,
                "source": item.get("source", "inferred"),
            }
        )

        if len(safe_edge_cases) >= 5:
            break

    return {
        "task_summary": task.get("summary", ""),
        "signature": task.get("signature", ""),
        "imports_required": task.get("imports_required", []),
        "helper_functions_required": task.get("helper_functions_required", []),
        "interface": interface,
        "semantic_postconditions": safe_postconditions,
        "edge_cases": safe_edge_cases,
    }
    
def build_prompt(task: dict[str, Any], contract: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        entry_point=task_entry_point(task),
        prompt=task_prompt(task),
        contract=json.dumps(compact_contract(contract), indent=2, ensure_ascii=False),
    )

def make_record(task: dict[str, Any], benchmark: str, provider: str, model: str, temperature: float, max_tokens: int, **extra: Any) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "benchmark": benchmark,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": task.get("source"),
        "source_version": task.get("source_version"),
        **extra,
    }

def generate_one(task: dict[str, Any], args: argparse.Namespace, benchmark: str, model: str, client: Any) -> dict[str, Any]:
    task_id = task_identifier(task)
    contract_file = raw_contract_path(args.dataset, task_id, args.provider, model)
    prompt = raw_response = ""
    code = api_result = None
    common = dict(task=task, benchmark=benchmark, provider=args.provider, model=model, temperature=args.temperature, max_tokens=args.max_tokens)
    try:
        if not contract_file.exists():
            raise FileNotFoundError(f"Missing raw contract file: {contract_file}")
        prompt = build_prompt(task, extract_contract(load_json(contract_file)))
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=False,
        )
        code = extract_python_code(
        raw_response,
        entry_point=task_entry_point(task),
        validate=True,
        )
        code = add_missing_common_alias_imports(code)
        code = extract_python_code(
            code,
            entry_point=task_entry_point(task),
            validate=True,
        )
        return make_record(**common, status="success", contract_path=str(contract_file), generation_prompt=prompt, raw_response=raw_response, generated_code=code, api_result=api_result, speed=api_result, error=None)
    except Exception as exc:
        logging.exception("Stage 2B generation failed for %s", task.get("task_id"))
        return make_record(**common, status="failed", contract_path=str(contract_file), generation_prompt=prompt, raw_response=raw_response, generated_code=code, api_result=api_result, speed=api_result, error=str(exc))

def generate(args: argparse.Namespace) -> None:
    info = DATASETS[args.dataset]
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(info["path"]), start=args.start, count=args.count)
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=args.start):
        task_id = task_identifier(task)
        output_file = raw_contract_code_path(args.dataset, task_id, args.provider, model)
        if output_file.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {task_id}")
            continue
        result = generate_one(task, args, info["label"], model, client)
        save_json(output_file, result)
        ok = result["status"] == "success"
        counts["completed" if ok else "failed"] += 1
        print(f"[{index}] {'DONE' if ok else 'FAIL'} {task_id}" + ("" if ok else f": {result['error']}"))
        if args.delay > 0:
            time.sleep(args.delay)
    print("\nStage 2B raw contract-guided generation finished")
    print(f"Completed: {counts['completed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2B: generate code using raw contracts")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    setup_logging(LOG_ROOT / "stage2_raw_contract_guided_generation.log")
    generate(parse_args())

if __name__ == "__main__":
    main()