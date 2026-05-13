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
    "This is Stage 2 raw contract-guided generation. "
    "The original benchmark prompt is authoritative. "
    "The raw contract is unvalidated guidance only. "
    "Use the raw contract only when it agrees with and clarifies the original prompt. "
    "Return only Python code. "
    "Do not return markdown, prose, explanations, or text outside code."
)

PROMPT_TEMPLATE = """
You are generating a complete Python solution for a benchmark programming task.

STAGE 2 CONTEXT
This is raw contract-guided generation.
The contract below is raw and unvalidated.
It may be incomplete, vague, or wrong.

AUTHORITY ORDER
1. The original benchmark prompt is the source of truth.
2. Examples and exact requirements in the original prompt override the raw contract.
3. The raw contract is only supplementary guidance.
4. Ignore any raw contract clause that conflicts with the original prompt.

OUTPUT RULES
Return only executable Python code.
Do not include markdown fences.
Do not include explanations.
Do not include prose.
Do not include tests.
Do not include top-level assertions.
Do not include unnecessary print statements.
Preserve the required function name and signature.

EXACT-BEHAVIOR RULES
Preserve exact behavior required or implied by the original prompt:
- exact return type
- exact string values
- exact exception messages
- exact numeric precision or rounding rule
- exact list, tuple, set, dictionary, or object structure
- exact ordering, tie-breaking, filtering, and duplicate-handling behavior
- exact file paths and returned path types
- exact DataFrame columns, index, shape, ordering, and dtypes
- exact plot titles, labels, colors, legends, bins, and returned axis or figure objects
- exact external API call signatures and keyword arguments when relevant

Do not generalize or simplify exact strings.
Do not replace short aliases with equivalent long names when tests may inspect them.
For example, if the prompt implies color="r", do not use color="red".
If the prompt implies requests.post(url, json=payload), do not use data=payload.
If the prompt implies returning a pathlib.Path, do not return a string.

IMPORT AND SYMBOL RULES
Include every import used by the code.
Before finalizing code, verify that every symbol is imported or defined.
Common aliases must be imported explicitly when used:
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns

Do not assume aliases already exist.
Do not use undefined helper functions.
Include helper functions only when needed by the required solution.

BENCHMARK AND MOCKING RULES
Some benchmark tasks inspect exact output, exact side effects, or exact API calls.
Some tasks may use mocks for filesystem, network, subprocess, plotting, database,
or external-library behavior.

Do not perform unnecessary real filesystem, network, subprocess, FTP, OS, or
environment checks unless the original prompt explicitly requires them.

If the task involves an API that may be mocked, call the expected API directly
and preserve the call shape implied by the original prompt.
Do not reject valid mocked benchmark inputs merely because files, URLs,
processes, or paths do not exist in the real environment.

For algorithmic tasks:
- prefer a simple deterministic implementation
- do not import heavy external libraries unless the prompt requires them
- do not add input validation unless specified
- preserve edge-case behavior from examples

For requests/network tasks:
- preserve expected keyword arguments such as json=, data=, timeout=, headers=
- use response.text for text or HTML parsing when appropriate
- use response.content only when binary bytes are required
- do not replace json= with data=

For subprocess/OS tasks:
- preserve the subprocess or OS API shape implied by the prompt
- do not add stdout=, stderr=, shell=, or check= unless required
- do not add os.path.exists checks before a mocked call unless required

For plotting tasks:
- return the exact object type expected by the prompt
- set exact titles, labels, legends, colors, bins, and axes
- do not call plt.show()

For pandas/DataFrame tasks:
- preserve columns, index, shape, ordering, and dtypes
- handle mixed numeric/string columns only as required by the prompt
- do not transpose, reset index, or sort unless required
- avoid operations that fail on constant, empty, or mixed-type data unless the prompt requires them

DEFENSIVE-CHECK RULES
Do not add artificial precondition checks that reject valid benchmark inputs.
Do not raise errors for valid mocked benchmark scenarios.
Do not add invalid-input behavior unless the original prompt explicitly specifies it.

TIME AND EXECUTION RULES
Avoid unbounded loops, sleeps, retries, network waits, and long-running subprocess calls.
Use deterministic finite operations.
Do not call external services.

ENTRY POINT
{entry_point}

ORIGINAL BENCHMARK PROMPT
{prompt}

RAW CONTRACT GUIDANCE
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