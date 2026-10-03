from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import src.generation.generation_core_extended as core


SYSTEM_PROMPT = """
Solve the programming task exactly.
Return one complete Python solution only.
No tests, explanations, reasoning, Markdown, or code fences.
Preserve the required interface and I/O behavior.
Use an algorithm efficient enough for the stated constraints.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate vanilla TACO code using the guided pipeline's runtime."
        )
    )

    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")

    # Both vanilla and guided code use the unadapted code model.
    parser.set_defaults(adapter_path=None)

    core.add_runtime_arguments(parser)

    return core.finish_arguments(parser, parser.parse_args())


def build_user_prompt(task):
    question = core.taco_question(task)

    if not question:
        raise ValueError("Task has no question text")

    return "\n".join([
        "ORIGINAL TASK:",
        question,
        "",
        "REQUIRED INTERFACE:",
        core.taco_interface(task),
        "",
        "Return only the complete Python solution, "
        "without Markdown or explanation.",
    ])


def generate_record(runtime, args, job):
    record = {
        "task_id": job["task_id"],
        "dataset": "taco",
        "workflow": "vanilla",
        "provider": "local_transformers",
        "model": args.model_name,
        "generated_at": None,
        "code": "",
    }

    metadata = {}
    error = None

    try:
        encoded = core.encode_chat_32(
            runtime["tokenizer"],
            SYSTEM_PROMPT,
            build_user_prompt(job["task"]),
        )

        _, final, metadata = core.generate_response_32(
            runtime,
            args,
            encoded,
            job["index"],
            "code",
        )

        # Preserve the response exactly as returned by the shared core.
        record["code"] = final

    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"

    record["generated_at"] = core.now()

    return record, metadata, error


def main():
    args = parse_args()

    selected = core.selected_tasks(
        core.load_tasks(args.task_file),
        args.start,
        args.count,
    )

    if not selected:
        raise ValueError("No tasks selected; check --start and --count")

    jobs = core.prepare_jobs(selected, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime = None
    written = 0
    skipped = 0
    failed = 0
    time_hits = 0
    token_hits = 0
    empty = 0

    started = time.perf_counter()

    for job in jobs:
        previous = core.existing_record(
            job,
            "code",
            args.overwrite,
        )

        if previous is not None:
            skipped += 1
            print(f"REUSED {job['task_id']}", flush=True)
            continue

        if runtime is None:
            runtime = core.load_runtime_32(args, stage="code")

        print(f"GENERATING {job['task_id']}", flush=True)

        record, metadata, error = generate_record(
            runtime,
            args,
            job,
        )

        core.save_json(job["destination"], record)

        written += 1
        failed += int(error is not None)
        empty += int(not record["code"].strip())

        timed_out = metadata.get("time_budget_exhausted", False)
        limited = metadata.get("hit_output_limit", False)

        time_hits += int(timed_out)
        token_hits += int(limited)

        print(
            f"SAVED {job['task_id']} "
            f"tokens={metadata.get('generated_tokens')} "
            f"limit={limited} "
            f"time_limit={timed_out} "
            f"seconds={metadata.get('generation_seconds')} "
            f"tok/s={metadata.get('tokens_per_second')} "
            f"eos={metadata.get('ended_with_eos', False)} "
            f"issue={error}",
            flush=True,
        )

    summary = {
        "selected": len(jobs),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
    }

    count_label = args.count if args.count is not None else "all"
    core.save_json(
        args.output_dir / f"_summary_{args.start}_{count_label}.json",
        summary,
    )
    print(json.dumps(summary, indent=2), flush=True)

    print(
        f"NEW_TIME_LIMIT_HITS {time_hits}; "
        f"NEW_OUTPUT_LIMIT_HITS {token_hits}; "
        f"EMPTY_RESPONSES {empty}",
        flush=True,
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())