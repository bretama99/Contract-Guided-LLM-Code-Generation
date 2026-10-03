from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generation import generation_core_extended as core
from src.prompts.generation_prompts import CONTRACT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one contract per TACO task."
    )

    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")

    core.add_runtime_arguments(parser)

    return core.finish_arguments(parser, parser.parse_args())


def build_user_prompt(task):
    question = core.taco_question(task)

    if not question:
        raise ValueError("Task has no question text")

    return "\n".join([
        "REQUIRED INTERFACE:",
        core.taco_interface(task),
        "",
        "QUESTION:",
        question,
        "",
        "CONTRACT:",
    ])


def generate_record(runtime, args, job):
    record = {
        "task_id": job["task_id"],
        "task_index": job["index"],
        "contract": "",
        "generation_seconds": None,
        "seed": args.seed + job["index"],
        "generated_at": None,
    }

    metadata = {}
    error = None

    try:
        encoded = core.encode_chat_32(
            runtime["tokenizer"],
            CONTRACT_SYSTEM_PROMPT,
            build_user_prompt(job["task"]),
        )

        _, final, metadata = core.generate_response_32(
            runtime,
            args,
            encoded,
            job["index"],
            "contracts",
        )

        record["contract"] = final
        record["generation_seconds"] = metadata["generation_seconds"]
        record["seed"] = metadata["seed"]

        # Decode JSON for storage only; preserve every other response.
        try:
            value = json.loads(final)
        except ValueError:
            pass
        else:
            if isinstance(value, dict):
                record["contract"] = value

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
    time_hits = 0

    summary = {
        "selected": len(jobs),
        "saved": 0,
        "reused": 0,
        "empty_responses": 0,
        "generation_errors": 0,
        "new_output_limit_hits": 0,
        "started_at": core.now(),
    }

    for job in jobs:
        previous = core.existing_record(
            job,
            "contract",
            args.overwrite,
        )

        if previous is not None:
            summary["reused"] += 1
            print(f"REUSED {job['task_id']}", flush=True)
            continue

        if runtime is None:
            runtime = core.load_runtime_32(args, stage="contracts")

        print(f"GENERATING {job['task_id']}", flush=True)

        record, metadata, error = generate_record(
            runtime,
            args,
            job,
        )

        core.save_json(job["destination"], record)
        summary["saved"] += 1

        value = record["contract"]

        summary["empty_responses"] += int(
            isinstance(value, str) and not value.strip()
        )
        summary["generation_errors"] += int(error is not None)
        summary["new_output_limit_hits"] += int(
            metadata.get("hit_output_limit", False)
        )

        timed_out = metadata.get("time_budget_exhausted", False)
        time_hits += int(timed_out)

        print(
            f"SAVED {job['task_id']} "
            f"tokens={metadata.get('generated_tokens')} "
            f"limit={metadata.get('hit_output_limit', False)} "
            f"time_limit={timed_out} "
            f"seconds={record['generation_seconds']} "
            f"tok/s={metadata.get('tokens_per_second')} "
            f"eos={metadata.get('ended_with_eos', False)} "
            f"issue={error}",
            flush=True,
        )

    summary["finished_at"] = core.now()
    count_label = args.count if args.count is not None else "all"

    core.save_json(
        args.output_dir / f"_summary_{args.start}_{count_label}.json",
        summary,
    )

    print(
        json.dumps(summary, indent=2, ensure_ascii=False),
        flush=True,
    )
    print(
        f"NEW_TIME_LIMIT_HITS {time_hits}; "
        "partial responses were preserved.",
        flush=True,
    )

    return 1 if summary["generation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())