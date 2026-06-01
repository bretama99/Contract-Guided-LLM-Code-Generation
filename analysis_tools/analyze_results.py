#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from formatting import format_worksheet
except ImportError:
    from analysis_tools.formatting import format_worksheet

from src.common.config import DATASETS

HELPED = "Helped by Contracts"
REGRESSED = "Regressed by Contracts"
FAILED = "Failed Both"

GEN_FAILS = {"generation_failed", "generation_error", "missing_generation",
             "empty_code", "empty_generated_code", "missing_result"}

HELPED_COLS = ["Task ID", "Vanilla Explanation", "Prompt", "Vanilla Code",
               "Contract-Guided Code", "Vanilla Failing Test", "Preconditions",
               "Postconditions", "Invariants", "Interpretation"]

REGRESSED_COLS = ["Task ID", "Contract-Guided Explanation", "Prompt",
                  "Vanilla Code", "Contract-Guided Code",
                  "Contract-Guided Failing Test", "Preconditions",
                  "Postconditions", "Invariants", "Interpretation"]

FAILED_COLS = ["Task ID", "Vanilla Explanation", "Contract-Guided Explanation",
               "Prompt", "Vanilla Code", "Contract-Guided Code",
               "Vanilla Failing Test", "Contract-Guided Failing Test",
               "Preconditions", "Postconditions", "Invariants", "Interpretation"]


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("._-") or "x"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(x):
    return re.sub(r"\s+", " ", str(x or "").replace("\r", " ")).strip()


def passed(r):
    return bool(r and r.get("passed") is True)


def ftype(r):
    r = r or {}
    return str(r.get("failure_type") or (r.get("evaluation") or {}).get("failure_type") or "unknown_failure")


def genfail(r):
    if not r:
        return True
    return ftype(r) in GEN_FAILS or str((r.get("generation") or {}).get("status") or "") in GEN_FAILS


def code(r):
    g = (r or {}).get("generation") or {}
    return str(g.get("code") or g.get("generated_code") or (r or {}).get("generated_code") or "")


def load_tasks(dataset):
    path = Path(DATASETS[dataset]["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}

    tasks = {}
    for r in load(path):
        tasks[str(r.get("task_id"))] = {
            "entry_point": r.get("entry_point") or "",
            "prompt": r.get("prompt") or r.get("instruct_prompt") or r.get("complete_prompt") or "",
        }
    return tasks


def prompt(tasks, tid, r):
    if tid in tasks:
        return tasks[tid]["prompt"]
    t = (r or {}).get("task") or {}
    return str(t.get("prompt") or t.get("instruct_prompt") or t.get("complete_prompt") or "")


def entry(r):
    return str((r or {}).get("entry_point") or ((r or {}).get("task") or {}).get("entry_point") or "")


def explain(r):
    if not r:
        return "No result was found for this task."
    if passed(r):
        return "Passed."
    if r.get("failure_explanation"):
        return clean(r["failure_explanation"])

    ev, g, ft = r.get("evaluation") or {}, r.get("generation") or {}, ftype(r)

    if genfail(r):
        return clean(f"Generation failed before evaluation: {g.get('error') or ft}.")

    stage = "plus" if ft == "plus_test_failure" else "original" if ft == "original_test_failure" else "benchmark"
    idx, inp = ev.get("failed_case_index"), ev.get("input")
    exp, act = ev.get("expected"), ev.get("actual")
    err = ev.get("actual_error") or ev.get("exception")

    if exp is not None and act is not None:
        return clean(f"Failed {stage} test case #{idx}. Input: {inp}. Expected: {exp}. Actual: {act}.")
    if err:
        return clean(f"Failed {stage} test case #{idx}. Input: {inp}. Actual output could not be computed because: {err}")
    if ev.get("failed_assertion"):
        return clean(f"Failed assertion: {ev.get('failed_assertion')}.")

    return clean(f"Failed with failure type: {ft}.")


def call(ep, inp):
    if inp is None:
        return ""
    s = str(inp).strip()
    if ep and re.search(rf"\b{re.escape(ep)}\s*\(", s):
        return s
    if ep and "candidate(" in s:
        return re.sub(r"\bcandidate\s*\(", f"{ep}(", s)
    if not ep:
        return s
    return f"{ep}(*{s})" if s[:1] in "[(" else f"{ep}({s})"


def failing_test(r):
    if not r or passed(r):
        return ""
    if genfail(r):
        return f"# Generation failed before evaluation\n# reason: {((r.get('generation') or {}).get('error') or ftype(r))}"

    ev, ep, lines = r.get("evaluation") or {}, entry(r), []
    for k, v in [("failure_type", ftype(r)), ("failed_case_index", ev.get("failed_case_index")),
                 ("input", ev.get("input")), ("expected", ev.get("expected")), ("actual", ev.get("actual"))]:
        if v is not None:
            lines.append(f"# {k}: {v}")

    err = ev.get("actual_error") or ev.get("exception")
    if err:
        lines += ["# actual_error:"] + [f"# {x}" for x in str(err).splitlines()]

    lines.append("")

    if ev.get("failed_assertion"):
        lines.append(re.sub(r"\bcandidate\s*\(", f"{ep}(", str(ev["failed_assertion"])))
    else:
        c = call(ep, ev.get("input"))
        lines.append(f"assert {c} == {ev.get('expected')}" if c and ev.get("expected") is not None else c or "# Failed test input was not available.")

    return "\n".join(lines)


def contract_text(x):
    if not x:
        return ""
    if isinstance(x, list):
        return "\n".join(filter(None, map(contract_text, x)))
    if isinstance(x, dict):
        for k in ["description", "condition", "expression", "expected_behavior", "case"]:
            if x.get(k):
                return str(x[k]).strip()
        return "\n".join(filter(None, (contract_text(v) for v in x.values())))
    return str(x).strip()


def get_contract(root, dataset, provider, model, tid, r):
    g = (r or {}).get("generation") or {}
    obj = g.get("contract") or (r or {}).get("contract") or ((r or {}).get("task") or {}).get("contract") or {}

    paths = [
        g.get("contract_path"),
        root / "outputs" / "contracts" / "raw" / safe(provider) / safe(model) / dataset / f"{safe(tid)}_contract.json",
    ]

    for p in paths:
        if not p:
            continue
        p = Path(p)
        p = p if p.is_absolute() else root / p
        if p.exists():
            data = load(p)
            obj = data.get("contract", data) if isinstance(data, dict) else {}
            break

    obj = obj if isinstance(obj, dict) else {}
    return {
        "Preconditions": contract_text(obj.get("preconditions")),
        "Postconditions": contract_text(obj.get("postconditions")),
        "Invariants": contract_text(obj.get("invariants")),
    }


def bug(r):
    if not r:
        return "missing_result"
    if passed(r):
        return "passed"
    text = (explain(r) + " " + str((r.get("evaluation") or {}).get("actual_error") or "")).lower()
    if genfail(r):
        return "generation failure"
    if "timeout" in text:
        return "timeout / inefficient algorithm"
    if "nameerror" in text or "not defined" in text:
        return "missing import or undefined name"
    if "syntaxerror" in text:
        return "syntax error"
    if any(x in text for x in ["typeerror", "valueerror", "traceback"]):
        return "runtime exception"
    if ftype(r) == "plus_test_failure":
        return "EvalPlus edge-case failure"
    return "wrong algorithm / wrong output"


def effect(v, c):
    if passed(v) and passed(c):
        return "both_passed"
    if not passed(v) and passed(c):
        return "contract_helped"
    if passed(v) and not passed(c):
        return "contract_regressed"
    return "both_failed"


def interpretation(e, vb, cb, con):
    q = "has output guarantees only" if con["Postconditions"] else "missing or weak contract"
    if e == "contract_helped":
        return f"Contract guidance fixed a vanilla failure, likely by improving behavior related to: {vb}."
    if e == "contract_regressed":
        return f"Contract guidance introduced a regression. Contract quality: {q}. Contract-guided bug: {cb}."
    if e == "both_failed":
        return f"Both methods failed. Vanilla bug: {vb}. Contract-guided bug: {cb}. Raw contract did not fix the task."
    return "Both methods passed."


def rows(root, dataset, provider, model, vanilla, contract):
    tasks = load_tasks(dataset)
    V = {str(r.get("task_id")): r for r in vanilla if r.get("task_id")}
    C = {str(r.get("task_id")): r for r in contract if r.get("task_id")}

    def sort_key(x):
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", x)]

    out = []
    for tid in sorted(set(tasks) | set(V) | set(C), key=sort_key):
        v, c = V.get(tid), C.get(tid)
        con = get_contract(root, dataset, provider, model, tid, c)
        e = effect(v, c)

        row = {
            "Task ID": tid,
            "Vanilla Explanation": explain(v),
            "Contract-Guided Explanation": explain(c),
            "Prompt": prompt(tasks, tid, c or v),
            "Vanilla Code": code(v),
            "Contract-Guided Code": code(c),
            "Vanilla Failing Test": failing_test(v),
            "Contract-Guided Failing Test": failing_test(c),
            **con,
        }

        row["Contract Effect"] = e
        row["Interpretation"] = interpretation(e, bug(v), bug(c), con)
        out.append(row)

    return out


def total(s):
    return int(s.get("total_tasks") or s.get("total") or s.get("task_count") or 0)


def gen_fail_count(s):
    fc = s.get("failure_counts") or {}
    return sum(int(fc.get(k, 0) or 0) for k in GEN_FAILS) if isinstance(fc, dict) else 0


def summary_df(vs, cs):
    def line(name, s):
        t, p = total(s), int(s.get("passed") or 0)
        p1 = float(s.get("pass@1") or s.get("pass_at_1") or (p / t if t else 0))
        return {
            "Method": name,
            "Total Tasks": t,
            "Successful Generations": s.get("successful_generation_count", t - gen_fail_count(s) if t else ""),
            "Filled Generation Failures": s.get("filled_failure_count", gen_fail_count(s) or 0),
            "Passed": p,
            "Failed": int(s.get("failed", t - p)),
            "Pass@1": p1,
            "Pass@1 Percent": s.get("pass@1_percent", s.get("pass_at_1_percent", round(p1 * 100, 2))),
        }

    a, b = line("Vanilla", vs), line("Raw Contract Guided", cs)
    t = a["Total Tasks"] or b["Total Tasks"]
    net = {
        "Method": "Net Difference",
        "Total Tasks": t,
        "Successful Generations": b["Successful Generations"] - a["Successful Generations"],
        "Filled Generation Failures": b["Filled Generation Failures"] - a["Filled Generation Failures"],
        "Passed": b["Passed"] - a["Passed"],
        "Failed": b["Failed"] - a["Failed"],
        "Pass@1": (b["Passed"] - a["Passed"]) / t if t else 0,
        "Pass@1 Percent": round(((b["Passed"] - a["Passed"]) / t) * 100, 2) if t else 0,
    }
    return pd.DataFrame([a, b, net])


def effect_df(rs):
    c, t = Counter(r["Contract Effect"] for r in rs), len(rs)
    data = [
        ("both_passed", "Both passed", "Both methods solved the task."),
        ("both_failed", "Both failed", "The raw contract did not fix the task."),
        ("contract_helped", "Vanilla failed, contract passed", "The contract-guided method improved over vanilla."),
        ("contract_regressed", "Vanilla passed, contract failed", "The raw contract or contract prompt likely misled generation."),
    ]

    out = [{"Outcome": label, "Count": c[k], "Percent of Tasks": round(c[k] / t * 100, 2) if t else 0, "Meaning": m}
           for k, label, m in data]

    net = c["contract_helped"] - c["contract_regressed"]
    out.append({"Outcome": "Net improvement", "Count": net,
                "Percent of Tasks": round(net / t * 100, 2) if t else 0,
                "Meaning": "Contract-helped tasks minus contract-regressed tasks."})
    return pd.DataFrame(out)


def shift_df(vs, cs):
    vc, cc = vs.get("failure_counts") or {}, cs.get("failure_counts") or {}
    out = []

    for k in sorted((set(vc) | set(cc)) - {"passed"}):
        v, c = int(vc.get(k, 0) or 0), int(cc.get(k, 0) or 0)
        d = c - v

        meaning = (
            "EvalPlus edge-case/robustness failures" if k == "plus_test_failure"
            else "core benchmark functional failures" if k == "original_test_failure"
            else "failures before executable code was produced" if k in GEN_FAILS
            else "this failure category"
        )

        direction = "increased" if d > 0 else "decreased" if d < 0 else "did not change"

        out.append({
            "Failure Type": k,
            "Vanilla Count": v,
            "Contract-Guided Count": c,
            "Change": d,
            "Interpretation": f"{meaning} {direction} by {abs(d)}.",
        })

    return pd.DataFrame(out)


def candidates(root, dataset, provider, model, method, kind):
    sp, sm = safe(provider), safe(model)

    if DATASETS[dataset].get("evalplus_dataset"):
        return [root / "results" / "evalplus" / method / sp / sm / dataset / f"{kind}.json"]

    if method == "vanilla":
        return [
            root / "results" / "vanilla" / f"stage1_{dataset}_{provider}_{sm}_{kind}.json",
            root / "results" / "vanilla" / f"stage1_{dataset}_{sp}_{sm}_{kind}.json",
        ]

    return [root / "results" / "contract_guided_generation" / "raw_contracts" / sp / sm / dataset / f"{kind}.json"]


def first(paths):
    return next((p for p in paths if p.exists()), None)


def analyze(args, dataset):
    root, provider, model = Path(args.root).resolve(), args.provider, args.model

    vs = first(candidates(root, dataset, provider, model, "vanilla", "summary"))
    vd = first(candidates(root, dataset, provider, model, "vanilla", "details"))
    cs = first(candidates(root, dataset, provider, model, "raw_contracts", "summary"))
    cd = first(candidates(root, dataset, provider, model, "raw_contracts", "details"))

    if not all([vs, vd, cs, cd]):
        raise FileNotFoundError(f"{dataset}: missing vanilla/raw_contracts summary/details json files")

    v_sum, c_sum = load(vs), load(cs)
    rs = rows(root, dataset, provider, model, load(vd), load(cd))

    if dataset == "evalplus":
        fname = "EvalPlus_analysis.xlsx"
    elif dataset == "evalplus_mbpp":
        fname = "EvalPlus_MBPP_analysis.xlsx"
    else:
        fname = f"{dataset}_analysis.xlsx"

    out = Path(args.output) if args.output else root / "analysis" / safe(model) / "basic" / fname
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary_df(v_sum, c_sum).to_excel(w, sheet_name="Summary", index=False, startrow=0)
        effect_df(rs).to_excel(w, sheet_name="Summary", index=False, startrow=8)
        shift_df(v_sum, c_sum).to_excel(w, sheet_name="Summary", index=False, startrow=18)
        format_worksheet(w.sheets["Summary"])

        for sheet, eff, cols in [
            (HELPED, "contract_helped", HELPED_COLS),
            (REGRESSED, "contract_regressed", REGRESSED_COLS),
            (FAILED, "both_failed", FAILED_COLS),
        ]:
            pd.DataFrame([r for r in rs if r["Contract Effect"] == eff], columns=cols).to_excel(
                w, sheet_name=sheet, index=False
            )
            format_worksheet(w.sheets[sheet])

    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", default="evalplus", choices=["all", *sorted(DATASETS)])
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--model", default="openai/gpt-3.5-turbo")
    p.add_argument("--output")
    args = p.parse_args()

    if args.dataset == "all":
        if args.output:
            raise SystemExit("--output can only be used with one dataset")
        for dataset in sorted(DATASETS):
            try:
                analyze(args, dataset)
            except FileNotFoundError as e:
                print(f"SKIP {e}")
    else:
        analyze(args, args.dataset)


if __name__ == "__main__":
    main()