import argparse, re
from pathlib import Path
import pandas as pd
import unicodedata
from analysis_tools.old.analyze_results import HELPED, REGRESSED, FAILED, HELPED_COLS, REGRESSED_COLS, FAILED_COLS
from formatting import format_worksheet
from src.common.config import DATASETS
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object


# required keys for llm response
REQUIRED_KEYS = {
    "failure_reasons",
    "explanation",
    "minimal_fix_direction",
}


def analyze(root, dataset, client, model):
    
    # get analysis file
    analysis = get_input_path(root, dataset, model)
    
    if not analysis.exists():
        raise FileNotFoundError(f"{dataset}: missing analysis")

    new_sheets = []

    # analyze each sheet and create new contents
    for sheet in [HELPED, REGRESSED, FAILED]:
        df = pd.read_excel(analysis, sheet_name=sheet)
        result = analyze_sheet(df, client, model)
        new_sheets.append((sheet, result))

    # get output path
    if dataset == "evalplus":
        fname = "EvalPlus_detailed_analysis.xlsx"
    elif dataset == "evalplus_mbpp":
        fname = "EvalPlus_MBPP_detailed_analysis.xlsx"
    else:
        fname = f"{dataset}_detailed_analysis.xlsx"

    out = root / "analysis" / safe(model) / "detailed" / fname
    out.parent.mkdir(parents=True, exist_ok=True)

    # write contents
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for sheet, new_sheet in new_sheets:
            new_sheet.to_excel(w, sheet_name=sheet, index=False)
            format_worksheet(w.sheets[sheet])

    print(f"Wrote {out}")

    
# returns a new dataframe
def analyze_sheet(df, client, model):

    new_sheet = []

    # check which version failed
    df_cols = list(df.columns)
    if df_cols == HELPED_COLS:
        failure_type = "Vanilla"
    elif df_cols == REGRESSED_COLS:
        failure_type = "Contract-Guided"
    elif df_cols == FAILED_COLS:
        failure_type = "Both"
    else:
        raise ValueError("Unknown column names")

    # analyze rows and append results
    for row_num, row in df.iterrows():
        result = analyze_row(row, failure_type, client, model)
        new_sheet.append(result)
        print(f"[DONE] {failure_type}: {row_num}")

    return pd.DataFrame(new_sheet)


# returns a new row (as a dictionary) to be appended to a table
def analyze_row(row, failure_type, client, model):

    new_row = {"Task ID": row["Task ID"], "Prompt": row["Prompt"]}
    
    if failure_type == "Vanilla" or failure_type == "Contract-Guided":
        append_analysis(row, new_row, failure_type, client, model)

    elif failure_type == "Both":
        append_analysis(row, new_row, "Vanilla", client, model)

        vanilla_failure = extract_failure_signature(row["Vanilla Failing Test"])
        contract_failure = extract_failure_signature(row["Contract-Guided Failing Test"])
        
        empty_signature = ("", "", "", "")

        if (
            vanilla_failure != empty_signature
            and contract_failure != empty_signature
            and vanilla_failure == contract_failure
        ):
            append_analysis(
                row,
                new_row,
                "Contract-Guided",
                client,
                model,
                errors="Same as vanilla error type.",
                explanation="Same as vanilla explanation.",
                suggestion="Same as vanilla suggestion.",
            )
        else:
            append_analysis(row, new_row, "Contract-Guided", client, model)
            
    else:
        raise ValueError("Invalid Failure Type")
    
    return new_row


def append_analysis(old_row, new_row, failure_type, client, model, errors=None, explanation=None, suggestion=None):
    prompt = old_row["Prompt"]
    code = old_row[f"{failure_type} Code"]
    failure = old_row[f"{failure_type} Failing Test"]

    if errors is None or explanation is None or suggestion is None:
        try:
            errors, explanation, suggestion = get_llm_analysis(prompt, code, failure, client, model)
        except Exception as exc:
            errors = "analysis_failed"
            explanation = str(exc)
            suggestion = "suggestion could not be generated"

    if failure_type == "Contract-Guided":
        new_row["Generated Contract"] = get_contract_string(old_row)

    new_row[f"{failure_type} Code"] = code
    new_row[f"{failure_type} Failing Test"] = failure
    new_row[f"{failure_type} Error Type"] = errors
    new_row[f"{failure_type} Explanation"] = explanation
    new_row[f"{failure_type} Suggestion"] = suggestion


def get_contract_string(row):
    preconditions = clean_cell(row["Preconditions"])
    postconditions = clean_cell(row["Postconditions"])
    invariants = clean_cell(row["Invariants"])

    return "\n".join([
        f"Preconditions: {preconditions}",
        f"Postconditions: {postconditions}",
        f"Invariants: {invariants}",
    ])

def get_llm_analysis(prompt, code, failure, client, model):
    
    prompt = build_prompt(task_prompt=prompt, generated_code=code, failure_details=failure)

    raw_response, _ = call_chat_model(
        client=client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
        json_mode=True
    )

    result = parse_llm_response(raw_response)
    return ", ".join(result["failure_reasons"]), result["explanation"], result["minimal_fix_direction"]
    

def build_prompt(*, task_prompt, generated_code, failure_details):
    
    template = (Path(__file__).resolve().parent / "prompts" / "failure_analysis.txt").read_text(encoding="utf-8")
    
    return template.format(
        task_prompt=task_prompt,
        generated_code=generated_code,
        failure_details=failure_details
    )


def parse_llm_response(raw_response):
    
    parsed = extract_json_object(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response must be a JSON object")
    
    missing = REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Parsed response is missing keys: {sorted(missing)}")

    if not isinstance(parsed["failure_reasons"], list):
        raise ValueError("failure_reasons must be a list")

    return parsed


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("._-") or "x"


def clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value)


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)

    # normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # handle escaped newlines/tabs that may appear as literal "\n" or "\t"
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")

    # normalize real newlines/tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # collapse all whitespace
    text = " ".join(text.split())

    return text.strip()


def extract_failure_signature(value):
    text = normalize_text(value)

    fields = {}
    current_key = None

    for part in text.split(" # "):
        part = part.strip()
        if not part:
            continue

        if part.startswith("# "):
            part = part[2:].strip()

        if ": " in part:
            key, val = part.split(": ", 1)
            key = key.strip()
            val = val.strip()

            if key in {"failure_type", "input", "expected", "actual"}:
                fields[key] = val

    return (
        fields.get("failure_type", ""),
        fields.get("input", ""),
        fields.get("expected", ""),
        fields.get("actual", ""),
    )


def get_input_path(root, dataset, model):

    if dataset == "evalplus":
        fname = "EvalPlus_analysis.xlsx"
    elif dataset == "evalplus_mbpp":
        fname = "EvalPlus_MBPP_analysis.xlsx"
    else:
        fname = f"{dataset}_analysis.xlsx"

    return root / "analysis" / safe(model) / "basic" / fname


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    p.add_argument("--model")
    args = p.parse_args()

    root, dataset, provider = Path(args.root).resolve(), args.dataset, args.provider
    model = args.model or default_model(args.provider)
    client = get_client(provider)

    analyze(root, dataset, client, model)


if __name__ == "__main__":
    main()