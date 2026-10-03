# Contract-Guided LLM Code Generation

Code for **CG²**, a framework that uses structured software contracts as an intermediate representation between a natural-language programming task and its generated Python implementation.

The framework supports contract generation, supervised fine-tuning, contract verification and repair, code generation, and execution-based evaluation.

Contracts contain an **interface**, **preconditions**, **postconditions**, and **invariants**.

| Condition | Code-generation input | Contract source |
| --- | --- | --- |
| **Vanilla** | Original task and required interface | None |
| **Base contract** | Original task, interface, and contract | Base model |
| **SFT contract** | Original task, interface, and contract | Model with a contract-generation LoRA adapter |
| **Validated SFT** | Original task, interface, and selected contract | Verification and repair of SFT contracts |

Fine-tuning and repair operate on the intermediate contract. The downstream code-generating model remains unchanged across conditions. Generated solutions are evaluated separately using execution-based tests.

## Repository layout

```text
dataset/
    training_data/
        train.jsonl
        val.jsonl
    testing_data.jsonl
outputs/
    code/
    results/
    contracts/
    finetuning/
src/
README.md
requirements.txt
.gitignore
```

- `dataset/`: training, validation, and held-out evaluation data.
- `outputs/code/`: generated Python solutions and generation records.
- `outputs/results/`: execution-based evaluation summaries and task-level results.
- `outputs/contracts/`: base, SFT, and validated contract records.
- `outputs/finetuning/`: training artifacts and LoRA adapters.
- `src/`: dataset preparation, training, prompts, generation, verification, and evaluation.

Validated contracts are stored in a separate subdirectory of `outputs/contracts/`; they do not overwrite the original SFT contracts.

## Setup

Use Python 3.11 or newer and a PyTorch installation compatible with your GPU environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run all commands from the repository root.

The following Bash examples use Qwen14. Replace the model path with your local checkpoint:

```bash
MODEL_NAME=qwen14
MODEL_PATH=/path/to/base-model
GPU_ID=0
TASK_FILE=dataset/testing_data.jsonl
ADAPTER_PATH="outputs/finetuning/${MODEL_NAME}_contract_lora/final_adapter"
```

The evaluation dataset contains **1,007 held-out tasks**. Keep evaluation task IDs disjoint from training and validation task IDs.

## Preparing training data

If preparing a new SFT dataset from cleaned contract records:

```bash
python -m src.prepare_dataset.build_contract_sft --input-file /path/to/cleaned_contracts.jsonl --output-dir dataset/training_data --all-cleaned --val-ratio 0.1 --seed 42
```

This writes training and validation JSONL files. When reproducing an existing experiment, use its provided split.

## Fine-tuning

Train the Qwen14 contract-generation adapter:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.finetuning.train_qwen14 --model-path "$MODEL_PATH" --train-file dataset/training_data/train.jsonl --val-file dataset/training_data/val.jsonl --output-dir "outputs/finetuning/${MODEL_NAME}_contract_lora" --max-length 4096 --epochs 4 --learning-rate 2e-5 --batch-size 1 --grad-accum 16
```

The Qwen14 script saves its final adapter under `final_adapter/`.

Model-specific training scripts are available under `src/finetuning/`. Their command-line options and adapter output names may differ; inspect the selected script with `--help`.

## Generating contracts

Generate base-model contracts:

```bash
python -m src.generation.generate_contracts --task-file "$TASK_FILE" --output-dir "outputs/contracts/${MODEL_NAME}_base" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_base_contract" --gpu-id "$GPU_ID" --attention-implementation sdpa --max-output-tokens 1536 --max-generation-seconds 100
```

Generate SFT contracts:

```bash
python -m src.generation.generate_contracts --task-file "$TASK_FILE" --output-dir "outputs/contracts/${MODEL_NAME}_sft" --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" --model-name "${MODEL_NAME}_sft_contract" --gpu-id "$GPU_ID" --attention-implementation sdpa --max-output-tokens 1536 --max-generation-seconds 100
```

Generation records include task identifiers, generated contracts, and elapsed generation time.

## Verifying and repairing contracts

Verify SFT contracts against their original task statements:

```bash
python -m src.verification.verify_and_repair --task-file "$TASK_FILE" --contract-dir "outputs/contracts/${MODEL_NAME}_sft" --output-dir "outputs/contracts/validated/${MODEL_NAME}_sft" --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" --model-name "${MODEL_NAME}_sft_verifier" --gpu-id "$GPU_ID" --attention-implementation sdpa
```

The workflow records original verdicts, repair attempts, selected contracts, and unresolved cases. A repaired candidate is accepted only after fresh verification returns a `CORRECT` verdict.

## Generating code

Use the same unadapted code model for all four conditions.

Vanilla:

```bash
python -m src.generation.generate_vanilla_code --task-file "$TASK_FILE" --output-dir "outputs/code/${MODEL_NAME}_vanilla" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_vanilla" --gpu-id "$GPU_ID" --attention-implementation sdpa
```

Base-contract guidance:

```bash
python -m src.generation.generate_guided_code --task-file "$TASK_FILE" --contract-dir "outputs/contracts/${MODEL_NAME}_base" --output-dir "outputs/code/${MODEL_NAME}_base_guided" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_base_guided" --gpu-id "$GPU_ID" --attention-implementation sdpa
```

SFT-contract guidance:

```bash
python -m src.generation.generate_guided_code --task-file "$TASK_FILE" --contract-dir "outputs/contracts/${MODEL_NAME}_sft" --output-dir "outputs/code/${MODEL_NAME}_sft_guided" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_sft_guided" --gpu-id "$GPU_ID" --attention-implementation sdpa
```

Validated SFT guidance:

```bash
python -m src.generation.generate_guided_code --task-file "$TASK_FILE" --contract-dir "outputs/contracts/validated/${MODEL_NAME}_sft" --output-dir "outputs/code/${MODEL_NAME}_validated_guided" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_validated_guided" --gpu-id "$GPU_ID" --attention-implementation sdpa
```

## Extended workflows

Extended generation and verification use separate runtime implementations.

| Stage | Module |
| --- | --- |
| Contract generation | `src.generation.generate_contracts_extended` |
| Vanilla code generation | `src.generation.generate_vanilla_code_extended` |
| Guided code generation | `src.generation.generate_guided_code_extended` |
| Verification and repair | `src.verification.verify_and_repair_extended` |

Example extended contract-generation command:

```bash
python -m src.generation.generate_contracts_extended --task-file "$TASK_FILE" --output-dir "outputs/contracts/${MODEL_NAME}_sft_extended" --model-path "$MODEL_PATH" --adapter-path "$ADAPTER_PATH" --model-name "${MODEL_NAME}_sft_contract_extended" --gpu-id "$GPU_ID" --load-in-4bit --max-output-tokens 1536 --max-generation-seconds 100
```

Example extended guided-code command:

```bash
python -m src.generation.generate_guided_code_extended --task-file "$TASK_FILE" --contract-dir "outputs/contracts/${MODEL_NAME}_sft_extended" --output-dir "outputs/code/${MODEL_NAME}_sft_guided_extended" --model-path "$MODEL_PATH" --model-name "${MODEL_NAME}_sft_guided_extended" --gpu-id "$GPU_ID" --load-in-4bit --max-generation-seconds 100 --require-contract
```

Use a model and adapter supported by the selected backend. Extended training scripts may save their selected adapter under a different name; update `ADAPTER_PATH` accordingly.

## Running evaluation

Evaluate each condition using the same task file, test selection, and timeout.

Vanilla:

```bash
python -m src.generation.evaluate_code --task-file "$TASK_FILE" --code-dir "outputs/code/${MODEL_NAME}_vanilla" --results-dir "outputs/results/${MODEL_NAME}_vanilla" --workflow no_solution --model-name "${MODEL_NAME}_vanilla" --timeout 60 --workers 4
```

Base-contract guidance:

```bash
python -m src.generation.evaluate_code --task-file "$TASK_FILE" --code-dir "outputs/code/${MODEL_NAME}_base_guided" --results-dir "outputs/results/${MODEL_NAME}_base_guided" --workflow no_solution --model-name "${MODEL_NAME}_base_guided" --timeout 60 --workers 4
```

SFT-contract guidance:

```bash
python -m src.generation.evaluate_code --task-file "$TASK_FILE" --code-dir "outputs/code/${MODEL_NAME}_sft_guided" --results-dir "outputs/results/${MODEL_NAME}_sft_guided" --workflow no_solution --model-name "${MODEL_NAME}_sft_guided" --timeout 60 --workers 4
```

Validated SFT guidance:

```bash
python -m src.generation.evaluate_code --task-file "$TASK_FILE" --code-dir "outputs/code/${MODEL_NAME}_validated_guided" --results-dir "outputs/results/${MODEL_NAME}_validated_guided" --workflow no_solution --model-name "${MODEL_NAME}_validated_guided" --timeout 60 --workers 4
```

The evaluator writes `summary.json` and `details.json`. The timeout is specified in seconds per test.

## Task ranges and resuming

Use `--start` and `--count` to select a task range. `--start` is a zero-based position in the task file.

For example, `--start 200 --count 100` selects positions 200 through 299.

Generation scripts reuse existing task records unless `--overwrite` is supplied. Shards for the same model and condition share one output directory.

## Checks

Check Python syntax:

```bash
python -m compileall -q src
```

Check verification imports:

```bash
python -c 'import src.verification.contract_verifier; import src.verification.contract_repair; import src.verification.verify_and_repair; import src.verification.verify_and_repair_extended'
```

Check evaluator dependencies:

```bash
python -c 'import src.generation.testing_util'
```

The evaluator requires `testing_util.py` and `pyext2.py` under `src/generation/`.

## Citation

If you use this code, please cite the accompanying paper. Bibliographic details will be added when available.