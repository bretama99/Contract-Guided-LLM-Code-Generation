# Run commands for Qwen and OpenRouter GPT-3.5

All commands use explicit provider/model values. No shell variables are required.

## 0. Data preparation

```bash
python scripts/prepare_humaneval.py
python scripts/prepare_bigcodebench.py
python scripts/prepare_evalplus.py --evalplus-dataset humaneval
python scripts/prepare_evalplus.py --evalplus-dataset mbpp
python scripts/prepare_livecodebench.py
```

# Qwen

## HumanEval

```bash
python -m src.vanilla.generate \
  --dataset humaneval \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python -m src.vanilla.evaluate \
  --dataset humaneval \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --retry-timeout 45 \
  --workers 1 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset humaneval \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset humaneval \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset humaneval \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15
```

## BigCodeBench

```bash
python -m src.vanilla.generate \
  --dataset bigcodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python -m src.vanilla.evaluate \
  --dataset bigcodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --retry-timeout 45 \
  --workers 1 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset bigcodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15
```

## EvalPlus HumanEval+

```bash
python -m src.vanilla.generate \
  --dataset evalplus \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method raw_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --parallel 4
```

## EvalPlus MBPP+

```bash
python -m src.vanilla.generate \
  --dataset evalplus_mbpp \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus_mbpp \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus_mbpp \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method raw_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --parallel 4
```

## LiveCodeBench

```bash
python -m src.vanilla.generate \
  --dataset livecodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset livecodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 2.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset livecodebench \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 2.0 \
  --overwrite
```

## ClassEval vanilla

```bash
python -m src.classeval.generate_vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.evaluate_vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --overwrite
```

## ClassEval v1 contract-guided

```bash
python -m src.classeval.generate_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.generate_from_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --contract-provider qwen \
  --contract-model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --overwrite
```

## ClassEval Stage 2R feedback-refined contract

```bash
python -m src.classeval.rl_generate_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --base-provider qwen \
  --base-model qwen2.5-32b-instruct \
  --temperature 0.2 \
  --max-tokens 4096 \
  --overwrite

python -m src.classeval.generate_from_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --contract-provider qwen \
  --contract-model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided_optimized \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --overwrite
```

## Optional ClassEval policy learning after v2 exists

```bash
python -m src.classeval.rl_build_contract_preferences \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --min-gap 0.01 \
  --overwrite

python -m src.classeval.rl_optimize_contract_policy \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --overwrite
```

## Optional ClassEval policy-guided refinement

```bash
python -m src.classeval.rl_generate_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --base-provider qwen \
  --base-model qwen2.5-32b-instruct \
  --temperature 0.2 \
  --max-tokens 4096 \
  --use-policy \
  --overwrite

python -m src.classeval.generate_from_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --contract-provider qwen \
  --contract-model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided_optimized \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --overwrite
```

# OpenRouter GPT-3.5

## HumanEval

```bash
python -m src.vanilla.generate \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python -m src.vanilla.evaluate \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15 \
  --retry-timeout 45 \
  --workers 1 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15
```

## BigCodeBench

```bash
python -m src.vanilla.generate \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python -m src.vanilla.evaluate \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15 \
  --retry-timeout 45 \
  --workers 1 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15
```

## EvalPlus HumanEval+

```bash
python -m src.vanilla.generate \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method vanilla \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4
```

## EvalPlus MBPP+

```bash
python -m src.vanilla.generate \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method vanilla \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4
```

## LiveCodeBench

```bash
python -m src.vanilla.generate \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0 \
  --overwrite

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0 \
  --overwrite
```

## ClassEval vanilla

```bash
python -m src.classeval.generate_vanilla \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.evaluate_vanilla \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --overwrite
```

## ClassEval v1 contract-guided

```bash
python -m src.classeval.generate_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.generate_from_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --contract-provider openrouter \
  --contract-model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15 \
  --overwrite
```

## ClassEval Stage 2R feedback-refined contract

```bash
python -m src.classeval.rl_generate_optimized_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --base-provider openrouter \
  --base-model openai/gpt-3.5-turbo \
  --temperature 0.2 \
  --max-tokens 4096 \
  --overwrite

python -m src.classeval.generate_from_optimized_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --contract-provider openrouter \
  --contract-model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided_optimized \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15 \
  --overwrite
```

## Optional ClassEval policy learning after v2 exists

```bash
python -m src.classeval.rl_build_contract_preferences \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --min-gap 0.01 \
  --overwrite

python -m src.classeval.rl_optimize_contract_policy \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --overwrite
```

## Optional ClassEval policy-guided refinement

```bash
python -m src.classeval.rl_generate_optimized_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --base-provider openrouter \
  --base-model openai/gpt-3.5-turbo \
  --temperature 0.2 \
  --max-tokens 4096 \
  --use-policy \
  --overwrite

python -m src.classeval.generate_from_optimized_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --contract-provider openrouter \
  --contract-model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided_optimized \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15 \
  --overwrite
```

python -m src.classeval.rl_generate_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --base-provider qwen \
  --base-model qwen2.5-32b-instruct \
  --temperature 0.2 \
  --max-tokens 4096 \
  --overwrite

python -m src.classeval.generate_from_optimized_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --contract-provider qwen \
  --contract-model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --overwrite

python -m src.classeval.evaluate_contract_guided_optimized \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --overwrite

python -m src.classeval.rl_build_contract_preferences \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --min-gap 0.01 \
  --overwrite

python -m src.classeval.rl_optimize_contract_policy \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --overwrite