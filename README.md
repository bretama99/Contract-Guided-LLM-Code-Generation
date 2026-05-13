python scripts/prepare_humaneval.py
python scripts/prepare_bigcodebench.py
python scripts/prepare_evalplus.py --evalplus-dataset humaneval
python scripts/prepare_evalplus.py --evalplus-dataset mbpp
python scripts/prepare_livecodebench.py


python -m src.vanilla.generate \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt3.5-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.vanilla.evaluate \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-4o-mini \
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
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --timeout 15


python -m src.vanilla.generate \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.vanilla.evaluate \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --timeout 15 \
  --retry-timeout 45 \
  --workers 1 \
  --overwrite

python -m src.contract_synthesis.generate_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --timeout 15


python -m src.vanilla.generate \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method vanilla \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --parallel 4

python -m src.vanilla.generate \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method vanilla \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --parallel 4

python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --parallel 4

python -m src.vanilla.generate \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_synthesis.generate_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

