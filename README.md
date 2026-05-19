python scripts/prepare_humaneval.py
python scripts/prepare_bigcodebench.py
python scripts/prepare_evalplus.py --evalplus-dataset humaneval
python scripts/prepare_evalplus.py --evalplus-dataset mbpp
python scripts/prepare_livecodebench.py

python -m src.vanilla.generate \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

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
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset humaneval \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15
python -m src.vanilla.generate \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

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
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --timeout 15
python -m src.vanilla.generate \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

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
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4
python -m src.vanilla.generate \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

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
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus_mbpp \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python scripts/evaluate_evalplus.py \
  --dataset evalplus_mbpp \
  --method raw_contracts \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --parallel 4
python -m src.vanilla.generate \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0

python -m src.contract_synthesis.generate_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 3000 \
  --delay 1.0

python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset livecodebench \
  --provider openrouter \
  --model openai/gpt-3.5-turbo \
  --temperature 0.0 \
  --max-tokens 2048 \
  --delay 1.0




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
BigCodeBench
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

EvalPlus HumanEval+
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
EvalPlus MBPP+
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
LiveCodeBench
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

  analysis
  python analysis_tools/analyze_results_humaneval.py \
  --dataset evalplus \
  --provider openai \
  --model claude-3-sonnet

  pip uninstall -y datasets
pip install "datasets==2.21.0"

python scripts/prepare_livecodebench.py --limit 20

python -m src.livecodebench.generate_vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --count 20 \
  --overwrite

python -m src.livecodebench.evaluate_vanilla \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --count 20 \
  --overwrite

python -m src.livecodebench.generate_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 4096 \
  --count 20 \
  --overwrite

python -m src.livecodebench.generate_from_contracts \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --temperature 0.0 \
  --max-tokens 8192 \
  --count 20 \
  --overwrite

python -m src.livecodebench.evaluate_contract_guided \
  --provider qwen \
  --model qwen2.5-32b-instruct \
  --timeout 15 \
  --count 20 \
  --overwrite

docker run --rm -it `                         
>>   --env-file .env `                                                                                                 
>>   -v "${PWD}:/workspace" `
>>   -w /workspace `
>>   thesis-eval `
>>   bash

