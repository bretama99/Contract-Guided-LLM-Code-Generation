# 0. Compile touched files
python -m py_compile src/contract_synthesis/generate_contracts.py
python -m py_compile src/contract_guided_generation/generate_from_raw_contracts.py
python -m py_compile src/evalplus_integration/evaluate_evalplus.py
python -m py_compile src/evalplus_integration/paths.py
python -m py_compile src/evalplus_integration/native_runner.py
python -m py_compile src/rl_method_level/rl_generate_contract.py

# 1. Generate raw/v1 contracts
python -m src.contract_synthesis.generate_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --temperature 0.0 \
  --overwrite

# 2. Generate code from raw/v1 contracts
python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source raw \
  --temperature 0.0 \
  --overwrite

# 3. Evaluate raw/v1 contract-guided code
python -m src.evalplus_integration.evaluate_evalplus \
  --dataset evalplus \
  --method raw_contracts \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --parallel 4

# 4. Generate optimized/RL contracts from raw evaluation feedback
python -m src.rl_method_level.rl_generate_contract \
  --dataset evalplus \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --selection epsilon_greedy \
  --epsilon 0.2 \
  --temperature 0.2 \
  --overwrite

# 5. Generate code from optimized/RL contracts
python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset evalplus \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source optimized_rl \
  --temperature 0.0 \
  --overwrite

# 6. Evaluate optimized/RL contract-guided code
python -m src.evalplus_integration.evaluate_evalplus \
  --dataset evalplus \
  --method optimized_rl \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --parallel 4

# 0. Compile touched files
python -m py_compile src/contract_synthesis/generate_contracts.py
python -m py_compile src/contract_guided_generation/generate_from_raw_contracts.py
python -m py_compile src/contract_guided_generation/evaluate_raw_contracts.py
python -m py_compile src/rl_method_level/rl_generate_contract.py

# 1. Generate raw/v1 contracts
python -m src.contract_synthesis.generate_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --temperature 0.0 \
  --overwrite

# 2. Generate code from raw/v1 contracts
python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source raw \
  --temperature 0.0 \
  --overwrite

# 3. Evaluate raw/v1 contract-guided code
python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source raw \
  --timeout 15

# 4. Generate optimized/RL contracts from raw evaluation feedback
python -m src.rl_method_level.rl_generate_contract \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --selection epsilon_greedy \
  --epsilon 0.2 \
  --temperature 0.2 \
  --overwrite

# 5. Generate code from optimized/RL contracts
python -m src.contract_guided_generation.generate_from_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source optimized_rl \
  --temperature 0.0 \
  --overwrite

# 6. Evaluate optimized/RL contract-guided code
python -m src.contract_guided_generation.evaluate_raw_contracts \
  --dataset bigcodebench \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source optimized_rl \
  --timeout 15



python -m py_compile src/classeval/contract_clause_utils.py
python -m py_compile src/classeval/generate_contracts.py
python -m py_compile src/classeval/generate_from_contracts.py
python -m py_compile src/classeval/rl_generate_optimized_contracts.py
python -m py_compile src/classeval/evaluate_contract_guided.py

python -m src.classeval.generate_contracts \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.generate_from_contracts \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source raw \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.evaluate_contract_guided \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source raw \
  --timeout 15 \
  --overwrite

python -m src.classeval.rl_generate_optimized_contracts \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --temperature 0.2 \
  --epsilon 0.15 \
  --overwrite

python -m src.classeval.generate_from_contracts \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source optimized_rl \
  --temperature 0.0 \
  --overwrite

python -m src.classeval.evaluate_contract_guided \
  --provider openrouter \
  --model gpt-3.5-turbo \
  --contract-source optimized_rl \
  --timeout 15 \
  --overwrite